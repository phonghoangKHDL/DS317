"""Reusable BAF data, classical models, result auditing and proof export.

No PyTorch import is needed to inspect results or export evidence.
The neural model and its training loop live together in gru.py.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import sys
import textwrap
import threading
import time
import traceback
import uuid
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

CATEGORICAL_COLUMNS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

MISSING_MINUS_ONE = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

BASE_COLUMNS = set(
    "fraud_bool income name_email_similarity prev_address_months_count "
    "current_address_months_count customer_age days_since_request "
    "intended_balcon_amount payment_type zip_count_4w velocity_6h "
    "velocity_24h velocity_4w bank_branch_count_8w "
    "date_of_birth_distinct_emails_4w employment_status credit_risk_score "
    "email_is_free housing_status phone_home_valid phone_mobile_valid "
    "bank_months_count has_other_cards proposed_credit_limit foreign_request "
    "source session_length_in_minutes device_os keep_alive_session "
    "device_distinct_emails_8w device_fraud_count month".split()
)

BASELINE_METHODS = ["lr_none", "lr_balanced", "lgbm_none", "lgbm_balanced"]
COST_METHODS = ["lgbm_cost_0.0114", "lgbm_cost_0.057", "lgbm_cost_0.285"]

TREE_PARAMETERS = {
    "objective": "binary",
    "class_weight": None,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def require(condition, message):
    """Validate even when Python is launched with optimization enabled."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def new_directory(parent, label):
    suffix = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return Path(parent) / f"{label}_{suffix}"


def package_versions(include_torch=False):
    names = ["numpy", "pandas", "scipy", "scikit-learn", "psutil", "lightgbm"]
    if include_torch:
        names.append("torch")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def load_data(path, expected_rows=1_000_000):
    """Read full Base.csv by default; explicit row override is for verification."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find BAF data: {path.resolve()}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Use a CSV or parquet file")
    validate_data(frame, expected_rows)
    return frame


def validate_data(frame, expected_rows=1_000_000):
    missing = sorted(BASE_COLUMNS - set(frame.columns))
    require(not missing, f"Missing BAF columns: {missing}")
    if expected_rows is not None:
        require(len(frame) == expected_rows, f"Expected {expected_rows:,} rows")
    require(frame.index.is_unique, "Row index must be unique")
    require(not frame.fraud_bool.isna().any(), "Labels contain missing values")
    require(set(frame.fraud_bool.unique()) == {0, 1}, "Both binary labels required")
    require(set(frame.month.unique()) == set(range(8)), "Expected months 0..7")


def temporal_parts(frame):
    parts = {
        "train": frame.loc[frame.month < 6],
        "validation": frame.loc[frame.month == 6],
        "test": frame.loc[frame.month == 7],
    }
    for name, part in parts.items():
        require(part.fraud_bool.nunique() == 2, f"Both classes required in {name}")
    return parts


def profile(parts):
    return {
        name: {
            "n": len(part),
            "positive": int(part.fraud_bool.sum()),
            "months": sorted(map(int, part.month.unique())),
        }
        for name, part in parts.items()
    }


def clean_features(frame):
    features = frame.drop(columns=["fraud_bool", "month"]).copy()
    for column in MISSING_MINUS_ONE:
        features[column] = features[column].mask(features[column] == -1)
    column = "intended_balcon_amount"
    features[column] = features[column].mask(features[column] < 0)
    for column in features.columns:
        if column not in CATEGORICAL_COLUMNS:
            features[column] = pd.to_numeric(features[column], errors="raise")
            features[column] = features[column].astype("float32")
            features[column] = features[column].replace([np.inf, -np.inf], np.nan)
    return features


def sample_weights(labels, method):
    labels = np.asarray(labels)
    if method.endswith("balanced"):
        return compute_sample_weight("balanced", labels).astype(float)
    if "_cost_" in method:
        negative_weight = float(method.rsplit("_", 1)[1])
        require(negative_weight > 0, "Cost weight must be positive")
        return np.where(labels == 0, negative_weight, 1.0)
    require(method.endswith("none"), f"Unknown weighting method: {method}")
    return np.ones(len(labels), dtype=float)


def select_threshold(labels, scores, max_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    allowed = np.flatnonzero(fpr <= max_fpr)
    best_recall = tpr[allowed].max()
    index = allowed[tpr[allowed] == best_recall][0]
    return float(thresholds[index])


def evaluate(labels, scores, threshold, max_fpr=0.05):
    """Keep source score dtype: float32 GRU ties must stay float32 on read-back."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = scores >= threshold
    fixed_predictions = scores > 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return {
        "ap": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall_at_fpr5_curve": float(tpr[fpr <= max_fpr].max()),
        "recall_frozen_threshold": float(tp / (tp + fn)),
        "fpr_frozen_threshold": float(fp / (fp + tn)),
        "precision_frozen_threshold": float(tp / (tp + fp)) if tp + fp else 0.0,
        "alert_rate_frozen_threshold": float(predictions.mean()),
        "recall_at_0p5": float(fixed_predictions[labels == 1].mean()),
        "precision_at_0p5": (
            float(labels[fixed_predictions].mean()) if fixed_predictions.any() else 0.0
        ),
        "f1_weighted_at_0p5": float(
            f1_score(labels, fixed_predictions, average="weighted")
        ),
        "accuracy_at_0p5": float((fixed_predictions == labels).mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class MemoryMonitor:
    """Sample RSS when available; keep an explicit warning otherwise."""

    def __enter__(self):
        self.peak = None
        self.warning = ""
        self.process = None
        self.thread = None
        self.stop = threading.Event()
        try:
            self.process = psutil.Process()
        except (psutil.Error, OSError) as error:
            self.warning = f"RSS unavailable: {error}"
        if self.process is not None and self._measure():
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    @property
    def peak_mib(self):
        return self.peak / 2**20 if self.peak is not None else None

    def _measure(self):
        try:
            current = self.process.memory_info().rss
            self.peak = current if self.peak is None else max(self.peak, current)
            return True
        except (psutil.Error, OSError) as error:
            self.warning = f"RSS sampling unavailable/incomplete: {error}"
            return False

    def _sample(self):
        while not self.stop.wait(0.1):
            if not self._measure():
                break

    def __exit__(self, *exception):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.process is not None:
            self._measure()


@contextlib.contextmanager
def experiment(output):
    """Create a new run, capture logs, and always leave an explicit status."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    (output / "predictions").mkdir()
    status = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with (output / "evidence/training.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            with contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    yield output
                    status["state"] = "completed"
                except BaseException:
                    status["state"] = "failed_or_interrupted"
                    traceback.print_exc()
                    raise
                finally:
                    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    write_json(output / "evidence/status.json", status)


def save_manifest(output, data_path, frame, kind, config, parts, **extra):
    sources = [Path(__file__).resolve()]
    if kind == "gru":
        sources.append(Path(__file__).resolve().with_name("gru.py"))
    source_dir = Path(output) / "evidence/src"
    source_dir.mkdir()
    for source in sources:
        shutil.copy2(source, source_dir / source.name)
    manifest = {
        "schema_version": 2,
        "kind": kind,
        "data_name": Path(data_path).name,
        "data_sha256": sha256(data_path),
        "data_rows": len(frame),
        "dataset_tag": config.get("dataset_tag", "BAF_Base"),
        "protocol": config.get("protocol", "temporal"),
        "mode": config.get("mode", "full"),
        "config": config,
        "partitions": profile(parts),
        "python": platform.python_version(),
        "packages": package_versions(include_torch=kind == "gru"),
        "source_hashes": {source.name: sha256(source) for source in sources},
        "test_evaluated": bool(
            set(config.get("evaluation_partitions", [])) & {"test", "random_test"}
        ),
        "holdout_history": "Earlier group notebooks inspected month 7; scores are exploratory",
        **extra,
    }
    write_json(Path(output) / "manifest.json", manifest)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(pd.DataFrame(manifest["partitions"]).T.to_string())
    return manifest


def save_predictions(output, method, seed, partition, frame, scores):
    path = Path(output) / "predictions" / f"{method}_{seed}_{partition}.csv.gz"
    pd.DataFrame(
        {
            "row_index": frame.index,
            "y_true": frame.fraud_bool.to_numpy(),
            "score": np.asarray(scores).astype("float64"),
        }
    ).to_csv(path, index=False)


def _fit_classical(method, seed, train, labels, weights, threads, tree_parameters):
    if method.startswith("lr_"):
        numeric = [column for column in train if column not in CATEGORICAL_COLUMNS]
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True),
                        StandardScaler(),
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                    CATEGORICAL_COLUMNS,
                ),
            ],
            sparse_threshold=1.0,
        )
        model = make_pipeline(
            preprocessor,
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        )
        model.fit(train, labels, logisticregression__sample_weight=weights)
        return model

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        **tree_parameters,
        random_state=seed,
        n_jobs=threads,
    )
    model.fit(
        train,
        labels,
        sample_weight=weights,
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return model


def train_classical(
    data_path,
    output,
    methods=BASELINE_METHODS,
    seeds=(42, 317, 2026),
    threads=4,
    tree_parameters=None,
    frame=None,
    expected_rows=1_000_000,
    dataset_tag="BAF_Base",
):
    require(bool(methods), "Choose at least one method")
    require(len(set(methods)) == len(methods), "Duplicate methods")
    require(
        all(m in BASELINE_METHODS + COST_METHODS for m in methods), "Unknown method"
    )
    require(bool(seeds) and len(set(seeds)) == len(seeds), "Unique seeds required")
    require(threads > 0, "threads must be positive")
    parameters = TREE_PARAMETERS | (tree_parameters or {})
    config = {
        "methods": list(methods),
        "seeds": list(seeds),
        "threads": threads,
        "tree_parameters": parameters,
        "protocol": "temporal",
        "mode": "full",
        "evaluation_partitions": ["validation"],
        "dataset_tag": dataset_tag,
    }
    rows = []
    with experiment(output):
        frame = load_data(data_path, expected_rows) if frame is None else frame
        validate_data(frame, expected_rows)
        parts = temporal_parts(frame)
        save_manifest(output, data_path, frame, "classical", config, parts)
        train = clean_features(parts["train"])
        validation = clean_features(parts["validation"])
        labels = parts["train"].fraud_bool.to_numpy()
        for method in methods:
            for seed in list(seeds)[:1] if method.startswith("lr_") else seeds:
                train_features = train.copy()
                val_features = validation.copy()
                if method.startswith("lgbm_"):
                    for column in CATEGORICAL_COLUMNS:
                        vocabulary = pd.Index(train[column].dropna().unique())
                        dtype = pd.CategoricalDtype(categories=vocabulary)
                        train_features[column] = train_features[column].astype(dtype)
                        val_features[column] = val_features[column].astype(dtype)
                weights = sample_weights(labels, method)
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with MemoryMonitor() as memory:
                        model = _fit_classical(
                            method,
                            seed,
                            train_features,
                            labels,
                            weights,
                            threads,
                            parameters,
                        )
                fit_seconds = time.perf_counter() - start
                scores = model.predict_proba(val_features)[:, 1]
                threshold = select_threshold(parts["validation"].fraud_bool, scores)
                row = {
                    "method": method,
                    "seed": seed,
                    "partition": "validation",
                    "threshold": threshold if np.isfinite(threshold) else "infinity",
                    "fit_seconds": fit_seconds,
                    "peak_rss_mib": memory.peak_mib,
                    "mean_weight": float(weights.mean()),
                    "weight_negative": float(weights[labels == 0][0]),
                    "weight_positive": float(weights[labels == 1][0]),
                    "warnings": " | ".join(str(item.message) for item in caught),
                    "memory_warning": memory.warning,
                    **evaluate(parts["validation"].fraud_bool, scores, threshold),
                }
                save_predictions(
                    output, method, seed, "validation", parts["validation"], scores
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(Path(output) / "metrics.csv", index=False)
                print(json.dumps(row), flush=True)
                del model, train_features, val_features
                gc.collect()
    return pd.DataFrame(rows)


def summarize(metrics):
    groups = [
        name for name in ["protocol", "mode", "method", "partition"] if name in metrics
    ]
    return metrics.groupby(groups, as_index=False).agg(
        n_runs=("seed", "count"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        recall_mean=("recall_at_fpr5_curve", "mean"),
        recall_std=("recall_at_fpr5_curve", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )


def _normalize_manifest(manifest):
    config = manifest.get("config", manifest.get("arguments", {}))
    kind = manifest.get("kind", "gru" if "embedding_dim" in config else "classical")
    parts = manifest["partitions"]
    if isinstance(parts, list):
        parts = {
            part["partition"]: {
                "n": part["n"],
                "positive": part["fraud"],
                "months": part.get("months", []),
            }
            for part in parts
        }
    return kind, config, parts


@dataclass
class Comparison:
    metrics: pd.DataFrame
    summary: pd.DataFrame
    prediction_audit: pd.DataFrame
    deltas: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    changed_cases: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name in self.__dataclass_fields__:
            getattr(self, name).to_csv(output / f"{name}.csv", index=False)
        return output


def audit_results(run_paths, reference=None):
    """Verify new runs and earlier ZIP results without importing their code."""
    paths = [Path(path).resolve() for path in run_paths]
    require(bool(paths) and len(set(paths)) == len(paths), "Unique run paths required")
    all_metrics = []
    audit_rows = []
    predictions = {}
    common_labels = {}
    manifests = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        kind, config, parts = _normalize_manifest(manifest)
        manifests.append(manifest)
        status_path = path / "evidence/status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            require(status["state"] == "completed", f"Incomplete run: {path}")
        for name, digest in manifest.get("source_hashes", {}).items():
            require(
                sha256(path / "evidence/src" / name) == digest,
                f"Source changed: {name}",
            )
        if kind == "gru" and "source_hashes" not in manifest:
            script = path / "evidence/baf_gru_step04.py"
            require(
                sha256(script) == manifest["script_sha256"], "Legacy GRU source changed"
            )
            model = path / "evidence/gru_author_model.py"
            require(
                sha256(model) == manifest["author_source"]["model_sha256"],
                "Legacy model changed",
            )
        split_path = path / "evidence/split_indices.csv.gz"
        split_indices = pd.read_csv(split_path) if split_path.is_file() else None
        if split_indices is not None:
            require(
                not split_indices.row_index.duplicated().any(),
                "Overlapping split row IDs",
            )
        scores = pd.read_csv(path / "metrics.csv")
        keys = ["method", "seed", "partition"]
        require(not scores.duplicated(keys).any(), f"Duplicate result rows: {path}")
        eval_parts = config.get("evaluation_partitions")
        if eval_parts is None:
            eval_parts = ["validation"]
            if kind == "gru" and config.get("protocol") == "author_fixed":
                eval_parts.append("random_test")
            elif config.get("evaluate_test"):
                eval_parts.append("test")
        expected = set()
        for method in config["methods"]:
            name = (
                f"gru_{method}"
                if kind == "gru" and not method.startswith("gru_")
                else method
            )
            seeds = config["seeds"][:1] if name.startswith("lr_") else config["seeds"]
            expected.update((name, seed, part) for seed in seeds for part in eval_parts)
        require(
            set(map(tuple, scores[keys].to_numpy())) == expected,
            f"Missing/extra results: {path}",
        )
        for row in scores.itertuples():
            key = (row.method, row.seed, row.partition)
            require(
                key not in predictions,
                f"Duplicate model/seed/partition across runs: {key}",
            )
            file = (
                path / "predictions" / f"{row.method}_{row.seed}_{row.partition}.csv.gz"
            )
            table = pd.read_csv(file).sort_values("row_index").reset_index(drop=True)
            require(
                not table.row_index.duplicated().any(), f"Duplicate row IDs: {file}"
            )
            require(not table.isna().any().any(), f"Missing prediction values: {file}")
            require(set(table.y_true.unique()) == {0, 1}, "Both labels required")
            require(
                np.isfinite(table.score).all() and table.score.between(0, 1).all(),
                "Invalid score",
            )
            require(
                len(table) == parts[row.partition]["n"], "Prediction count mismatch"
            )
            require(
                int(table.y_true.sum()) == parts[row.partition]["positive"],
                "Fraud count mismatch",
            )
            if split_indices is not None:
                expected_ids = split_indices.loc[
                    split_indices.partition == row.partition, "row_index"
                ]
                require(
                    set(expected_ids) == set(table.row_index), "Split row IDs mismatch"
                )
            labels = table[["row_index", "y_true"]]
            if row.partition in common_labels:
                pd.testing.assert_frame_equal(common_labels[row.partition], labels)
            common_labels[row.partition] = labels
            dtype = np.float32 if kind == "gru" else np.float64
            probabilities = table.score.to_numpy(dtype=dtype)
            threshold = float(row.threshold)
            if row.partition == "validation":
                expected_threshold = select_threshold(table.y_true, probabilities)
            else:
                val_row = scores[
                    (scores.method == row.method)
                    & (scores.seed == row.seed)
                    & (scores.partition == "validation")
                ].iloc[0]
                expected_threshold = float(val_row.threshold)
            require(
                np.isclose(threshold, expected_threshold), f"Threshold mismatch: {key}"
            )
            values = evaluate(table.y_true, probabilities, threshold)
            for name, value in values.items():
                if name in scores.columns:
                    require(
                        np.isclose(value, getattr(row, name), rtol=1e-7, atol=1e-8),
                        f"Metric mismatch: {key}, {name}",
                    )
            predictions[key] = (table, probabilities >= threshold)
            audit_rows.append(
                {
                    "method": row.method,
                    "seed": row.seed,
                    "partition": row.partition,
                    "sha256": sha256(file),
                    "verified": True,
                }
            )
        if kind == "gru":
            require(
                (scores.groupby("seed").initial_model_sha256.nunique() == 1).all(),
                "Different initial models across losses",
            )
            history = pd.read_csv(path / "epoch_history.csv")
            epochs = 1 if config["mode"] == "pilot" else config["epochs"]
            expected_epochs = {
                (method, seed, epoch)
                for method in config["methods"]
                for seed in config["seeds"]
                for epoch in range(1, epochs + 1)
            }
            require(
                not history.duplicated(["method", "seed", "epoch"]).any(),
                "Duplicate epochs",
            )
            require(
                set(map(tuple, history[["method", "seed", "epoch"]].to_numpy()))
                == expected_epochs,
                "Incomplete epoch history",
            )
            if config.get("save_models"):
                for method in config["methods"]:
                    for seed in config["seeds"]:
                        require(
                            (path / "models" / f"gru_{method}_{seed}.pt").is_file(),
                            "Missing checkpoint",
                        )
        all_metrics.append(scores.assign(source_run=path.name))
    if len(manifests) > 1:
        first = manifests[0]
        for current in manifests[1:]:
            for key in [
                "data_sha256",
                "partitions",
                "packages",
                "python",
                "source_hashes",
                "script_sha256",
            ]:
                require(first.get(key) == current.get(key), f"Runs differ in {key}")
            first_kind, first_config, _ = _normalize_manifest(first)
            current_kind, current_config, _ = _normalize_manifest(current)
            require(first_kind == current_kind, "Backbones/protocol families differ")
            for key in ["seeds", "threads", "protocol", "mode", "tree_parameters"]:
                require(
                    first_config.get(key) == current_config.get(key),
                    f"Configs differ in {key}",
                )
    metrics = pd.concat(all_metrics, ignore_index=True)
    result = Comparison(metrics, summarize(metrics), pd.DataFrame(audit_rows))
    if reference is not None:
        require(reference in set(metrics.method), f"Missing reference: {reference}")
        deltas = []
        overlap = []
        cases = []
        for key, (table, candidate) in predictions.items():
            method, seed, partition = key
            ref_key = (reference, seed, partition)
            if method == reference or ref_key not in predictions:
                continue
            ref_table, baseline = predictions[ref_key]
            pd.testing.assert_frame_equal(
                ref_table[["row_index", "y_true"]], table[["row_index", "y_true"]]
            )
            fraud = table.y_true.to_numpy() == 1
            gained = fraud & ~baseline & candidate
            lost = fraud & baseline & ~candidate
            baseline_metric = metrics[
                (metrics.method == reference)
                & (metrics.seed == seed)
                & (metrics.partition == partition)
            ].iloc[0]
            candidate_metric = metrics[
                (metrics.method == method)
                & (metrics.seed == seed)
                & (metrics.partition == partition)
            ].iloc[0]
            deltas.append(
                {
                    "reference": reference,
                    "method": method,
                    "seed": seed,
                    "partition": partition,
                    "delta_recall_pp": 100
                    * (
                        candidate_metric.recall_at_fpr5_curve
                        - baseline_metric.recall_at_fpr5_curve
                    ),
                    "delta_ap": candidate_metric.ap - baseline_metric.ap,
                }
            )
            overlap.append(
                {
                    "reference": reference,
                    "method": method,
                    "seed": seed,
                    "partition": partition,
                    "fraud_gained": int(gained.sum()),
                    "fraud_lost": int(lost.sum()),
                    "net_fraud": int(gained.sum() - lost.sum()),
                    "baseline_fp": int(baseline[~fraud].sum()),
                    "candidate_fp": int(candidate[~fraud].sum()),
                }
            )
            for index in np.flatnonzero(gained | lost):
                cases.append(
                    {
                        "reference": reference,
                        "method": method,
                        "seed": seed,
                        "partition": partition,
                        "row_index": int(table.row_index.iloc[index]),
                        "change": "gained" if gained[index] else "lost",
                    }
                )
        result.deltas = pd.DataFrame(deltas)
        result.overlap = pd.DataFrame(overlap)
        result.changed_cases = pd.DataFrame(cases)
    return result


def export_proof(run_paths, output_root, reference=None):
    """Create and execute one portable audit notebook; no frontend autosave."""
    paths = [Path(path).resolve() for path in run_paths]
    output = new_directory(output_root, "proof")
    require(
        all(not output.is_relative_to(path) for path in paths),
        "Proof output must be outside source run folders",
    )
    output.mkdir(parents=True, exist_ok=False)
    input_names = []
    for index, source in enumerate(paths, 1):
        name = f"run_{index:02d}"
        shutil.copytree(source, output / "inputs" / name)
        input_names.append(name)
    (output / "src").mkdir()
    shutil.copy2(__file__, output / "src/baf.py")
    write_json(
        output / "proof_config.json",
        {"runs": input_names, "reference": reference, "training_performed": False},
    )
    source = textwrap.dedent("""
        from pathlib import Path
        import json
        import sys

        PROOF_DIR = None  # Set to the extracted proof folder if auto-detection fails.
        here = Path.cwd().resolve()
        if PROOF_DIR is None:
            PROOF_DIR = next(
                (path for path in [here, *here.parents]
                 if (path / "proof_config.json").is_file()),
                None,
            )
        if PROOF_DIR is None:
            PROOF_DIR = Path(__FALLBACK__)
        else:
            PROOF_DIR = Path(PROOF_DIR).resolve()
        sys.path.insert(0, str(PROOF_DIR / "src"))
        import baf

        config = json.loads((PROOF_DIR / "proof_config.json").read_text())
        runs = [PROOF_DIR / "inputs" / name for name in config["runs"]]
        report = baf.audit_results(runs, reference=config["reference"])
        print("Verified prediction files:", len(report.prediction_audit))
        print(report.summary.to_string(index=False))
        if not report.deltas.empty:
            print("PAIRED SEED DELTAS")
            print(report.deltas.to_string(index=False))
            print("FRAUD OVERLAP")
            print(report.overlap.to_string(index=False))
        for run in runs:
            log = run / "evidence/training.log"
            if log.is_file():
                print("ORIGINAL TRAINING LOG:", run.name)
                print(log.read_text(encoding="utf-8"))
        """).strip().replace("__FALLBACK__", repr(str(output.resolve())))
    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace = {"__name__": "__proof_audit__"}
    error = None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(source, "<proof audit>", "exec"), namespace)
    except BaseException as exception:
        error = exception
    cell_outputs = []
    for name, stream in [("stdout", stdout), ("stderr", stderr)]:
        if stream.getvalue():
            cell_outputs.append(
                {"output_type": "stream", "name": name, "text": stream.getvalue()}
            )
    if error is not None:
        cell_outputs.append(
            {
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": traceback.format_exception(
                    type(error), error, error.__traceback__
                ),
            }
        )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "evidence": {
                "capture": "actual read-back execution, no model fitting",
                "frontend_notebook_snapshot": False,
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": "# BAF: independent result audit\n\nThis code was executed against packaged predictions. It is a read-back audit, not a copy of unsaved frontend output. Original logs, source and checkpoints are preserved in inputs. Run All to verify after extraction. Pilot scores are not full-data benchmark scores.",
            },
            {
                "cell_type": "code",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": source,
                "execution_count": 1,
                "outputs": cell_outputs,
            },
        ],
    }
    write_json(output / "verify.ipynb", notebook)
    write_json(
        output / "export_status.json",
        {"state": "failed" if error else "completed", "training_performed": False},
    )
    (output / "verification.log").write_text(
        stdout.getvalue() + stderr.getvalue(), encoding="utf-8"
    )
    if error is not None:
        raise error
    report = namespace["report"]
    for name in report.__dataclass_fields__:
        getattr(report, name).to_csv(output / f"{name}.csv", index=False)
    hashes = {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    write_json(output / "sha256.json", hashes)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in output.rglob("*"):
            if path.is_file():
                bundle.write(path, Path(output.name) / path.relative_to(output))
    print("Verified prediction files:", len(report.prediction_audit))
    print("Proof ZIP:", archive)
    return archive
