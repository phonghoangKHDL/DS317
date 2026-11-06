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


def audit_results(run_paths, reference=None, allow_seed_extension=False):
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
    if allow_seed_extension:
        _validate_seed_extension(paths, manifests)
    else:
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
                require(
                    first_kind == current_kind, "Backbones/protocol families differ"
                )
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


def export_proof(run_paths, output_root, reference=None, allow_seed_extension=False):
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
        {
            "runs": input_names,
            "reference": reference,
            "training_performed": False,
            "allow_seed_extension": allow_seed_extension,
        },
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
        report = baf.audit_results(
            runs, reference=config["reference"],
            allow_seed_extension=config.get("allow_seed_extension", False),
        )
        print("Verified prediction files:", len(report.prediction_audit))
        print(report.summary.to_string(index=False))
        if config.get("allow_seed_extension", False):
            print("OPERATING POINT SUMMARY: sample std across seeds")
            print(baf.summarize_operating_points(report.metrics).to_string(index=False))
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
    if allow_seed_extension:
        summarize_operating_points(report.metrics).to_csv(
            output / "operating_points.csv", index=False
        )
        write_json(
            output / "seed_extension.json",
            {
                "seeds": sorted(int(seed) for seed in report.metrics.seed.unique()),
                "compatibility": "same training source except label to_numpy copy=True; audit/export changes only",
                "training_performed_by_export": False,
            },
        )
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


def _training_signature(source_directory):
    """Compare training code while allowing the explicit read-only-label fix.

    baf code before audit_results contains all training/data/metric functions.
    Only audit/export code may differ after that boundary. In gru, only the
    ownership of an otherwise identical NumPy array may differ.
    """
    source_directory = Path(source_directory)
    common = (source_directory / "baf.py").read_text(encoding="utf-8")
    require("\ndef audit_results(" in common, "Unknown BAF source layout")
    neural = (source_directory / "gru.py").read_text(encoding="utf-8")
    neural = neural.replace(
        "to_numpy(dtype=np.int64, copy=True)", "to_numpy(dtype=np.int64)"
    )
    return common.split("\ndef audit_results(", 1)[0], neural


def _validate_seed_extension(paths, manifests):
    """Permit disjoint GRU seeds; reject environment or recipe changes."""
    first = manifests[0]
    kind, first_config, _ = _normalize_manifest(first)
    require(kind == "gru" and first_config["mode"] == "full", "Full GRU runs required")
    require(
        set(first_config["methods"]) == {"none", "balanced"}, "Paired losses required"
    )
    signature = _training_signature(paths[0] / "evidence/src")
    preprocessing = json.loads((paths[0] / "evidence/preprocessing.json").read_text())
    split = (
        pd.read_csv(paths[0] / "evidence/split_indices.csv.gz")
        .sort_values(["partition", "row_index"])
        .reset_index(drop=True)
    )
    seeds = set()
    for path, manifest in zip(paths, manifests):
        current_kind, config, _ = _normalize_manifest(manifest)
        require(current_kind == kind, "Different model families")
        for key in [
            "data_sha256",
            "data_rows",
            "dataset_tag",
            "partitions",
            "packages",
            "python",
            "device",
            "gpu",
            "torch_cuda",
            "preprocessing_note",
            "author_commit",
            "scope",
            "holdout_history",
        ]:
            require(manifest.get(key) == first.get(key), f"Runs differ in {key}")
        require(
            {k: v for k, v in config.items() if k != "seeds"}
            == {k: v for k, v in first_config.items() if k != "seeds"},
            "Training configs differ beyond seeds",
        )
        current_seeds = set(config["seeds"])
        require(not seeds.intersection(current_seeds), "Seed appears in multiple runs")
        seeds.update(current_seeds)
        require(
            _training_signature(path / "evidence/src") == signature,
            "Training source differs beyond the label copy fix",
        )
        require(
            json.loads((path / "evidence/preprocessing.json").read_text())
            == preprocessing,
            "Preprocessing parameters differ",
        )
        current_split = (
            pd.read_csv(path / "evidence/split_indices.csv.gz")
            .sort_values(["partition", "row_index"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(split, current_split)


def summarize_operating_points(metrics):
    """Means and sample std, with frozen validation thresholds named explicitly."""
    columns = [
        "ap",
        "roc_auc",
        "recall_at_0p5",
        "precision_at_0p5",
        "recall_frozen_threshold",
        "fpr_frozen_threshold",
        "precision_frozen_threshold",
        "fit_seconds",
    ]
    result = metrics.groupby(["method", "partition"])[columns].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    result = result.reset_index()
    counts = metrics.groupby(["method", "partition"]).seed.nunique().rename("n_seeds")
    return result.merge(counts.reset_index(), on=["method", "partition"])


# Class-weight selection uses the temporal protocol from notebooks 01-03.
# Existing training, audit and GRU APIs above are intentionally preserved.

WEIGHTING_METHODS = ("none", "balanced", "tuned")
BASE_DATA_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"


@dataclass(frozen=True)
class WeightingConfig:
    candidates: int = 24
    plan_seed: int = 9026
    search_seed: int = 42
    seeds: tuple = (42, 317, 2026)
    threads: int = 4
    max_trees: int = 1000
    max_fpr: float = 0.05
    powers: tuple = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    heartbeat_seconds: float = 15.0
    expected_rows: int = 1_000_000
    expected_data_sha256: str = BASE_DATA_SHA256
    dataset_tag: str = "BAF_Base"

    def validate(self):
        require(self.candidates >= len(self.powers), "Cover every weight power")
        require(self.threads > 0 and self.max_trees >= 20, "Invalid fit budget")
        require(0 < self.max_fpr < 1, "max_fpr must lie between zero and one")
        require(self.heartbeat_seconds > 0, "Heartbeat must be positive")
        require(len(set(self.seeds)) == len(self.seeds), "Duplicate seeds")
        require(len(self.seeds) >= 2, "Use at least two evaluation seeds")
        require(self.search_seed in self.seeds, "Include the HPO seed")
        require({0.0, 1.0}.issubset(self.powers), "Include none and balanced")
        require(all(0 <= a <= 2 for a in self.powers), "Invalid weight power")
        require(self.expected_rows > 0, "Expected rows must be positive")
        if self.dataset_tag == "BAF_Base":
            require(self.expected_rows == 1_000_000, "Base requires full data")
            require(self.expected_data_sha256 == BASE_DATA_SHA256, "Wrong Base SHA")

    def to_dict(self):
        return json.loads(json.dumps(self.__dict__))


def _weight_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _weight_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending." + uuid.uuid4().hex)
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def class_weight_values(counts, power):
    """Return mean-one class weights; powers zero/one match none/balanced."""
    negative, positive = map(int, counts)
    require(negative > 0 and positive > 0, "Both train classes are required")
    total = negative + positive
    if power == 0:
        return [1.0, 1.0]
    if power == 1:
        return [total / (2 * negative), total / (2 * positive)]
    ratio = (negative / positive) ** float(power)
    weight_zero = total / (negative + positive * ratio)
    return [weight_zero, weight_zero * ratio]


def power_sample_weights(labels, power):
    labels = np.asarray(labels, dtype=np.int64)
    require(set(np.unique(labels)) == {0, 1}, "Binary labels required")
    values = class_weight_values(np.bincount(labels, minlength=2), power)
    return np.asarray(values, dtype=float)[labels]


def weighting_candidate_plan(config):
    """Same model candidates and tree budget for all three methods."""
    config.validate()
    rng = np.random.RandomState(config.plan_seed)
    powers = np.resize(np.asarray(config.powers), config.candidates)
    powers = rng.permutation(powers)
    tree_counts = sorted({min(n, config.max_trees) for n in [150, 300, 600, 1000]})
    plan = []
    for index in range(config.candidates):
        parameters = TREE_PARAMETERS.copy()
        if index == 0:
            parameters["n_estimators"] = min(300, config.max_trees)
        else:
            depth = int(rng.choice([4, 6, 8, -1]))
            leaves = int(rng.choice([15, 31, 63]))
            parameters.update(
                n_estimators=int(rng.choice(tree_counts)),
                max_depth=depth,
                num_leaves=min(leaves, 2**depth) if depth > 0 else leaves,
                learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
                min_child_samples=int(rng.randint(30, 201)),
                subsample=float(rng.uniform(0.75, 1)),
                colsample_bytree=float(rng.uniform(0.7, 1)),
                reg_lambda=float(np.exp(rng.uniform(np.log(0.01), np.log(10)))),
            )
        plan.append(
            {
                "candidate": index,
                "parameters": parameters,
                "power": float(powers[index]),
            }
        )
    return plan


def _method_power(method, candidate):
    return {"none": 0.0, "balanced": 1.0, "tuned": candidate["power"]}[method]


def _weight_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    binary = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, binary, labels=[0, 1]).ravel()
    return {
        "ap": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "fpr": float(fp / (fp + tn)),
        "alert_rate": float(binary.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class _WeightProgress:
    """A heartbeat also covers histogram construction before the first tree."""

    order = 20
    before_iteration = False

    def __init__(self, label, total, interval):
        self.label, self.total, self.interval = label, total, interval
        self.iteration = 0
        self.stop = threading.Event()

    def __enter__(self):
        self.started = time.perf_counter()
        print(f"FIT {self.label}: requested trees={self.total}", flush=True)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            elapsed = time.perf_counter() - self.started
            print(
                f"HEARTBEAT {self.label}: trees={self.iteration}/{self.total}, "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    def __call__(self, environment):
        self.iteration = environment.iteration + 1

    def __exit__(self, *error):
        self.stop.set()
        self.thread.join()


def _weight_event(root, event, **extra):
    path = Path(root) / "events.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "time": time.time(), **extra}) + "\n")


def _weight_protocol(data_path, config):
    config.validate()
    require(sha256(data_path) == config.expected_data_sha256, "Data SHA mismatch")
    return {
        "schema": "BAF_class_weight_study_v1",
        "config": config.to_dict(),
        "data_sha256": config.expected_data_sha256,
        "source_sha256": sha256(__file__),
        "python": platform.python_version(),
        "packages": package_versions(),
        "split": {"train": [0, 1, 2, 3, 4, 5], "validation": [6], "test": [7]},
        "primary_selection_metric": "validation_average_precision",
        "threshold_rule": "validation ROC <= max_fpr, then frozen; score >= threshold",
        "holdout_history": (
            "Month7 was seen in earlier experiments; " "exploratory results"
        ),
        "comparison": "Equal candidate count and requested trees; no early stopping",
    }


def prepare_weighting_run(data_path, output, config):
    """Bind resume to identical data, source, versions, budget and seed plan."""
    output = Path(output).resolve()
    protocol = _weight_protocol(data_path, config)
    path = output / "protocol.json"
    if path.exists():
        require(_weight_json(path) == protocol, "Resume protocol/source/env mismatch")
        require(
            sha256(output / "evidence/src/baf.py") == protocol["source_sha256"],
            "Snapshot source changed",
        )
        require(
            _weight_json(output / "candidate_plan.json")
            == weighting_candidate_plan(config),
            "Candidate plan changed",
        )
        return output
    require(not output.exists(), "Use a new run directory or a valid resume run")
    (output / "evidence/src").mkdir(parents=True)
    shutil.copy2(__file__, output / "evidence/src/baf.py")
    _weight_write(path, protocol)
    _weight_write(output / "candidate_plan.json", weighting_candidate_plan(config))
    _weight_write(
        output / "request.json", {"data_path": str(Path(data_path).resolve())}
    )
    for name in ["fits", "models", "validation_scores", "test_scores", "test_results"]:
        (output / name).mkdir()
    _weight_event(output, "run_prepared")
    return output


def _weight_save_scores(path, scores):
    path = Path(path)
    pending = path.with_name(path.stem + ".pending." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(pending, score=np.asarray(scores, dtype=np.float64))
    pending.replace(path)


def _weight_scores(root, name):
    with np.load(Path(root) / name, allow_pickle=False) as saved:
        scores = saved["score"]
    require(scores.dtype == np.float64, "Scores must preserve float64")
    require(np.isfinite(scores).all(), "Nonfinite prediction")
    require(((scores >= 0) & (scores <= 1)).all(), "Scores outside [0,1]")
    return scores


def _weight_artifacts(root, record):
    for name, expected in record["artifacts"].items():
        path = (Path(root) / name).resolve()
        require(path.is_relative_to(Path(root).resolve()), "Artifact outside run")
        require(sha256(path) == expected, f"Changed artifact: {name}")


class WeightingStudy:
    """Search train/validation first; seal all thresholds before test access."""

    def __init__(self, data_path, output, config):
        self.root = prepare_weighting_run(data_path, output, config)
        self.config = config
        self.plan = weighting_candidate_plan(config)
        frame = load_data(data_path, config.expected_rows)
        self.parts = temporal_parts(frame)
        self.labels = self.parts["train"].fraud_bool.to_numpy(dtype=np.int64)
        self.val_labels = self.parts["validation"].fraud_bool.to_numpy(dtype=np.int64)
        self.counts = np.bincount(self.labels, minlength=2)
        self.train = clean_features(self.parts["train"])
        self.validation = clean_features(self.parts["validation"])
        self.vocabularies = {}
        for column in CATEGORICAL_COLUMNS:
            vocabulary = self.train[column].dropna().unique().tolist()
            self.vocabularies[column] = vocabulary
            dtype = pd.CategoricalDtype(vocabulary)
            self.train[column] = self.train[column].astype(dtype)
            self.validation[column] = self.validation[column].astype(dtype)
        self._save_preparation()

    def _save_preparation(self):
        preparation = {
            "partitions": profile(self.parts),
            "feature_order": self.train.columns.tolist(),
            "categorical_vocabularies": self.vocabularies,
            "missing_minus_one": MISSING_MINUS_ONE,
            "drop_features": ["fraud_bool", "month"],
            "negative_intended_balance": "missing",
        }
        path = self.root / "preprocessing.json"
        if path.exists():
            require(_weight_json(path) == preparation, "Preprocessing changed")
        else:
            _weight_write(path, preparation)
        tables = [
            pd.DataFrame(
                {
                    "partition": name,
                    "row_index": part.index,
                    "y_true": part.fraud_bool.to_numpy(),
                    "month": part.month.to_numpy(),
                }
            )
            for name, part in self.parts.items()
        ]
        splits = pd.concat(tables, ignore_index=True)
        path = self.root / "split_indices.csv.gz"
        if path.exists():
            require(pd.read_csv(path).equals(splits), "Split/order/labels changed")
        else:
            splits.to_csv(path, index=False)
        print(pd.DataFrame(preparation["partitions"]).T.to_string(), flush=True)

    def _fit(self, phase, method, candidate, seed):
        from lightgbm import LGBMClassifier
        import gzip

        key = f"{phase}_{method}_{candidate['candidate']:03d}_{seed}"
        path = self.root / "fits" / (key + ".json")
        power = _method_power(method, candidate)
        signature = {
            "phase": phase,
            "method": method,
            "candidate": candidate["candidate"],
            "seed": seed,
            "power": power,
            "parameters": candidate["parameters"],
        }
        if path.exists():
            row = _weight_json(path)
            require(row["signature"] == signature, "Cached fit signature mismatch")
            _weight_artifacts(self.root, row)
            scores = _weight_scores(self.root, row["validation_file"])
            require(len(scores) == len(self.val_labels), "Validation length mismatch")
            _weight_close(
                average_precision_score(self.val_labels, scores),
                row["validation_ap"],
                "Cached validation AP mismatch",
            )
            print("CACHE", key, flush=True)
            return row
        model = LGBMClassifier(
            **candidate["parameters"],
            random_state=int(seed),
            n_jobs=self.config.threads,
        )
        weights = power_sample_weights(self.labels, power)
        started = time.perf_counter()
        label = (
            f"{phase} {method} candidate={candidate['candidate'] + 1}/"
            f"{self.config.candidates} seed={seed}"
        )
        with MemoryMonitor() as memory:
            with _WeightProgress(
                label, model.n_estimators, self.config.heartbeat_seconds
            ) as progress:
                model.fit(
                    self.train,
                    self.labels,
                    sample_weight=weights,
                    categorical_feature=CATEGORICAL_COLUMNS,
                    callbacks=[progress],
                )
        seconds = time.perf_counter() - started
        scores = model.predict_proba(self.validation)[:, 1]
        model_file = "models/" + key + ".txt.gz"
        validation_file = "validation_scores/" + key + ".npz"
        with gzip.open(self.root / model_file, "wt", encoding="utf-8") as stream:
            stream.write(model.booster_.model_to_string())
        _weight_save_scores(self.root / validation_file, scores)
        row = {
            "key": key,
            "signature": signature,
            "fit_seconds": seconds,
            "peak_rss_mib": memory.peak_mib,
            "validation_ap": float(average_precision_score(self.val_labels, scores)),
            "validation_auc": float(roc_auc_score(self.val_labels, scores)),
            "class_weights": class_weight_values(self.counts, power),
            "mean_sample_weight": float(weights.mean()),
            "model_file": model_file,
            "validation_file": validation_file,
            "fitted_trees": model.booster_.num_trees(),
            "artifacts": {
                p: sha256(self.root / p) for p in [model_file, validation_file]
            },
        }
        _weight_write(path, row)
        _weight_event(self.root, "fit_completed", key=key)
        print(
            f"DONE {key}: validation AP={row['validation_ap']:.6f}, fit={seconds:.1f}s",
            flush=True,
        )
        del model, weights
        gc.collect()
        return row

    def search(self):
        results, selected = [], {}
        for method in WEIGHTING_METHODS:
            print(
                "SEARCH_METHOD",
                method,
                "candidates=",
                self.config.candidates,
                flush=True,
            )
            rows = [
                self._fit("hpo", method, candidate, self.config.search_seed)
                for candidate in self.plan
            ]
            winner = min(
                rows, key=lambda r: (-r["validation_ap"], r["signature"]["candidate"])
            )
            selected[method] = winner["key"]
            results.extend(rows)
        selection = {
            "primary_metric": "validation_ap",
            "tie_break": "lowest_candidate_index",
            "candidate_plan_sha256": sha256(self.root / "candidate_plan.json"),
            "selected": selected,
        }
        path = self.root / "selection.json"
        if path.exists():
            require(_weight_json(path) == selection, "Frozen selection changed")
        else:
            _weight_write(path, selection)
            _weight_event(self.root, "selection_frozen")
        return pd.DataFrame(
            [
                {
                    **r["signature"],
                    "validation_ap": r["validation_ap"],
                    "fit_seconds": r["fit_seconds"],
                }
                for r in results
            ]
        ).drop(columns="parameters")

    def freeze(self):
        selection = _weight_json(self.root / "selection.json")
        entries = []
        for method, key in selection["selected"].items():
            winner = _weight_json(self.root / "fits" / (key + ".json"))
            candidate = self.plan[winner["signature"]["candidate"]]
            for seed in self.config.seeds:
                row = (
                    winner
                    if seed == self.config.search_seed
                    else self._fit("final", method, candidate, seed)
                )
                _weight_artifacts(self.root, row)
                scores = _weight_scores(self.root, row["validation_file"])
                threshold = select_threshold(
                    self.val_labels, scores, self.config.max_fpr
                )
                values = _weight_metrics(self.val_labels, scores, threshold)
                require(
                    values["fpr"] <= self.config.max_fpr + 1e-12,
                    "Validation FPR exceeds cap",
                )
                entries.append(
                    {
                        "method": method,
                        "seed": seed,
                        "fit_key": row["key"],
                        "threshold": (
                            float(threshold) if np.isfinite(threshold) else "infinity"
                        ),
                        "validation_metrics": values,
                    }
                )
        plan = {
            "selection_sha256": sha256(self.root / "selection.json"),
            "preprocessing_sha256": sha256(self.root / "preprocessing.json"),
            "split_sha256": sha256(self.root / "split_indices.csv.gz"),
            "entries": entries,
            "fits": {
                e["fit_key"]: sha256(self.root / "fits" / (e["fit_key"] + ".json"))
                for e in entries
            },
        }
        path = self.root / "frozen_plan.json"
        if path.exists():
            require(
                _weight_json(path) == plan, "Threshold/model plan changed after freeze"
            )
            require(
                _weight_json(self.root / "freeze_seal.json")["sha256"] == sha256(path),
                "Freeze seal changed",
            )
        else:
            _weight_write(path, plan)
            _weight_write(self.root / "freeze_seal.json", {"sha256": sha256(path)})
            _weight_event(self.root, "thresholds_frozen")
        print("ALL_MODELS_AND_THRESHOLDS_FROZEN", len(entries), flush=True)
        return pd.DataFrame(entries).drop(columns="validation_metrics")

    def evaluate_test(self):
        import gzip
        from lightgbm import Booster

        plan = _weight_json(self.root / "frozen_plan.json")
        require(
            sha256(self.root / "frozen_plan.json")
            == _weight_json(self.root / "freeze_seal.json")["sha256"],
            "Refusing test access: freeze plan changed",
        )
        require(
            sha256(self.root / "selection.json") == plan["selection_sha256"],
            "Selection changed",
        )
        require(
            sha256(self.root / "preprocessing.json") == plan["preprocessing_sha256"],
            "Preprocessing changed",
        )
        require(
            sha256(self.root / "split_indices.csv.gz") == plan["split_sha256"],
            "Split changed",
        )
        for key, expected in plan["fits"].items():
            path = self.root / "fits" / (key + ".json")
            require(sha256(path) == expected, "Selected fit changed before test access")
            _weight_artifacts(self.root, _weight_json(path))
        test = clean_features(self.parts["test"])
        for column, vocabulary in self.vocabularies.items():
            test[column] = test[column].astype(pd.CategoricalDtype(vocabulary))
        labels = self.parts["test"].fraud_bool.to_numpy(dtype=np.int64)
        rows = []
        _weight_event(self.root, "test_evaluation_started")
        for entry in plan["entries"]:
            fit_path = self.root / "fits" / (entry["fit_key"] + ".json")
            require(
                sha256(fit_path) == plan["fits"][entry["fit_key"]],
                "Selected fit changed",
            )
            fit = _weight_json(fit_path)
            _weight_artifacts(self.root, fit)
            threshold = (
                float("inf") if entry["threshold"] == "infinity" else entry["threshold"]
            )
            stem = f"{entry['method']}_{entry['seed']}"
            path = self.root / "test_results" / (stem + ".json")
            if path.exists():
                record = _weight_json(path)
                require(
                    record["freeze_sha256"] == sha256(self.root / "frozen_plan.json"),
                    "Test uses different freeze",
                )
                _weight_artifacts(self.root, record)
                scores = _weight_scores(self.root, record["prediction_file"])
                require(
                    record["metrics"] == _weight_metrics(labels, scores, threshold),
                    "Cached test metric mismatch",
                )
            else:
                with gzip.open(
                    self.root / fit["model_file"], "rt", encoding="utf-8"
                ) as stream:
                    model = Booster(model_str=stream.read())
                scores = model.predict(test, num_threads=self.config.threads)
                prediction_file = "test_scores/" + stem + ".npz"
                _weight_save_scores(self.root / prediction_file, scores)
                record = {
                    "method": entry["method"],
                    "seed": entry["seed"],
                    "freeze_sha256": sha256(self.root / "frozen_plan.json"),
                    "prediction_file": prediction_file,
                    "artifacts": {prediction_file: sha256(self.root / prediction_file)},
                    "metrics": _weight_metrics(labels, scores, threshold),
                }
                _weight_write(path, record)
                del model
            for partition, values in [
                ("validation", entry["validation_metrics"]),
                ("test", record["metrics"]),
            ]:
                rows.append(
                    {
                        "method": entry["method"],
                        "seed": entry["seed"],
                        "partition": partition,
                        "power": fit["signature"]["power"],
                        "threshold": entry["threshold"],
                        "fit_seconds": fit["fit_seconds"],
                        **values,
                    }
                )
        pd.DataFrame(rows).to_csv(self.root / "metrics.csv", index=False)
        _weight_event(self.root, "test_evaluation_completed")
        print("WEIGHTING_STUDY_TEST_COMPLETE", len(plan["entries"]), flush=True)
        return pd.DataFrame(rows)


@dataclass
class WeightingReport:
    candidates: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    summary: pd.DataFrame
    deltas: pd.DataFrame


def _weight_close(left, right, message):
    require(np.isclose(left, right, rtol=1e-10, atol=1e-12), message)


def audit_weighting_study(output, verify_bundle=True):
    """Read scores and model text only. No model fitting or raw CSV required."""
    import gzip

    root = Path(output).resolve()
    protocol = _weight_json(root / "protocol.json")
    config = WeightingConfig(**protocol["config"])
    config.validate()
    require(protocol["schema"] == "BAF_class_weight_study_v1", "Wrong protocol")
    require(
        sha256(root / "evidence/src/baf.py") == protocol["source_sha256"],
        "Source changed",
    )
    require(
        _weight_json(root / "status.json")["state"] == "completed", "Incomplete run"
    )
    plan = weighting_candidate_plan(config)
    require(
        plan == _weight_json(root / "candidate_plan.json"), "Candidate plan changed"
    )
    splits = pd.read_csv(root / "split_indices.csv.gz")
    require(len(splits) == config.expected_rows, "Incomplete split")
    require(
        np.array_equal(np.sort(splits.row_index), np.arange(len(splits))), "Bad row IDs"
    )
    parts = {
        name: splits[splits.partition == name]
        for name in ["train", "validation", "test"]
    }
    require(parts["train"].month.lt(6).all(), "Training months changed")
    require(parts["validation"].month.eq(6).all(), "Validation month changed")
    require(parts["test"].month.eq(7).all(), "Test month changed")
    for part in parts.values():
        require(part.y_true.nunique() == 2, "Both classes required")
        require(part.row_index.is_monotonic_increasing, "Changed partition order")
    counts = np.bincount(parts["train"].y_true, minlength=2)
    labels = parts["validation"].y_true.to_numpy()
    records, candidate_rows = {}, []
    for path in sorted((root / "fits").glob("*.json")):
        row = _weight_json(path)
        signature = row["signature"]
        expected_key = (
            f"{signature['phase']}_{signature['method']}_"
            f"{signature['candidate']:03d}_{signature['seed']}"
        )
        require(row["key"] == expected_key, "Fit key/signature mismatch")
        candidate = plan[signature["candidate"]]
        require(signature["method"] in WEIGHTING_METHODS, "Unknown weighting method")
        require(
            signature["parameters"] == candidate["parameters"],
            "Model parameters changed",
        )
        power = _method_power(signature["method"], candidate)
        require(signature["power"] == power, "Weight power changed")
        require(row["fit_seconds"] > 0, "Missing fit timing")
        require(
            1 <= row["fitted_trees"] <= candidate["parameters"]["n_estimators"],
            "Tree count mismatch",
        )
        values = class_weight_values(counts, power)
        require(np.allclose(row["class_weights"], values, rtol=1e-12), "Wrong weights")
        _weight_close(row["mean_sample_weight"], 1.0, "Weights are not normalized")
        _weight_artifacts(root, row)
        scores = _weight_scores(root, row["validation_file"])
        require(scores.shape == labels.shape, "Validation score shape mismatch")
        ap = float(average_precision_score(labels, scores))
        _weight_close(ap, row["validation_ap"], "Validation AP changed")
        _weight_close(
            roc_auc_score(labels, scores),
            row["validation_auc"],
            "Validation AUC changed",
        )
        with gzip.open(root / row["model_file"], "rt", encoding="utf-8") as stream:
            model_text = stream.read()
        require(model_text.startswith("tree\n"), "Invalid model text")
        require(
            model_text.count("\nTree=") == row["fitted_trees"],
            "Checkpoint tree count mismatch",
        )
        require(path.stem == row["key"], "Fit record name mismatch")
        records[row["key"]] = row
        if signature["phase"] == "hpo":
            require(signature["seed"] == config.search_seed, "HPO seed changed")
            candidate_rows.append(
                {**signature, "validation_ap": ap, "fit_seconds": row["fit_seconds"]}
            )
        else:
            require(signature["phase"] == "final", "Unknown fit phase")
    expected_hpo = {
        f"hpo_{method}_{i:03d}_{config.search_seed}"
        for method in WEIGHTING_METHODS
        for i in range(config.candidates)
    }
    actual_hpo = {
        key for key, row in records.items() if row["signature"]["phase"] == "hpo"
    }
    require(actual_hpo == expected_hpo, "Unequal/incomplete HPO budgets")
    selection = _weight_json(root / "selection.json")
    require(
        selection["candidate_plan_sha256"] == sha256(root / "candidate_plan.json"),
        "Plan/selection mismatch",
    )
    selected_rows = []
    for method in WEIGHTING_METHODS:
        choices = [
            records[k]
            for k in actual_hpo
            if records[k]["signature"]["method"] == method
        ]
        winner = min(
            choices, key=lambda r: (-r["validation_ap"], r["signature"]["candidate"])
        )
        require(
            selection["selected"][method] == winner["key"],
            "Selection was not validation-only argmax",
        )
        selected_rows.append(
            {
                "method": method,
                "candidate": winner["signature"]["candidate"],
                "power": winner["signature"]["power"],
                "validation_ap": winner["validation_ap"],
            }
        )
    frozen = _weight_json(root / "frozen_plan.json")
    frozen_sha = sha256(root / "frozen_plan.json")
    require(
        _weight_json(root / "freeze_seal.json")["sha256"] == frozen_sha,
        "Freeze seal mismatch",
    )
    for file, key in [
        ("selection.json", "selection_sha256"),
        ("preprocessing.json", "preprocessing_sha256"),
        ("split_indices.csv.gz", "split_sha256"),
    ]:
        require(sha256(root / file) == frozen[key], "Frozen input changed: " + file)
    expected_pairs = {(m, s) for m in WEIGHTING_METHODS for s in config.seeds}
    pairs = [(e["method"], e["seed"]) for e in frozen["entries"]]
    require(
        set(pairs) == expected_pairs and len(pairs) == len(expected_pairs),
        "Incomplete seed plan",
    )
    rows, final_keys = [], set()
    for entry in frozen["entries"]:
        fit = records[entry["fit_key"]]
        signature = fit["signature"]
        winner = records[selection["selected"][entry["method"]]]
        require(signature["seed"] == entry["seed"], "Fit/seed mismatch")
        require(signature["method"] == entry["method"], "Fit/method mismatch")
        require(
            signature["candidate"] == winner["signature"]["candidate"],
            "Retuned evaluation seed",
        )
        if entry["seed"] == config.search_seed:
            require(
                fit["key"] == winner["key"], "Search seed must reuse the selected fit"
            )
        else:
            require(signature["phase"] == "final", "Missing extra seed fit")
            final_keys.add(fit["key"])
        require(
            sha256(root / "fits" / (fit["key"] + ".json"))
            == frozen["fits"][fit["key"]],
            "Frozen fit changed",
        )
        scores = _weight_scores(root, fit["validation_file"])
        threshold = select_threshold(labels, scores, config.max_fpr)
        expected_threshold = float(threshold) if np.isfinite(threshold) else "infinity"
        require(
            entry["threshold"] == expected_threshold,
            "Threshold was not selected on validation",
        )
        values = _weight_metrics(labels, scores, threshold)
        require(
            values == entry["validation_metrics"], "Frozen validation metric changed"
        )
        require(values["fpr"] <= config.max_fpr + 1e-12, "Validation FPR cap exceeded")
        stem = f"{entry['method']}_{entry['seed']}"
        result = _weight_json(root / "test_results" / (stem + ".json"))
        require(
            result["freeze_sha256"] == frozen_sha, "Test evaluated a different freeze"
        )
        _weight_artifacts(root, result)
        test_scores = _weight_scores(root, result["prediction_file"])
        require(len(test_scores) == len(parts["test"]), "Test length mismatch")
        test_values = _weight_metrics(parts["test"].y_true, test_scores, threshold)
        require(test_values == result["metrics"], "Test metric changed")
        for partition, metrics in [("validation", values), ("test", test_values)]:
            rows.append(
                {
                    "method": entry["method"],
                    "seed": entry["seed"],
                    "partition": partition,
                    "power": signature["power"],
                    "threshold": entry["threshold"],
                    "fit_seconds": fit["fit_seconds"],
                    **metrics,
                }
            )
    require(
        set(records) == expected_hpo | final_keys, "Unexpected/incomplete fitted models"
    )
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    phases = [e["event"] for e in events]
    require(
        phases.index("selection_frozen") < phases.index("thresholds_frozen"),
        "Invalid freeze order",
    )
    require(
        phases.index("thresholds_frozen") < phases.index("test_evaluation_started"),
        "Test accessed before freeze",
    )
    require("test_evaluation_completed" in phases, "Missing final evaluation")
    table = pd.DataFrame(rows)
    saved_table = pd.read_csv(root / "metrics.csv")
    require(len(saved_table) == len(table), "Incomplete metric table")
    for column in ["method", "seed", "partition"]:
        require(
            saved_table[column].tolist() == table[column].tolist(),
            "Metric labels changed",
        )
    for column in [
        "ap",
        "roc_auc",
        "recall",
        "precision",
        "fpr",
        "alert_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        require(
            np.allclose(saved_table[column], table[column], rtol=1e-10, atol=1e-12),
            "Metric CSV changed",
        )
    summary = table.groupby(["partition", "method"], as_index=False).agg(
        seeds=("seed", "count"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
    )
    test = table[table.partition == "test"]
    baseline = test[test.method == "none"].set_index("seed")
    deltas = []
    for row in test[test.method != "none"].itertuples():
        reference = baseline.loc[row.seed]
        deltas.append(
            {
                "method": row.method,
                "seed": row.seed,
                "delta_ap": row.ap - reference.ap,
                "delta_recall_pp": 100 * (row.recall - reference.recall),
                "delta_fpr_pp": 100 * (row.fpr - reference.fpr),
            }
        )
    if verify_bundle and (root / "sha256.json").exists():
        for name, expected in _weight_json(root / "sha256.json").items():
            require(sha256(root / name) == expected, "Bundle file changed: " + name)
    return WeightingReport(
        pd.DataFrame(candidate_rows).drop(columns="parameters"),
        pd.DataFrame(selected_rows),
        table,
        summary,
        pd.DataFrame(deltas),
    )


def execute_weighting_recipe(path):
    """Record real cell streams/displays in a fresh IPython process."""
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import display
    from IPython.utils.capture import capture_output

    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    shell.user_ns["display"] = display
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        stdout, stderr = io.StringIO(), io.StringIO()
        print("BEGIN_STAGE", count, flush=True)
        with capture_output(stdout=False, stderr=False, display=True) as rich:
            with contextlib.redirect_stdout(_Tee(sys.stdout, stdout)):
                with contextlib.redirect_stderr(_Tee(sys.stderr, stderr)):
                    result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = [
            nbformat.v4.new_output("stream", name=name, text=stream.getvalue())
            for name, stream in [("stdout", stdout), ("stderr", stderr)]
            if stream.getvalue()
        ]
        cell.outputs.extend(
            nbformat.v4.new_output("display_data", data=o.data, metadata=o.metadata)
            for o in rich.outputs
        )
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(error).__name__,
                    evalue=str(error),
                    traceback=traceback.format_exception(
                        type(error), error, error.__traceback__
                    ),
                )
            )
        nbformat.write(notebook, path)
        if error:
            raise RuntimeError(
                "Weighting stage failed; keep completed fits for resume"
            ) from error
        print("END_STAGE", count, flush=True)


def launch_weighting_study(data_path, output, config):
    """Run the notebook recipe with live output; resume completed fit records."""
    import os
    import subprocess
    import nbformat

    root = prepare_weighting_run(data_path, output, config)
    status_path = root / "status.json"
    if status_path.exists() and _weight_json(status_path)["state"] == "completed":
        print("Completed run retained:", root, flush=True)
        return root
    attempts = root / "evidence/executions"
    attempts.mkdir(exist_ok=True)
    number = len(list(attempts.glob("attempt_*_executed.ipynb"))) + 1
    recipe = attempts / f"attempt_{number:03d}_executed.ipynb"
    setup = textwrap.dedent(f"""
        from pathlib import Path
        import json
        import baf

        RUN_DIR = Path({str(root)!r})
        protocol = json.loads((RUN_DIR / "protocol.json").read_text(encoding="utf-8"))
        request = json.loads((RUN_DIR / "request.json").read_text(encoding="utf-8"))
        config = baf.WeightingConfig(**protocol["config"])
        study = baf.WeightingStudy(request["data_path"], RUN_DIR, config)
    """).strip()
    cells = [
        nbformat.v4.new_markdown_cell("# Class-weight study: actual execution outputs"),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell("candidates = study.search()\ndisplay(candidates)"),
        nbformat.v4.new_code_cell("frozen = study.freeze()\ndisplay(frozen)"),
        nbformat.v4.new_code_cell("metrics = study.evaluate_test()\ndisplay(metrics)"),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    nbformat.write(notebook, recipe)
    bootstrap = "\n".join(
        [
            "import sys",
            "sys.dont_write_bytecode = True",
            "sys.path.insert(0, " + repr(str(root / "evidence/src")) + ")",
            "import baf",
            "baf.execute_weighting_recipe(sys.argv[1])",
        ]
    )
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    status = {"state": "running", "attempt": number, "started": time.time()}
    _weight_write(status_path, status)
    process = None
    started = time.perf_counter()
    try:
        with (attempts / f"attempt_{number:03d}.log").open(
            "w", encoding="utf-8"
        ) as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", bootstrap, str(recipe)],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            require(process.wait() == 0, "Study failed; inspect execution notebook/log")
        status["state"] = "completed"
    except BaseException as error:
        status.update(state="failed_or_interrupted", error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(
            finished=time.time(), elapsed_seconds=time.perf_counter() - started
        )
        _weight_write(status_path, status)
        shutil.copy2(recipe, root / "study_executed.ipynb")
    print("WEIGHTING_STUDY_READY", root, flush=True)
    return root


def save_weighting_figures(report, output):
    from matplotlib import pyplot as plt

    table = report.summary[report.summary.partition == "test"].set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric, title in zip(
        axes,
        ["ap", "recall", "fpr"],
        ["Average precision", "Recall at frozen threshold", "FPR at frozen threshold"],
    ):
        values = table.loc[list(WEIGHTING_METHODS)]
        axis.bar(
            WEIGHTING_METHODS,
            values[metric + "_mean"],
            yerr=values[metric + "_std"].fillna(0),
            capsize=4,
            color=["#55718b", "#bd9158", "#4e8d80"],
        )
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Month 7: mean and sample SD across seeds; previously inspected test")
    fig.tight_layout()
    fig.savefig(Path(output) / "test_comparison.png", dpi=180)
    plt.close(fig)


def export_weighting_proof(output, proof_root):
    """Audit scores and package real execution outputs once; no autosave needed."""
    import nbformat

    root = Path(output).resolve()
    proof_root = Path(proof_root).resolve()
    require(not proof_root.is_relative_to(root), "Keep proof ZIP outside run")
    proof_root.mkdir(parents=True, exist_ok=True)
    archive = proof_root / (root.name + "_proof.zip")
    if (root / "sha256.json").exists():
        audit_weighting_study(root)
        if archive.exists():
            with zipfile.ZipFile(archive) as bundle:
                require(bundle.testzip() is None, "Existing proof ZIP is corrupt")
                require(
                    bundle.read(root.name + "/sha256.json")
                    == (root / "sha256.json").read_bytes(),
                    "Existing proof ZIP belongs to different files",
                )
            print("Existing verified proof:", archive)
            return archive
    source = textwrap.dedent("""
        from pathlib import Path
        import importlib.util
        import sys

        PROOF_DIR = None
        here = Path.cwd().resolve()
        root = Path(PROOF_DIR).resolve() if PROOF_DIR else next(
            (p for p in [here, *here.parents] if (p / "protocol.json").is_file()), None
        )
        if root is None:
            raise FileNotFoundError("Open the extracted proof folder or set PROOF_DIR")
        spec = importlib.util.spec_from_file_location(
            "baf_weighting_proof", root / "evidence/src/baf.py"
        )
        baf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baf
        spec.loader.exec_module(baf)

        report = baf.audit_weighting_study(root)
        print("CLASS_WEIGHT_PROOF_VERIFIED")
        print(report.selections.to_string(index=False))
        print(report.summary.to_string(index=False))
        print("PAIRED TEST DELTAS VS NONE")
        print(report.deltas.to_string(index=False))
        print("Test was previously inspected; sample SD is not a confidence interval.")
    """).strip()
    execution = source.replace("PROOF_DIR = None", "PROOF_DIR = " + repr(str(root)), 1)
    stdout = io.StringIO()
    namespace = {"__name__": "__audit__"}
    print("Verifying validation/test scores and checkpoints; no fitting.", flush=True)
    with contextlib.redirect_stdout(stdout):
        exec(compile(execution, "<weighting proof read-back>", "exec"), namespace)
    report = namespace["report"]
    for name in report.__dataclass_fields__:
        getattr(report, name).to_csv(root / (name + ".csv"), index=False)
    save_weighting_figures(report, root)
    verify = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Verify class-weight evidence\n\n"
                "No raw CSV/GPU/model fitting required."
            ),
            nbformat.v4.new_code_cell(
                source,
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=stdout.getvalue()
                    )
                ],
            ),
        ]
    )
    nbformat.write(verify, root / "verify.ipynb")
    _weight_write(
        root / "export_status.json", {"state": "completed", "training_performed": False}
    )
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != "sha256.json" and ".pending." not in p.name
    ]
    _weight_write(
        root / "sha256.json", {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    )
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*paths, root / "sha256.json"]:
            bundle.write(path, Path(root.name) / path.relative_to(root))
    print(stdout.getvalue())
    print("Proof ZIP:", archive)
    return archive
