"""GRU model and training, kept together for reuse across protocols and runs.

Based on Objective over Architecture, Computation 2025, DOI 10.3390/computation13120290.
Author code commit: 6e8252cb9080378905b2aae30033c5946b72335b.
The model class is renamed; parameter names and forward computation are preserved.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gc
import hashlib
import json
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import baf

AUTHOR_NUMERIC = [
    "income",
    "name_email_similarity",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "bank_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
]

AUTHOR_CATEGORICAL = [
    "payment_type",
    "employment_status",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "source",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
]


@dataclass
class Config:
    protocol: str = "author_fixed"
    mode: str = "pilot"
    methods: list[str] = field(default_factory=lambda: ["none", "balanced"])
    seeds: list[int] = field(default_factory=lambda: [1234])
    threads: int = 4
    device: str = "auto"
    embedding_dim: int = 186
    hidden_dim: int = 733
    learning_rate: float = 0.000448103301026688
    epochs: int = 8
    batch_size: int = 256
    dropout: float = 0.3
    pilot_train: int = 8192
    pilot_eval: int = 4096
    save_models: bool = True
    dataset_tag: str = "BAF_Base"

    def validate(self):
        baf.require(self.protocol in ["author_fixed", "temporal"], "Unknown protocol")
        baf.require(self.mode in ["pilot", "full"], "Unknown run mode")
        baf.require(
            set(self.methods) == {"none", "balanced"} and len(self.methods) == 2,
            "Use paired none/balanced methods",
        )
        baf.require(
            bool(self.seeds) and len(set(self.seeds)) == len(self.seeds),
            "Unique seeds required",
        )
        baf.require(
            min(
                self.threads,
                self.batch_size,
                self.epochs,
                self.embedding_dim,
                self.hidden_dim,
            )
            > 0,
            "Training dimensions must be positive",
        )


@dataclass
class PreparedData:
    datasets: dict
    parts: dict
    preprocessing: dict
    embedding_sizes: dict
    note: str


class TabularGRU(nn.Module):

    def __init__(
        self,
        num_cols,
        cat_cols,
        cat_num_embeddings,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=2,
        dropout=0.3,
    ):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.n_num = len(num_cols)
        self.n_cat = len(cat_cols)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_proj = nn.Linear(1, embedding_dim)
        self.cat_embeddings = nn.ModuleDict(
            {c: nn.Embedding(cat_num_embeddings[c], embedding_dim) for c in cat_cols}
        )
        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        num_tensor, cat_tensor = input_ids
        B = num_tensor.size(0)
        emb_list = []
        if self.n_num > 0:
            num_flat = num_tensor.view(-1, 1)
            num_proj_flat = self.num_proj(num_flat)
            num_proj = num_proj_flat.view(B, self.n_num, -1)
            emb_list.append(num_proj)
        if self.n_cat > 0:
            cat_embs = []
            for i, c in enumerate(self.cat_cols):
                emb_i = self.cat_embeddings[c](cat_tensor[:, i])
                cat_embs.append(emb_i.unsqueeze(1))
            cat_embs = torch.cat(cat_embs, dim=1)
            emb_list.append(cat_embs)
        seq = torch.cat(emb_list, dim=1)
        _, hidden = self.gru(seq)
        out = self.fc(self.dropout(hidden[-1]))
        return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _subset(frame, count):
    if len(frame) <= count:
        return frame
    indices, _ = train_test_split(
        np.arange(len(frame)),
        train_size=count,
        stratify=frame.fraud_bool,
        random_state=42,
    )
    return frame.iloc[indices]


def prepare(frame, config):
    """Fit preprocessing once, then reuse exactly the same tensors for both losses."""
    if config.protocol == "author_fixed":
        outer_train, random_test = train_test_split(
            frame,
            test_size=0.2,
            stratify=frame.fraud_bool,
            random_state=42,
        )
        train, validation = train_test_split(
            outer_train,
            test_size=0.2,
            stratify=outer_train.fraud_bool,
            random_state=42,
        )
        parts = {"train": train, "validation": validation, "random_test": random_test}
        fit_frame = outer_train
        numeric_columns = AUTHOR_NUMERIC
        categorical_columns = AUTHOR_CATEGORICAL
        note = (
            "Author fixed recipe: random64/16/20, preprocessing fit outer80% as code. "
            "Inner validation features enter preprocessing. No HPO. "
            "Seed reset and paired unweighted ablation added. Random split includes month7."
        )
    else:
        parts = {
            name: part
            for name, part in baf.temporal_parts(frame).items()
            if name != "test"
        }
        fit_frame = parts["train"]
        categorical_columns = baf.CATEGORICAL_COLUMNS
        numeric_columns = [
            column
            for column in frame
            if column not in ["fraud_bool", "month", *categorical_columns]
        ]
        note = (
            "Temporal adaptation: train0..5, validation6, no month7 predictions. "
            "Shared BAF schema/sentinel handling; train-only preprocessing. "
            "Numeric median/scaler without missing indicators; five categorical columns."
        )
    if config.mode == "pilot":
        parts = {
            name: _subset(
                part, config.pilot_train if name == "train" else config.pilot_eval
            )
            for name, part in parts.items()
        }
        if config.protocol == "temporal":
            fit_frame = parts["train"]
        note += " PILOT subsample/one epoch: not a full BAF benchmark."

    def features(part):
        if config.protocol == "temporal":
            return baf.clean_features(part)
        return part[numeric_columns + categorical_columns].copy()

    fit_features = features(fit_frame)
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    numeric.fit(fit_features[numeric_columns])
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="__missing__"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    categorical.fit(fit_features[categorical_columns])
    categories = categorical[-1].categories_
    mappings = {
        column: {str(value): index for index, value in enumerate(values)}
        for column, values in zip(categorical_columns, categories)
    }
    embedding_sizes = {column: len(mapping) + 1 for column, mapping in mappings.items()}
    datasets = {}
    for name, part in parts.items():
        baf.require(part.fraud_bool.nunique() == 2, f"Both classes required in {name}")
        table = features(part)
        numeric_array = np.asarray(
            numeric.transform(table[numeric_columns]), dtype=np.float32
        )
        categorical_array = np.column_stack(
            [
                table[column]
                .fillna("__MISSING__")
                .astype(str)
                .map(mappings[column])
                .fillna(len(mappings[column]))
                .to_numpy(dtype=np.int64)
                for column in categorical_columns
            ]
        )
        baf.require(np.isfinite(numeric_array).all(), "Non-finite numeric input")
        datasets[name] = TensorDataset(
            torch.from_numpy(numeric_array),
            torch.from_numpy(categorical_array),
            torch.from_numpy(part.fraud_bool.to_numpy(dtype=np.int64, copy=True)),
        )
    preprocessing = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categories": {column: list(mapping) for column, mapping in mappings.items()},
        "imputer_statistics": numeric[0].statistics_.tolist(),
        "scaler_mean": numeric[1].mean_.tolist(),
        "scaler_scale": numeric[1].scale_.tolist(),
        "fit_rows": len(fit_frame),
    }
    return PreparedData(datasets, parts, preprocessing, embedding_sizes, note)


@torch.no_grad()
def predict(model, dataset, batch_size, device):
    model.eval()
    probabilities = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for numeric, categorical, _ in loader:
        logits = model((numeric.to(device), categorical.to(device)))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def train(
    data_path,
    output,
    config=None,
    frame=None,
    expected_rows=1_000_000,
):
    config = Config() if config is None else config
    config.validate()
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the matching PyTorch build")
    torch.set_num_threads(config.threads)
    rows = []
    history = []
    with baf.experiment(output):
        frame = baf.load_data(data_path, expected_rows) if frame is None else frame
        baf.validate_data(frame, expected_rows)
        prepared = prepare(frame, config)
        arguments = vars(config) | {
            "epochs": config.epochs,
            "evaluation_partitions": [
                name for name in prepared.parts if name != "train"
            ],
        }
        baf.save_manifest(
            output,
            data_path,
            frame,
            "gru",
            arguments,
            prepared.parts,
            preprocessing_note=prepared.note,
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            device=str(device),
            torch_cuda=torch.version.cuda,
            author_commit="6e8252cb9080378905b2aae30033c5946b72335b",
            scope="fixed-configuration author recipe / temporal adaptation; no HPO",
        )
        print(prepared.note)
        baf.write_json(
            Path(output) / "evidence/preprocessing.json", prepared.preprocessing
        )
        split_indices = pd.concat(
            [
                pd.DataFrame({"partition": name, "row_index": part.index})
                for name, part in prepared.parts.items()
            ]
        )
        split_indices.to_csv(
            Path(output) / "evidence/split_indices.csv.gz", index=False
        )
        model_dir = Path(output) / "models"
        model_dir.mkdir()
        labels = prepared.datasets["train"].tensors[2].numpy()
        epochs = 1 if config.mode == "pilot" else config.epochs
        initial_hashes = {}
        for method in config.methods:
            for seed in config.seeds:
                set_seed(seed)
                model = TabularGRU(
                    prepared.preprocessing["numeric_columns"],
                    prepared.preprocessing["categorical_columns"],
                    prepared.embedding_sizes,
                    embedding_dim=config.embedding_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                ).to(device)
                initial_hash = hashlib.sha256(
                    b"".join(
                        value.detach().cpu().numpy().tobytes()
                        for value in model.state_dict().values()
                    )
                ).hexdigest()
                if seed in initial_hashes:
                    baf.require(
                        initial_hashes[seed] == initial_hash,
                        "Initialization differs across losses",
                    )
                initial_hashes[seed] = initial_hash
                weights = (
                    compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
                    if method == "balanced"
                    else np.ones(2)
                )
                loss_weights = (
                    torch.tensor(weights, dtype=torch.float32, device=device)
                    if method == "balanced"
                    else None
                )
                criterion = nn.CrossEntropyLoss(weight=loss_weights)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config.learning_rate
                )
                loader = DataLoader(
                    prepared.datasets["train"],
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator().manual_seed(seed),
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with baf.MemoryMonitor() as memory:
                        for epoch in range(epochs):
                            model.train()
                            losses = []
                            epoch_started = time.perf_counter()
                            for numeric, categorical, target in loader:
                                optimizer.zero_grad(set_to_none=True)
                                logits = model(
                                    (numeric.to(device), categorical.to(device))
                                )
                                loss = criterion(logits, target.to(device))
                                loss.backward()
                                optimizer.step()
                                losses.append(loss.item())
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            epoch_row = {
                                "method": method,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "loss_batch_mean": float(np.mean(losses)),
                                "seconds": time.perf_counter() - epoch_started,
                            }
                            history.append(epoch_row)
                            pd.DataFrame(history).to_csv(
                                Path(output) / "epoch_history.csv", index=False
                            )
                            print("EPOCH", json.dumps(epoch_row), flush=True)
                fit_seconds = time.perf_counter() - started
                validation = prepared.datasets["validation"]
                validation_scores = predict(
                    model, validation, config.batch_size, device
                )
                threshold = baf.select_threshold(
                    validation.tensors[2].numpy(), validation_scores
                )
                for name, dataset in prepared.datasets.items():
                    if name == "train":
                        continue
                    probabilities = (
                        validation_scores
                        if name == "validation"
                        else predict(model, dataset, config.batch_size, device)
                    )
                    row = {
                        "protocol": config.protocol,
                        "mode": config.mode,
                        "method": f"gru_{method}",
                        "seed": seed,
                        "partition": name,
                        "threshold": (
                            threshold if np.isfinite(threshold) else "infinity"
                        ),
                        "fit_seconds": fit_seconds,
                        "peak_rss_mib": memory.peak_mib,
                        "peak_cuda_allocated_mib": (
                            torch.cuda.max_memory_allocated(device) / 2**20
                            if device.type == "cuda"
                            else None
                        ),
                        "initial_model_sha256": initial_hash,
                        "memory_warning": memory.warning,
                        "weight_negative": float(weights[0]),
                        "weight_positive": float(weights[1]),
                        "warnings": " | ".join(str(item.message) for item in caught),
                        **baf.evaluate(
                            dataset.tensors[2].numpy(), probabilities, threshold
                        ),
                    }
                    rows.append(row)
                    baf.save_predictions(
                        output,
                        f"gru_{method}",
                        seed,
                        name,
                        prepared.parts[name],
                        probabilities,
                    )
                    pd.DataFrame(rows).to_csv(Path(output) / "metrics.csv", index=False)
                    print("METRICS", json.dumps(row), flush=True)
                if config.save_models:
                    torch.save(
                        model.state_dict(), model_dir / f"gru_{method}_{seed}.pt"
                    )
                del model, optimizer, criterion, loader
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# Research adapters: BAF transfers of published weighting mechanisms.
# Legacy GRU functions above are retained for notebooks 04--06.
import contextlib
import copy
import importlib.util
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import asdict

import joblib
from scipy.special import expit, logsumexp
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch.func import functional_call

RESEARCH_VERSION = "baf-research-20261028-v1"
BAF_RESEARCH_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"


def research_json(path, value):
    """Write atomically so an interrupted run cannot leave a valid-looking JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def research_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def research_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def research_versions():
    versions = baf.package_versions(include_torch=True)
    for name in ["xgboost", "nbformat", "ipython", "joblib"]:
        try:
            versions[name] = __import__("importlib.metadata").metadata.version(name)
        except __import__("importlib.metadata").metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "packages": versions}


def research_sources(project, clone=True):
    """Obtain pinned author repositories and verify the files used by each port."""
    project = Path(project).resolve()
    lock = research_read(project / "docs/research_sources.json")
    identities = {}
    for tag, spec in lock.items():
        target = project / "external/research_sources" / tag
        if not target.exists():
            if not clone:
                raise FileNotFoundError(f"Source repository missing: {target}")
            if shutil.which("git") is None:
                raise RuntimeError(
                    "Install Git for Windows, reopen VS Code, then run 11."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    spec["repo"],
                    str(target),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-C",
                    str(target),
                    "checkout",
                    "--detach",
                    spec["commit"],
                ],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        remote = subprocess.check_output(
            ["git", "-C", str(target), "remote", "get-url", "origin"], text=True
        ).strip()
        baf.require(actual == spec["commit"], f"Wrong author commit: {tag}")
        baf.require(remote == spec["repo"], f"Wrong author repository: {tag}")
        for name, digest in spec["files"].items():
            baf.require(
                baf.sha256(target / name) == digest, f"Source changed: {tag}/{name}"
            )
        identities[tag] = {
            "repo": remote,
            "commit": actual,
            "verified_files": len(spec["files"]),
        }
        print("SOURCE_VERIFIED", tag, actual, flush=True)
    return identities


def research_source_module(project, tag, filename):
    """Import a specific pinned Python file, without importing its legacy trainer."""
    project = Path(project)
    spec = research_read(project / "docs/research_sources.json")[tag]
    path = project / "external/research_sources" / tag / filename
    baf.require(
        baf.sha256(path) == spec["files"][filename], f"Changed source: {filename}"
    )
    module_name = (
        "_baf_author_" + tag + "_" + hashlib.sha256(filename.encode()).hexdigest()[:10]
    )
    if module_name not in sys.modules:
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module_spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
    return sys.modules[module_name]


@dataclass
class ResearchConfig:
    mode: str = "full"
    seeds: tuple = (42, 317, 2026)
    threads: int = 4
    device: str = "auto"
    meta_per_class: int = 128
    epochs: int = 20
    batch_size: int = 2048
    meta_batch_size: int = 256
    hidden: tuple = (128, 64)
    learning_rates: tuple = (0.01, 0.03, 0.1)
    weight_decay: float = 0.0001
    momentum: float = 0.9
    patience: int = 5
    max_fpr: float = 0.05
    tree_rounds: int = 300
    vae_epochs: int = 30
    diffusion_epochs: int = 50
    auxiliary_patience: int = 10
    auxiliary_batch_size: int = 1024
    diffusion_width: int = 1024
    score_batch_size: int = 2048
    score_noise_repeats: int = 32
    score_steps: int = 10
    score_noise_chunk: int = 4
    auxiliary_seed: int = 42
    lsr_centering: str = "author"
    bootstrap_repeats: int = 1000
    pilot_train: int = 20000
    pilot_eval: int = 10000

    def validate(self):
        baf.require(
            self.mode in ["full", "pilot", "synthetic"], "Unknown research mode"
        )
        baf.require(
            self.seeds and len(set(self.seeds)) == len(self.seeds), "Duplicate seeds"
        )
        baf.require(
            self.epochs > 0 and self.batch_size > 0 and self.threads > 0,
            "Invalid budget",
        )
        baf.require(
            self.meta_per_class > 0, "A balanced meta set is required by LRE/MWN"
        )
        baf.require(
            self.score_steps >= 2 and self.score_noise_repeats > 0,
            "Invalid score budget",
        )
        baf.require(self.lsr_centering in ["author", "shared"], "Unknown LSR centering")
        baf.require(0 < self.max_fpr < 1, "Invalid FPR cap")
        baf.require(
            all(rate > 0 for rate in self.learning_rates), "Invalid learning rate"
        )
        return self


def research_device(config):
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "This kernel cannot use CUDA; select the existing ml-gpu kernel."
        )
    return torch.device(device)


def research_subset(frame, limit, seed=42):
    if len(frame) <= limit:
        return frame.copy()
    from sklearn.model_selection import train_test_split

    indices, _ = train_test_split(
        np.arange(len(frame)),
        train_size=limit,
        random_state=seed,
        stratify=frame.fraud_bool.to_numpy(),
    )
    return frame.iloc[np.sort(indices)].copy()


def research_prepare(project, data_path, config, dataset_label="BAF_Base"):
    """Prepare one train-only feature cache shared by every research method."""
    config.validate()
    project = Path(project).resolve()
    data_path = Path(data_path).resolve()
    data_hash = baf.sha256(data_path)
    if config.mode != "synthetic":
        baf.require(
            data_hash == BAF_RESEARCH_SHA256, "Base.csv is not the audited BAF Base"
        )
    else:
        baf.require(
            dataset_label.startswith("SYNTHETIC"),
            "Synthetic data must be labeled explicitly",
        )
    plan = {
        "version": RESEARCH_VERSION,
        "data_sha256": data_hash,
        "mode": config.mode,
        "meta_per_class": config.meta_per_class,
        "pilot_train": config.pilot_train,
        "pilot_eval": config.pilot_eval,
        "feature_rule": "drop fraud_bool/month; sentinel cleaning; fit core train only",
    }
    cache = project / "results/research_data" / research_hash(plan)[:16]
    if (cache / "COMPLETE.json").is_file():
        research_verify_cache(cache)
        print("DATA_CACHE_REUSED", cache)
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    frame = baf.load_data(
        data_path, expected_rows=None if config.mode == "synthetic" else 1000000
    )
    parts = baf.temporal_parts(frame)
    if config.mode in ["pilot", "synthetic"]:
        parts["train"] = research_subset(parts["train"], config.pilot_train)
        for name in ["validation", "test"]:
            parts[name] = research_subset(parts[name], config.pilot_eval)
    rng = np.random.default_rng(317)
    meta_rows = []
    last_month = int(parts["train"].month.max())
    for label in [0, 1]:
        eligible = parts["train"].loc[
            (parts["train"].month == last_month) & (parts["train"].fraud_bool == label)
        ]
        baf.require(
            len(eligible) >= config.meta_per_class,
            "Not enough samples for the fixed meta set",
        )
        meta_rows.extend(
            rng.choice(eligible.index.to_numpy(), config.meta_per_class, replace=False)
        )
    parts["meta"] = parts["train"].loc[sorted(meta_rows)].copy()
    parts["train"] = parts["train"].drop(index=meta_rows).copy()
    for name, part in parts.items():
        baf.require(part.fraud_bool.nunique() == 2, f"Both classes required: {name}")
    features = {name: baf.clean_features(part) for name, part in parts.items()}
    numeric = [c for c in features["train"] if c not in baf.CATEGORICAL_COLUMNS]
    categorical = [c for c in baf.CATEGORICAL_COLUMNS if c in features["train"]]
    num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    raw_num_train = num_imputer.fit_transform(features["train"][numeric])
    scaler = StandardScaler().fit(raw_num_train)
    quantile = QuantileTransformer(
        n_quantiles=min(1000, len(raw_num_train)),
        output_distribution="normal",
        subsample=min(100000, len(raw_num_train)),
        random_state=42,
    ).fit(raw_num_train)
    categories = {
        col: sorted(
            features["train"][col].fillna("__MISSING__").astype(str).unique().tolist()
        )
        for col in categorical
    }
    category_maps = {
        col: {value: i + 1 for i, value in enumerate(values)}
        for col, values in categories.items()
    }
    cardinalities = [len(categories[col]) + 1 for col in categorical]
    eye = [np.eye(size, dtype=np.float32) for size in cardinalities]
    counts = {}
    for name, part in parts.items():
        folder = cache / name
        folder.mkdir(exist_ok=True)
        num = num_imputer.transform(features[name][numeric])
        cat = np.column_stack(
            [
                features[name][col]
                .fillna("__MISSING__")
                .astype(str)
                .map(category_maps[col])
                .fillna(0)
                .to_numpy(dtype=np.int64)
                for col in categorical
            ]
        )
        dense = np.column_stack(
            [
                scaler.transform(num),
                *[matrix[cat[:, j]] for j, matrix in enumerate(eye)],
            ]
        )
        np.save(folder / "dense.npy", dense.astype(np.float32))
        np.save(folder / "lsr_num.npy", quantile.transform(num).astype(np.float32))
        np.save(folder / "lsr_cat.npy", cat)
        np.savez_compressed(
            folder / "rows.npz",
            row_index=part.index.to_numpy(dtype=np.int64),
            y=part.fraud_bool.to_numpy(dtype=np.int64),
            month=part.month.to_numpy(dtype=np.int64),
            customer_age=part.customer_age.to_numpy(dtype=np.int64),
        )
        counts[name] = {
            "n": len(part),
            "fraud": int(part.fraud_bool.sum()),
            "months": sorted(map(int, part.month.unique())),
        }
    joblib.dump(
        {
            "num_imputer": num_imputer,
            "scaler": scaler,
            "quantile": quantile,
            "category_maps": category_maps,
            "numeric": numeric,
            "categorical": categorical,
        },
        cache / "preprocessors.joblib",
        compress=3,
    )
    metadata = {
        **plan,
        "dataset_label": dataset_label,
        "data_path": str(data_path),
        "counts": counts,
        "categories": categories,
        "cardinalities": cardinalities,
        "numeric": numeric,
        "categorical": categorical,
        "meta_rule": "seed317; balanced labels; held out from last train month; removed from all learners",
        "feature_count": int(dense.shape[1]),
        "versions": research_versions(),
    }
    research_json(cache / "metadata.json", metadata)
    hashes = {
        p.relative_to(cache).as_posix(): baf.sha256(p)
        for p in cache.rglob("*")
        if p.is_file() and p.name != "COMPLETE.json"
    }
    research_json(cache / "COMPLETE.json", hashes)
    print(pd.DataFrame(counts).T.to_string())
    print("RESEARCH_DATA_READY", cache)
    return cache


def research_verify_cache(cache):
    cache = Path(cache)
    for name, digest in research_read(cache / "COMPLETE.json").items():
        baf.require(baf.sha256(cache / name) == digest, f"Cache changed: {name}")


def research_rows(cache, partition):
    with np.load(Path(cache) / partition / "rows.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def research_array(cache, partition, feature="dense"):
    return np.load(Path(cache) / partition / (feature + ".npy"), mmap_mode="r")


def research_effective_number(counts, beta):
    counts = np.asarray(counts, dtype=np.float64)
    baf.require(
        (counts > 0).all() and 0 <= beta < 1, "Invalid effective-number arguments"
    )
    if beta == 0:
        weights = np.ones(len(counts))
    else:
        effective = -np.expm1(counts * np.log(beta))
        weights = (1 - beta) / effective
    return weights / weights.sum() * len(counts)


class ResearchNormedLinear(nn.Module):
    """Device-independent counterpart of the author's normalized classifier head."""

    def __init__(self, input_dim, classes=2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, inputs):
        return nn.functional.normalize(inputs, dim=1) @ nn.functional.normalize(
            self.weight, dim=0
        )


class ResearchMLP(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64), normed=False, lsr=False):
        super().__init__()
        widths = (1024, 1024, 512) if lsr else tuple(hidden)
        layers = []
        previous = input_dim
        for width in widths:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.LeakyReLU() if lsr else nn.ReLU())
            previous = width
        layers.append(
            ResearchNormedLinear(previous) if normed else nn.Linear(previous, 2)
        )
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class ResearchWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, losses):
        return self.network(losses.reshape(-1, 1)).reshape(-1)


def research_loss(logits, labels, method, counts, recipe, epoch, epochs):
    F = nn.functional
    ce = F.cross_entropy(logits, labels, reduction="none")
    if method == "norm_ce":
        return F.cross_entropy(30.0 * logits, labels)
    if method in ["ce", "lre", "mwn"]:
        return ce.mean()
    if method == "balanced":
        weights = torch.as_tensor(
            sum(counts) / (2 * np.asarray(counts)),
            device=logits.device,
            dtype=logits.dtype,
        )
        return F.cross_entropy(logits, labels, weight=weights)
    if method == "focal":
        pt = torch.exp(-ce)
        alpha = recipe.get("alpha", 0.25)
        alpha_y = torch.where(labels == 1, alpha, 1 - alpha)
        return (alpha_y * (1 - pt).pow(recipe.get("gamma", 2.0)) * ce).mean()
    beta = recipe.get("beta", 0.9999)
    deferred = method in ["drw", "ldam_drw"]
    if deferred and epoch < max(1, int(epochs * 0.8)):
        beta = 0.0
    weights = torch.as_tensor(
        research_effective_number(counts, beta),
        device=logits.device,
        dtype=logits.dtype,
    )
    if method in ["cb_ce", "drw"]:
        if method == "drw":
            return F.cross_entropy(logits, labels, weight=weights)
        return (weights[labels] * ce).mean()
    if method == "cb_focal":
        target = F.one_hot(labels, 2).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        return (
            weights[labels, None] * (1 - pt).pow(recipe.get("gamma", 2.0)) * bce
        ).sum() / len(labels)
    if method in ["ldam", "ldam_drw"]:
        margins = np.asarray(counts, dtype=np.float64) ** (-0.25)
        margins *= recipe.get("max_margin", 0.5) / margins.max()
        margin = torch.as_tensor(margins, device=logits.device, dtype=logits.dtype)[
            labels
        ]
        adjusted = logits - F.one_hot(labels, 2).to(logits.dtype) * margin[:, None]
        return F.cross_entropy(
            30.0 * adjusted, labels, weight=weights if method == "ldam_drw" else None
        )
    raise ValueError(f"Unknown neural method: {method}")


def research_lre_weights(model, inputs, labels, meta_inputs, meta_labels, inner_lr):
    """Ren et al.: differentiate a clean meta loss through one virtual SGD step."""
    losses = nn.functional.cross_entropy(model(inputs), labels, reduction="none")
    epsilon = torch.zeros_like(losses, requires_grad=True)
    gradients = torch.autograd.grad(
        (epsilon * losses).sum(), model.parameters(), create_graph=True
    )
    virtual = OrderedDict(
        (name, param - inner_lr * grad)
        for (name, param), grad in zip(model.named_parameters(), gradients)
    )
    meta_logits = functional_call(model, virtual, (meta_inputs,))
    meta_loss = nn.functional.cross_entropy(meta_logits, meta_labels)
    epsilon_gradient = torch.autograd.grad(meta_loss, epsilon)[0]
    positive = (-epsilon_gradient).clamp(min=0)
    total = positive.sum()
    return (
        positive / total if total.item() > 0 else positive
    ).detach(), meta_loss.detach()


def research_mwn_step(
    model,
    weight_net,
    weight_optimizer,
    inputs,
    labels,
    meta_inputs,
    meta_labels,
    inner_lr,
):
    """Port of the class-imbalance MW-Net update, including weight normalization."""
    losses = nn.functional.cross_entropy(model(inputs), labels, reduction="none")
    weights = weight_net(losses)
    normalized = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    gradients = torch.autograd.grad(
        (losses * normalized).sum(), model.parameters(), create_graph=True
    )
    virtual = OrderedDict(
        (name, param - inner_lr * grad)
        for (name, param), grad in zip(model.named_parameters(), gradients)
    )
    meta_logits = functional_call(model, virtual, (meta_inputs,))
    meta_loss = nn.functional.cross_entropy(meta_logits, meta_labels)
    weight_optimizer.zero_grad()
    meta_loss.backward()
    weight_optimizer.step()
    with torch.no_grad():
        current = nn.functional.cross_entropy(model(inputs), labels, reduction="none")
        weights = weight_net(current)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return weights, meta_loss.detach()


def research_predict_nn(model, array, device, batch_size, link="softmax"):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = torch.as_tensor(
                np.array(array[start : start + batch_size], copy=True), device=device
            )
            logits = model(batch)
            probability = (
                torch.sigmoid(logits)[:, 1]
                if link == "sigmoid_class1"
                else logits.softmax(dim=1)[:, 1]
            )
            scores.append(probability.cpu().numpy().astype(np.float32))
    return np.concatenate(scores)


def research_fairness(labels, scores, ages, threshold):
    prediction = np.asarray(scores) >= threshold
    rows = {}
    for name, mask in [
        ("younger", np.asarray(ages) < 50),
        ("older", np.asarray(ages) >= 50),
    ]:
        tn, fp, fn, tp = confusion_matrix(
            np.asarray(labels)[mask], prediction[mask], labels=[0, 1]
        ).ravel()
        rows[name + "_fpr"] = float(fp / (tn + fp)) if tn + fp else None
        rows[name + "_tpr"] = float(tp / (tp + fn)) if tp + fn else None
        rows[name + "_n"] = int(mask.sum())
    fprs = [rows["younger_fpr"], rows["older_fpr"]]
    if all(value is not None for value in fprs):
        rows["fpr_gap_older_minus_younger"] = fprs[1] - fprs[0]
        rows["fpr_ratio_min_max"] = min(fprs) / max(fprs) if max(fprs) > 0 else 1.0
    return rows


def research_weight_summary(weights, labels):
    weights = np.asarray(weights, dtype=np.float64)
    result = {
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
    }
    for label in [0, 1]:
        values = weights[np.asarray(labels) == label]
        result[str(label)] = {
            "n": int(len(values)),
            "mass": float(values.sum()),
            "ess": (
                float(values.sum() ** 2 / np.square(values).sum())
                if np.square(values).sum()
                else 0.0
            ),
        }
    return result


def research_recipes(suite, config, counts):
    recipes = []
    if suite in ["losses", "meta"]:
        methods = (
            [
                "ce",
                "balanced",
                "focal",
                "cb_ce",
                "cb_focal",
                "drw",
                "norm_ce",
                "ldam",
                "ldam_drw",
            ]
            if suite == "losses"
            else ["lre", "mwn"]
        )
        for method in methods:
            for index, rate in enumerate(config.learning_rates):
                recipe = {
                    "method": method,
                    "lr": float(rate),
                    "candidate": index,
                    "family": "nn",
                    "normed": method in ["norm_ce", "ldam", "ldam_drw"],
                    "feature": "dense",
                }
                if method in ["cb_ce", "cb_focal"]:
                    recipe.update(beta=[0.9999, 0.99999, 0.999999][index % 3], lr=0.03)
                if method == "focal":
                    recipe.update(
                        gamma=2.0, alpha=[0.25, 0.75, 0.99][index % 3], lr=0.03
                    )
                recipes.append(recipe)
    elif suite == "xgboost":
        ratio = float(counts[0] / counts[1])
        for method in ["xgb_none", "xgb_weighted", "xgb_focal"]:
            for index, depth in enumerate([3, 5, 7]):
                recipes.append(
                    {
                        "method": method,
                        "family": "xgboost",
                        "candidate": index,
                        "depth": depth,
                        "gamma": 2.0,
                        "alpha": ratio ** [0.25, 0.5, 1.0][index],
                        "feature": "dense",
                    }
                )
    elif suite == "lsr":
        for method in ["lsr_nn_none", "lsr_nn_balanced", "lsr_nn_weighted"]:
            for index, rate in enumerate([0.0003, 0.001, 0.003]):
                recipes.append(
                    {
                        "method": method,
                        "family": "lsr_nn",
                        "candidate": index,
                        "lr": rate,
                        "feature": "latent",
                        "lsr": True,
                        "temperature": 3.0,
                    }
                )
        for method in ["lsr_lgbm_none", "lsr_lgbm_balanced", "lsr_lgbm_weighted"]:
            for index, leaves in enumerate([15, 31, 63]):
                recipes.append(
                    {
                        "method": method,
                        "family": "lsr_lgbm",
                        "candidate": index,
                        "leaves": leaves,
                        "feature": "dense",
                        "temperature": 3.0,
                    }
                )
    else:
        raise ValueError(f"Unknown suite: {suite}")
    if config.mode == "synthetic":
        recipes = [r for r in recipes if r["candidate"] == 0]
    return recipes


class ResearchStudy:
    """Shared orchestration for source-locked, restartable weighting experiments."""

    def __init__(self, project, cache, suite, config, output):
        self.project = Path(project).resolve()
        self.cache = Path(cache).resolve()
        self.output = Path(output).resolve()
        self.config = config.validate()
        self.suite = suite
        self.device = research_device(config)
        torch.set_num_threads(config.threads)
        research_verify_cache(cache)
        self.data = research_read(self.cache / "metadata.json")
        baf.require(self.data["mode"] == config.mode, "Cache and run modes differ")
        self.train_rows = research_rows(cache, "train")
        self.validation_rows = research_rows(cache, "validation")
        self.meta_rows = research_rows(cache, "meta")
        self.counts = np.bincount(self.train_rows["y"], minlength=2)
        self.recipes = research_recipes(suite, config, self.counts)
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "version": RESEARCH_VERSION,
            "suite": suite,
            "config": asdict(config),
            "cache_hash": baf.sha256(self.cache / "COMPLETE.json"),
            "data_metadata": self.data,
            "source_hashes": {
                name: baf.sha256(self.project / "src" / name)
                for name in ["baf.py", "gru.py"]
            },
            "author_lock_sha256": baf.sha256(
                self.project / "docs/research_sources.json"
            ),
            "environment": research_versions(),
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "scope": "Published weighting mechanisms transferred to BAF; not original image-dataset experiments",
            "evaluation": "Validation only during training; test opened by notebook16 after all suites freeze",
        }
        signature = research_hash(self.identity)
        manifest = self.output / "manifest.json"
        if manifest.exists():
            baf.require(
                research_read(manifest)["signature"] == signature,
                "Resume identity mismatch: data/source/config/environment",
            )
        else:
            research_json(manifest, {"signature": signature, **self.identity})
            evidence = self.output / "evidence"
            (evidence / "src").mkdir(parents=True, exist_ok=True)
            for name in ["baf.py", "gru.py"]:
                shutil.copy2(self.project / "src" / name, evidence / "src" / name)
            shutil.copy2(
                self.project / "docs/research_sources.json",
                evidence / "research_sources.json",
            )
            shutil.copy2(
                self.project / "docs/RESEARCH_METHODS.md",
                evidence / "RESEARCH_METHODS.md",
            )
            shutil.copy2(self.cache / "metadata.json", evidence / "data_metadata.json")
            for tag, spec in research_read(
                self.project / "docs/research_sources.json"
            ).items():
                for name in spec["files"]:
                    target = evidence / "upstream" / tag / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        self.project / "external/research_sources" / tag / name, target
                    )
            for partition in ["train", "meta", "validation", "test"]:
                shutil.copy2(
                    self.cache / partition / "rows.npz",
                    evidence / (partition + "_rows.npz"),
                )
        print(
            "RESEARCH_STUDY",
            suite,
            "mode=",
            config.mode,
            "device=",
            self.device,
            flush=True,
        )

    def feature(self, partition, feature="dense"):
        if feature == "latent":
            return np.load(
                self.output / "auxiliary" / (partition + "_latent.npy"), mmap_mode="r"
            )
        return research_array(self.cache, partition, feature)

    def fit(self, recipe, seed):
        name = f"{recipe['method']}_c{recipe['candidate']:02d}_s{seed}"
        folder = self.output / "fits" / name
        marker = folder / "COMPLETE.json"
        if marker.is_file():
            record = research_read(marker)
            for relative, digest in record["hashes"].items():
                baf.require(
                    baf.sha256(folder / relative) == digest,
                    f"Cached fit changed: {relative}",
                )
            print("FIT_REUSED", name, flush=True)
            return record
        folder.mkdir(parents=True, exist_ok=True)
        print("FIT_BEGIN", name, recipe, flush=True)
        started = time.perf_counter()
        with baf.MemoryMonitor() as memory:
            if recipe["family"] in ["nn", "lsr_nn"]:
                result = research_fit_neural(self, recipe, seed, folder)
            elif recipe["family"] == "xgboost":
                result = research_fit_xgboost(self, recipe, seed, folder)
            elif recipe["family"] == "lsr_lgbm":
                result = research_fit_lgbm(self, recipe, seed, folder)
            else:
                raise ValueError(recipe["family"])
        scores = np.load(folder / "validation_score.npy")
        threshold = baf.select_threshold(
            self.validation_rows["y"], scores, self.config.max_fpr
        )
        metrics = baf.evaluate(
            self.validation_rows["y"], scores, threshold, self.config.max_fpr
        )
        fairness = research_fairness(
            self.validation_rows["y"],
            scores,
            self.validation_rows["customer_age"],
            threshold,
        )
        record = {
            "name": name,
            "recipe": recipe,
            "seed": int(seed),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "threshold_is_posinf": bool(np.isposinf(threshold)),
            "fit_seconds": time.perf_counter() - started,
            "peak_rss_mib": memory.peak_mib,
            "metrics": metrics,
            "fairness": fairness,
            **result,
        }
        record["hashes"] = {
            p.relative_to(folder).as_posix(): baf.sha256(p)
            for p in folder.rglob("*")
            if p.is_file() and p.name != "COMPLETE.json"
        }
        research_json(marker, record)
        print("FIT_COMPLETE", name, "validation_AP=", metrics["ap"], flush=True)
        return record

    def run(self):
        if self.suite == "lsr":
            research_fit_lsr_auxiliary(self)
        records = [self.fit(recipe, self.config.seeds[0]) for recipe in self.recipes]
        selections = {}
        for method in sorted({r["recipe"]["method"] for r in records}):
            choices = [r for r in records if r["recipe"]["method"] == method]
            best = max(
                choices,
                key=lambda row: (row["metrics"]["ap"], -row["recipe"]["candidate"]),
            )
            selections[method] = best["recipe"]
        research_json(self.output / "selections.json", selections)
        final = []
        for method, recipe in selections.items():
            for seed in self.config.seeds:
                final.append(self.fit(recipe, seed))
        table = []
        for row in final:
            table.append(
                {
                    "method": row["recipe"]["method"],
                    "seed": row["seed"],
                    "validation_ap": row["metrics"]["ap"],
                    "threshold": research_threshold(row),
                    "fit_seconds": row["fit_seconds"],
                    "family": row["recipe"]["family"],
                    "feature": row["recipe"]["feature"],
                }
            )
        pd.DataFrame(table).to_csv(self.output / "validation_summary.csv", index=False)
        frozen = {
            "manifest_sha256": baf.sha256(self.output / "manifest.json"),
            "selections": selections,
            "final": [
                {
                    "name": row["name"],
                    "record_sha256": baf.sha256(
                        self.output / "fits" / row["name"] / "COMPLETE.json"
                    ),
                }
                for row in final
            ],
        }
        research_json(self.output / "FROZEN.json", frozen)
        print("ALL_RESEARCH_MODELS_AND_THRESHOLDS_FROZEN", len(final), flush=True)
        return pd.DataFrame(table)


def research_fit_neural(study, recipe, seed, folder):
    config = study.config
    device = study.device
    set_seed(seed)
    feature = recipe["feature"]
    train = study.feature("train", feature)
    validation = study.feature("validation", feature)
    model = ResearchMLP(
        train.shape[1],
        config.hidden,
        recipe.get("normed", False),
        recipe.get("lsr", False),
    ).to(device)
    initial_hash = research_hash(
        {
            name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()
        }
    )
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=recipe["lr"])
        if recipe["family"] == "lsr_nn"
        else torch.optim.SGD(
            model.parameters(),
            lr=recipe["lr"],
            momentum=config.momentum,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    )
    tensors = TensorDataset(
        torch.from_numpy(np.array(train, copy=True)),
        torch.from_numpy(study.train_rows["y"].copy()),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensors,
        batch_size=(
            config.meta_batch_size
            if recipe["method"] in ["lre", "mwn"]
            else config.batch_size
        ),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    weights_array = None
    if recipe["method"] == "lsr_nn_weighted":
        weights_array = np.load(study.output / "auxiliary/weights.npy")
    elif recipe["method"] == "lsr_nn_balanced":
        weights_array = np.asarray(study.counts.sum() / (2 * study.counts))[
            study.train_rows["y"]
        ]
    if weights_array is not None:
        tensors = TensorDataset(
            tensors.tensors[0],
            tensors.tensors[1],
            torch.from_numpy(weights_array.astype(np.float32)),
        )
        loader = DataLoader(
            tensors,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    meta_x = torch.from_numpy(np.array(study.feature("meta"), copy=True)).to(device)
    meta_y = torch.from_numpy(study.meta_rows["y"].copy()).to(device)
    meta_rng = np.random.default_rng(seed + 9026)
    weight_net = None
    if recipe["method"] == "mwn":
        with torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ):
            torch.manual_seed(seed + 10000)
            weight_net = ResearchWeightNet().to(device)
        weight_optimizer = torch.optim.SGD(
            weight_net.parameters(),
            lr=1e-5,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
        )
    best_ap = -1.0
    best_epoch = 0
    history = []
    link = "sigmoid_class1" if recipe["method"] == "cb_focal" else "softmax"
    zero_weight_batches = 0
    epoch_weight_stats = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        wsum = np.zeros(2)
        wsquare = np.zeros(2)
        wn = np.zeros(2, dtype=np.int64)
        epoch_started = time.perf_counter()
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            method = recipe["method"]
            batch_weights = None
            if method in ["lre", "mwn"]:
                chosen = meta_rng.choice(
                    len(meta_y),
                    size=min(config.meta_batch_size, len(meta_y)),
                    replace=False,
                )
                idx = torch.as_tensor(chosen, device=device)
                if method == "lre":
                    batch_weights, _ = research_lre_weights(
                        model, inputs, labels, meta_x[idx], meta_y[idx], recipe["lr"]
                    )
                    zero_weight_batches += int(batch_weights.sum().item() == 0)
                else:
                    batch_weights, _ = research_mwn_step(
                        model,
                        weight_net,
                        weight_optimizer,
                        inputs,
                        labels,
                        meta_x[idx],
                        meta_y[idx],
                        recipe["lr"],
                    )
                losses = nn.functional.cross_entropy(
                    model(inputs), labels, reduction="none"
                )
                loss = (batch_weights * losses).sum()
            elif method.startswith("lsr_nn"):
                losses = nn.functional.cross_entropy(
                    model(inputs), labels, reduction="none"
                )
                batch_weights = (
                    batch[2].to(device) if len(batch) == 3 else torch.ones_like(losses)
                )
                loss = (losses * batch_weights).mean()
            else:
                loss = research_loss(
                    model(inputs),
                    labels,
                    method,
                    study.counts,
                    recipe,
                    epoch,
                    config.epochs,
                )
            baf.require(
                torch.isfinite(loss).item(),
                "Non-finite loss; keep the failed run for diagnosis",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            sample_count += len(labels)
            if batch_weights is not None:
                observed = batch_weights.detach().cpu().numpy().astype(np.float64)
                observed_y = labels.detach().cpu().numpy()
                for label in [0, 1]:
                    values = observed[observed_y == label]
                    wsum[label] += values.sum()
                    wsquare[label] += np.square(values).sum()
                    wn[label] += len(values)
        scores = research_predict_nn(model, validation, device, config.batch_size, link)
        ap = float(average_precision_score(study.validation_rows["y"], scores))
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / sample_count,
            "validation_ap": ap,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        if wn.sum():
            epoch_weight_stats.append(
                {
                    "epoch": epoch + 1,
                    "mass": wsum.tolist(),
                    "sum_squared": wsquare.tolist(),
                    "n": wn.tolist(),
                }
            )
        pd.DataFrame(history).to_csv(folder / "epochs.csv", index=False)
        print("EPOCH", recipe["method"], seed, row, flush=True)
        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch + 1
            state = {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "input_dim": int(train.shape[1]),
                "hidden": list(config.hidden),
                "normed": recipe.get("normed", False),
                "lsr": recipe.get("lsr", False),
                "link": link,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, folder / "model.pt")
            if weight_net is not None:
                torch.save(weight_net.state_dict(), folder / "weight_net.pt")
        min_epoch = (
            int(config.epochs * 0.8) + 1
            if recipe["method"] in ["drw", "ldam_drw"]
            else 1
        )
        if epoch + 1 >= min_epoch and epoch + 1 - best_epoch >= config.patience:
            break
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    np.save(
        folder / "validation_score.npy",
        research_predict_nn(model, validation, device, config.batch_size, link),
    )
    research_json(
        folder / "weight_observations.json",
        {
            "kind": "Observed minibatch weights; temporal observations are not global static ESS",
            "epochs": epoch_weight_stats,
            "zero_weight_batches": zero_weight_batches,
        },
    )
    result = {
        "model_file": "model.pt",
        "initial_model_hash": initial_hash,
        "best_epoch": best_epoch,
        "score_link": link,
        "comparison_group": (
            "lsr_latent_mlp"
            if recipe["family"] == "lsr_nn"
            else ("normalized_head_mlp" if recipe.get("normed") else "shared_mlp")
        ),
    }
    del model, optimizer, loader, tensors, meta_x, meta_y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def research_fit_xgboost(study, recipe, seed, folder):
    import xgboost as xgb

    config = study.config
    dtrain = xgb.DMatrix(
        study.feature("train"), label=study.train_rows["y"], nthread=config.threads
    )
    dvalidation = xgb.DMatrix(
        study.feature("validation"),
        label=study.validation_rows["y"],
        nthread=config.threads,
    )
    objective = None
    if recipe["method"] == "xgb_weighted":
        module = research_source_module(
            study.project, "imxgb", "imxgboost/weighted_loss.py"
        )
        objective = module.Weight_Binary_Cross_Entropy(
            recipe["alpha"]
        ).weighted_binary_cross_entropy
    elif recipe["method"] == "xgb_focal":
        module = research_source_module(
            study.project, "imxgb", "imxgboost/focal_loss.py"
        )
        objective = module.Focal_Binary_Loss(recipe["gamma"]).focal_binary_object

    def metric(margins, matrix):
        return "unweighted_AP", float(
            average_precision_score(matrix.get_label(), expit(margins))
        )

    params = {
        "objective": "binary:logitraw",
        "base_score": 0.0,
        "eta": 0.05,
        "max_depth": recipe["depth"],
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "lambda": 1.0,
        "nthread": config.threads,
        "seed": seed,
        "disable_default_eval_metric": 1,
    }
    history = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=config.tree_rounds,
        obj=objective,
        evals=[(dvalidation, "validation")],
        custom_metric=metric,
        maximize=True,
        early_stopping_rounds=min(30, config.tree_rounds),
        evals_result=history,
        verbose_eval=25,
    )
    selected_rounds = int(booster.best_iteration) + 1
    booster = booster[:selected_rounds]
    booster.save_model(folder / "model.ubj")
    margins = booster.predict(dvalidation, output_margin=True)
    np.save(folder / "validation_score.npy", expit(margins).astype(np.float32))
    research_json(folder / "history.json", history)
    static = (
        np.where(study.train_rows["y"] == 1, recipe["alpha"], 1.0)
        if recipe["method"] == "xgb_weighted"
        else np.ones(len(study.train_rows["y"]))
    )
    research_json(
        folder / "weight_summary.json",
        research_weight_summary(static, study.train_rows["y"]),
    )
    return {
        "model_file": "model.ubj",
        "best_rounds": selected_rounds,
        "score_link": "sigmoid_raw_margin",
        "comparison_group": "xgboost_raw",
    }


def research_fit_lgbm(study, recipe, seed, folder):
    import lightgbm as lgb

    weights = (
        np.load(study.output / "auxiliary/weights.npy")
        if recipe["method"].endswith("weighted")
        else np.ones(len(study.train_rows["y"]))
    )
    if recipe["method"] == "lsr_lgbm_balanced":
        weights = np.asarray(study.counts.sum() / (2 * study.counts))[
            study.train_rows["y"]
        ]
    model = lgb.LGBMClassifier(
        n_estimators=study.config.tree_rounds,
        num_leaves=recipe["leaves"],
        learning_rate=0.05,
        min_child_samples=100,
        reg_lambda=1.0,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        n_jobs=study.config.threads,
        random_state=seed,
        verbosity=-1,
        metric="None",
        deterministic=True,
        force_col_wise=True,
    )

    def metric(labels, scores):
        return "unweighted_AP", average_precision_score(labels, scores), True

    model.fit(
        study.feature("train"),
        study.train_rows["y"],
        sample_weight=weights,
        eval_set=[(study.feature("validation"), study.validation_rows["y"])],
        eval_metric=metric,
        callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)],
    )
    model.booster_.save_model(
        str(folder / "model.txt"), num_iteration=model.best_iteration_
    )
    np.save(
        folder / "validation_score.npy",
        model.predict_proba(study.feature("validation"))[:, 1].astype(np.float32),
    )
    research_json(
        folder / "weight_summary.json",
        research_weight_summary(weights, study.train_rows["y"]),
    )
    return {
        "model_file": "model.txt",
        "best_rounds": int(model.best_iteration_),
        "score_link": "probability",
        "comparison_group": "lsr_raw_lightgbm",
    }


def research_lsr_classes(project):
    """Load the unmodified author VAE, diffusion network, and EDM loss classes."""
    vae = research_source_module(project, "lsr", "tabsyn/vae/model.py")
    repo = Path(project) / "external/research_sources/lsr"
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    try:
        import importlib

        diffusion = importlib.import_module("tabsyn.model")
        utils = importlib.import_module("tabsyn.diffusion_utils")
        baf.require(
            Path(diffusion.__file__).resolve() == (repo / "tabsyn/model.py").resolve(),
            "Another tabsyn module is already imported",
        )
        baf.require(
            Path(utils.__file__).resolve()
            == (repo / "tabsyn/diffusion_utils.py").resolve(),
            "Another diffusion module is already imported",
        )
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous
    return vae, diffusion, utils


def research_lsr_encode(study, partition, encoder=None):
    folder = study.output / "auxiliary"
    target = folder / (partition + "_latent.npy")
    if encoder is None:
        vae, _, _ = research_lsr_classes(study.project)
        architecture = research_read(folder / "architecture.json")
        encoder = vae.Encoder_model(
            2,
            architecture["numeric"],
            architecture["cardinalities"],
            4,
            n_head=1,
            factor=32,
        ).to(study.device)
        encoder.load_state_dict(
            torch.load(
                folder / "encoder.pt", map_location=study.device, weights_only=True
            )
        )
    encoder.eval()
    num = research_array(study.cache, partition, "lsr_num")
    cat = research_array(study.cache, partition, "lsr_cat")
    dim = (num.shape[1] + cat.shape[1]) * 4
    result = np.lib.format.open_memmap(
        target, mode="w+", dtype=np.float32, shape=(len(num), dim)
    )
    with torch.no_grad():
        for start in range(0, len(num), study.config.auxiliary_batch_size):
            stop = start + study.config.auxiliary_batch_size
            xnum = torch.as_tensor(
                np.array(num[start:stop], copy=True), device=study.device
            )
            xcat = torch.as_tensor(
                np.array(cat[start:stop], copy=True), device=study.device
            )
            result[start:stop] = encoder(xnum, xcat)[:, 1:, :].flatten(1).cpu().numpy()
    result.flush()
    return target


def research_lsr_vae_loss(model, num, cat):
    reconstruction_num, reconstruction_cat, mean, logvar = model(num, cat)
    mse = (num - reconstruction_num).square().mean()
    ce = torch.stack(
        [
            nn.functional.cross_entropy(logits, cat[:, i])
            for i, logits in enumerate(reconstruction_cat)
        ]
    ).mean()
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    return mse, ce, kl


def research_fit_lsr_auxiliary(study):
    config = study.config
    device = study.device
    folder = study.output / "auxiliary"
    folder.mkdir(exist_ok=True)
    complete = folder / "AUXILIARY_COMPLETE.json"
    if complete.is_file():
        for name, digest in research_read(complete)["hashes"].items():
            baf.require(
                baf.sha256(folder / name) == digest, f"LSR artifact changed: {name}"
            )
        print("LSR_AUXILIARY_REUSED", flush=True)
        return
    started = time.perf_counter()
    vae, diffusion, _ = research_lsr_classes(study.project)
    set_seed(config.auxiliary_seed)
    num = research_array(study.cache, "train", "lsr_num")
    cat = research_array(study.cache, "train", "lsr_cat")
    cardinalities = study.data["cardinalities"]
    architecture = {
        "numeric": int(num.shape[1]),
        "cardinalities": cardinalities,
        "layers": 2,
        "token_dim": 4,
        "factor": 32,
        "score_width": config.diffusion_width,
    }
    research_json(folder / "architecture.json", architecture)
    model = vae.Model_VAE(
        2, num.shape[1], cardinalities, 4, n_head=1, factor=32, bias=True
    ).to(device)
    if not (folder / "VAE_COMPLETE.json").is_file():
        train = TensorDataset(
            torch.from_numpy(np.array(num, copy=True)),
            torch.from_numpy(np.array(cat, copy=True)),
        )
        loader = DataLoader(
            train,
            batch_size=config.auxiliary_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.auxiliary_seed),
            num_workers=0,
        )
        valid_num = torch.as_tensor(
            np.array(research_array(study.cache, "meta", "lsr_num"), copy=True),
            device=device,
        )
        valid_cat = torch.as_tensor(
            np.array(research_array(study.cache, "meta", "lsr_cat"), copy=True),
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best = float("inf")
        best_epoch = 0
        beta = 0.01
        history = []
        for epoch in range(config.vae_epochs):
            model.train()
            total = 0.0
            count = 0
            for xnum, xcat in loader:
                mse, ce, kl = research_lsr_vae_loss(
                    model, xnum.to(device), xcat.to(device)
                )
                loss = mse + ce + beta * kl
                baf.require(torch.isfinite(loss).item(), "Non-finite LSR VAE loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xnum)
                count += len(xnum)
            model.eval()
            with torch.no_grad():
                _, validation_ce, _ = research_lsr_vae_loss(model, valid_num, valid_cat)
            value = float(validation_ce.item())
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "meta_reconstruction_ce": value,
                    "beta": beta,
                }
            )
            pd.DataFrame(history).to_csv(folder / "vae_history.csv", index=False)
            print("LSR_VAE_EPOCH", history[-1], flush=True)
            if value < best:
                best, best_epoch = value, epoch + 1
                torch.save(model.state_dict(), folder / "vae.pt")
            elif epoch + 1 - best_epoch == 10:
                beta = max(1e-5, beta * 0.7)
            if epoch + 1 - best_epoch >= config.auxiliary_patience:
                break
        research_json(
            folder / "VAE_COMPLETE.json",
            {"best_epoch": best_epoch, "model_sha256": baf.sha256(folder / "vae.pt")},
        )
        del optimizer, train, loader, valid_num, valid_cat
    marker = research_read(folder / "VAE_COMPLETE.json")
    baf.require(
        baf.sha256(folder / "vae.pt") == marker["model_sha256"],
        "VAE checkpoint changed",
    )
    model.load_state_dict(
        torch.load(folder / "vae.pt", map_location=device, weights_only=True)
    )
    encoder = vae.Encoder_model(
        2, num.shape[1], cardinalities, 4, n_head=1, factor=32
    ).to(device)
    encoder.load_weights(model)
    torch.save(encoder.state_dict(), folder / "encoder.pt")
    for partition in ["train", "meta", "validation"]:
        research_lsr_encode(study, partition, encoder)
    del model, encoder
    gc.collect()
    latent = np.load(folder / "train_latent.npy", mmap_mode="r")
    global_center = np.asarray(latent).mean(axis=0, dtype=np.float64).astype(np.float32)
    centers = []
    for label in [0, 1]:
        values = np.array(latent[study.train_rows["y"] == label], copy=True)
        center = (
            values.mean(axis=0).astype(np.float32)
            if config.lsr_centering == "author"
            else global_center
        )
        centers.append(center)
        values = (values - center) / 2.0
        set_seed(config.auxiliary_seed + label)
        network = diffusion.Model(
            diffusion.MLPDiffusion(latent.shape[1], config.diffusion_width),
            hid_dim=latent.shape[1],
        ).to(device)
        checkpoint = folder / f"diffusion_{label}.pt"
        marker_path = folder / f"DIFFUSION_{label}_COMPLETE.json"
        if not marker_path.is_file():
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
            loader = DataLoader(
                torch.from_numpy(values),
                batch_size=config.auxiliary_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(config.auxiliary_seed + label),
                num_workers=0,
            )
            best, best_epoch = float("inf"), 0
            history = []
            for epoch in range(config.diffusion_epochs):
                network.train()
                total, count = 0.0, 0
                for batch in loader:
                    loss = network(batch.to(device))
                    baf.require(
                        torch.isfinite(loss).item(), "Non-finite LSR diffusion loss"
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                    count += len(batch)
                value = total / count
                history.append(
                    {"epoch": epoch + 1, "class": label, "train_edm_loss": value}
                )
                pd.DataFrame(history).to_csv(
                    folder / f"diffusion_{label}_history.csv", index=False
                )
                print("LSR_DIFFUSION_EPOCH", history[-1], flush=True)
                if value < best:
                    best, best_epoch = value, epoch + 1
                    torch.save(network.state_dict(), checkpoint)
                if epoch + 1 - best_epoch >= config.auxiliary_patience:
                    break
            research_json(
                marker_path,
                {"best_epoch": best_epoch, "model_sha256": baf.sha256(checkpoint)},
            )
            del optimizer, loader
        baf.require(
            baf.sha256(checkpoint) == research_read(marker_path)["model_sha256"],
            "Diffusion checkpoint changed",
        )
        del network, values
        gc.collect()
    np.savez(
        folder / "centers.npz",
        global_center=global_center,
        class0=centers[0],
        class1=centers[1],
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sigma = torch.linspace(0, 1, config.score_steps, device=device)
    sigma = (80.0 ** (1 / 7) + sigma * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))).pow(7)
    for label in [0, 1]:
        target = folder / f"score_error_{label}.npy"
        done = folder / f"SCORE_{label}_COMPLETE.json"
        if done.is_file():
            baf.require(
                baf.sha256(target) == research_read(done)["sha256"],
                "Score errors changed",
            )
            continue
        set_seed(config.auxiliary_seed + label)
        network = diffusion.Model(
            diffusion.MLPDiffusion(latent.shape[1], config.diffusion_width),
            hid_dim=latent.shape[1],
        ).to(device)
        network.load_state_dict(
            torch.load(
                folder / f"diffusion_{label}.pt", map_location=device, weights_only=True
            )
        )
        network.eval()
        result = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float64, shape=(len(latent),)
        )
        with torch.no_grad():
            for start in range(0, len(latent), config.score_batch_size):
                batch = torch.as_tensor(
                    np.array(
                        latent[start : start + config.score_batch_size], copy=True
                    ),
                    device=device,
                )
                batch = (batch - torch.as_tensor(global_center, device=device)) / 2.0
                accumulated = torch.zeros(
                    len(batch), device=device, dtype=torch.float64
                )
                for current_sigma in sigma:
                    edm_weight = (current_sigma.square() + 0.25) / (
                        current_sigma * 0.5
                    ).square()
                    for repeat in range(
                        0, config.score_noise_repeats, config.score_noise_chunk
                    ):
                        repetitions = min(
                            config.score_noise_chunk,
                            config.score_noise_repeats - repeat,
                        )
                        clean = batch.repeat_interleave(repetitions, dim=0)
                        scales = current_sigma.expand(len(clean))
                        noisy = clean + torch.randn_like(clean) * current_sigma
                        reconstructed = network.denoise_fn_D(noisy, scales)
                        error = (
                            (reconstructed - clean)
                            .square()
                            .mean(dim=1)
                            .reshape(len(batch), repetitions)
                        )
                        accumulated += error.double().sum(dim=1) * edm_weight.double()
                result[start : start + len(batch)] = (
                    (accumulated / (config.score_steps * config.score_noise_repeats))
                    .cpu()
                    .numpy()
                )
                print(
                    "LSR_SCORE_BATCH",
                    label,
                    min(start + len(batch), len(latent)),
                    "/",
                    len(latent),
                    flush=True,
                )
        result.flush()
        del result, network
        research_json(done, {"sha256": baf.sha256(target)})
    error0 = np.load(folder / "score_error_0.npy")
    error1 = np.load(folder / "score_error_1.npy")
    difference = np.where(study.train_rows["y"] == 0, error1 - error0, error0 - error1)
    log_weight = -difference / 3.0
    weights = np.exp(log_weight - logsumexp(log_weight)) * len(log_weight)
    baf.require(np.isfinite(weights).all() and weights.sum() > 0, "Invalid LSR weights")
    np.save(folder / "weights.npy", weights)
    research_json(
        folder / "weight_summary.json",
        research_weight_summary(weights, study.train_rows["y"]),
    )
    metadata = {
        "auxiliary_seed": config.auxiliary_seed,
        "auxiliary_seconds": time.perf_counter() - started,
        "centering": config.lsr_centering,
        "centering_note": "author: class-centered training, globally centered scoring, as source; shared: explicit alternative transfer",
        "streaming_note": "EDM-weighted squared errors reduced over latent dimensions/noise/scales before saving; same arithmetic mean, different RNG draw order from the author tensor allocation",
        "budgets": {
            "vae_epochs": config.vae_epochs,
            "diffusion_epochs": config.diffusion_epochs,
            "noise_repeats": config.score_noise_repeats,
            "sigma_steps": config.score_steps,
        },
        "scope": "Author VAE/diffusion classes; BAF train-only preprocessing, balanced in-train meta validation, capped training budget; not original reported dataset scores",
    }
    research_json(folder / "auxiliary_metadata.json", metadata)
    metadata["hashes"] = {
        p.relative_to(folder).as_posix(): baf.sha256(p)
        for p in folder.rglob("*")
        if p.is_file() and p.name != complete.name
    }
    research_json(complete, metadata)
    print("LSR_AUXILIARY_COMPLETE", flush=True)


def research_create_campaign(
    project, data_path, config, resume=None, dataset_label="BAF_Base", suites=None
):
    config.validate()
    project = Path(project).resolve()
    research_sources(project)
    cache = research_prepare(project, data_path, config, dataset_label)
    campaign = (
        Path(resume).resolve()
        if resume
        else baf.new_directory(
            project / "results/research_campaigns", "weighting_research"
        )
    )
    campaign.mkdir(parents=True, exist_ok=True)
    suites = list(suites or ["losses", "meta", "lsr", "xgboost"])
    baf.require(
        len(suites) == len(set(suites))
        and all(s in ["losses", "meta", "lsr", "xgboost"] for s in suites),
        "Invalid suite selection",
    )
    baf.require(
        "meta" not in suites or "losses" in suites,
        "Meta methods require the shared CE baseline from losses",
    )
    content = {
        "version": RESEARCH_VERSION,
        "project": str(project),
        "cache": str(cache),
        "config": asdict(config),
        "suites": suites,
        "mode": config.mode,
        "source": {
            name: baf.sha256(project / "src" / name) for name in ["baf.py", "gru.py"]
        },
    }
    path = campaign / "campaign.json"
    if path.is_file():
        baf.require(
            research_hash(research_read(path)) == research_hash(content),
            "Campaign configuration changed",
        )
    else:
        research_json(path, content)
    print("CAMPAIGN_READY", campaign)
    return campaign


def research_resolve_campaign(project, selected=None):
    if selected:
        campaign = Path(selected).resolve()
        baf.require((campaign / "campaign.json").is_file(), "Selected campaign missing")
        return campaign
    candidates = sorted(
        (Path(project) / "results/research_campaigns").glob("*/campaign.json")
    )
    if len(candidates) != 1:
        raise ValueError(
            "Set CAMPAIGN_DIR explicitly; available campaigns:\n"
            + "\n".join(str(p.parent) for p in candidates)
        )
    return candidates[0].parent


def research_campaign_config(campaign):
    return ResearchConfig(**research_read(Path(campaign) / "campaign.json")["config"])


def research_launch_suite(campaign, suite):
    """Execute transparent recipe cells in a fresh process and save real outputs."""
    import nbformat

    campaign = Path(campaign).resolve()
    info = research_read(campaign / "campaign.json")
    project = Path(info["project"])
    baf.require(suite in info["suites"], "Unknown campaign suite")
    for name, digest in info["source"].items():
        baf.require(
            baf.sha256(project / "src" / name) == digest, "Campaign source changed"
        )
    root = campaign / "suites" / suite
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    attempt.mkdir()
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Actual weighting research execution\n\nThis notebook contains real cell outputs; only validation is used during training."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nsys.path.insert(0, "
            + repr(str(project / "src"))
            + ")\nimport gru\n"
        ),
        nbformat.v4.new_code_cell(
            "CAMPAIGN = Path("
            + repr(str(campaign))
            + ')\ninfo = gru.research_read(CAMPAIGN / "campaign.json")\nconfig = gru.ResearchConfig(**info["config"])\nstudy = gru.ResearchStudy(info["project"], info["cache"], '
            + repr(suite)
            + ', config, CAMPAIGN / "suites" / '
            + repr(suite)
            + ")\n"
        ),
        nbformat.v4.new_code_cell(
            "table = study.run()\nprint(table.to_string(index=False))\n"
        ),
    ]
    path = attempt / "training_executed.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        },
    )
    nbformat.write(notebook, path)
    command = "import sys; sys.path.insert(0, sys.argv[1]); import baf; baf.execute_weighting_recipe(sys.argv[2])"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    state = {"state": "running", "suite": suite, "attempt": str(attempt)}
    research_json(attempt / "status.json", state)
    process = None
    try:
        with (attempt / "training.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", command, str(project / "src"), str(path)],
                cwd=project,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(
                    "Research suite failed; keep this attempt and resume completed fits"
                )
        state["state"] = "completed"
    except BaseException as error:
        state["state"] = "failed_or_interrupted"
        state["error"] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        research_json(attempt / "status.json", state)
    print("RESEARCH_SUITE_COMPLETE", suite, root)
    return root


def research_suite_overview(root):
    root = Path(root)
    baf.require((root / "FROZEN.json").is_file(), "Suite has not frozen all models")
    return pd.read_csv(root / "validation_summary.csv")


def research_predict_record(study, record, partition):
    recipe = record["recipe"]
    folder = study.output / "fits" / record["name"]
    feature = recipe["feature"]
    if (
        feature == "latent"
        and not (study.output / "auxiliary" / (partition + "_latent.npy")).is_file()
    ):
        research_lsr_encode(study, partition)
    array = study.feature(partition, feature)
    if recipe["family"] in ["nn", "lsr_nn"]:
        saved = torch.load(
            folder / record["model_file"], map_location="cpu", weights_only=False
        )
        model = ResearchMLP(
            saved["input_dim"], saved["hidden"], saved["normed"], saved["lsr"]
        ).to(study.device)
        model.load_state_dict(saved["state_dict"])
        scores = research_predict_nn(
            model, array, study.device, study.config.batch_size, saved["link"]
        )
        del model
    elif recipe["family"] == "xgboost":
        import xgboost as xgb

        model = xgb.Booster(params={"nthread": study.config.threads})
        model.load_model(folder / record["model_file"])
        scores = expit(
            model.predict(
                xgb.DMatrix(array, nthread=study.config.threads), output_margin=True
            )
        ).astype(np.float32)
    else:
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(folder / record["model_file"]))
        scores = model.predict(array, num_threads=study.config.threads).astype(
            np.float32
        )
    return scores


def research_threshold(record):
    return (
        float("inf")
        if record.get("threshold_is_posinf")
        else float(record["threshold"])
    )


def research_weighted_ap_plan(labels, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_score = np.asarray(scores)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(order) - 1]
    return order, ends, np.asarray(labels, dtype=np.float64)[order]


def research_weighted_ap(plan, counts):
    order, ends, positives = plan
    values = counts[order]
    tp = np.cumsum(values * positives)[ends]
    total = np.cumsum(values)[ends]
    if tp[-1] == 0:
        return 0.0
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    return float(np.sum(np.diff(np.r_[0.0, tp]) * precision) / tp[-1])


def research_paired_bootstrap(labels, pairs, repeats=1000, seed=9026):
    """Paired row bootstrap, averaging paired seeds; exact weighted AP handles ties."""
    labels = np.asarray(labels)
    plans = [
        (research_weighted_ap_plan(labels, a), research_weighted_ap_plan(labels, b))
        for a, b in pairs
    ]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        counts = rng.multinomial(
            len(labels), np.full(len(labels), 1 / len(labels))
        ).astype(np.float64)
        values.append(
            float(
                np.mean(
                    [
                        research_weighted_ap(right, counts)
                        - research_weighted_ap(left, counts)
                        for left, right in plans
                    ]
                )
            )
        )
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return {
        "delta_ap_ci_low": float(low),
        "delta_ap_bootstrap_median": float(median),
        "delta_ap_ci_high": float(high),
        "bootstrap_repeats": repeats,
        "bootstrap_kind": "paired IID rows; same resample across seeds; conditional on fitted selected models",
    }


def research_evaluate_campaign(campaign):
    campaign = Path(campaign).resolve()
    info = research_read(campaign / "campaign.json")
    config = ResearchConfig(**info["config"])
    root = campaign / "evaluation"
    root.mkdir(exist_ok=True)
    if (root / "EVALUATION_COMPLETE.json").is_file():
        research_verify_run(root)
        print("SAVED_TEST_EVALUATION_REUSED; no model selection or training repeated")
        return pd.read_csv(root / "research_table.csv")
    sealed = {}
    records = []
    for suite in info["suites"]:
        path = campaign / "suites" / suite
        baf.require(
            (path / "FROZEN.json").is_file(),
            f"Finish suite {suite} before opening test",
        )
        frozen = research_read(path / "FROZEN.json")
        baf.require(
            baf.sha256(path / "manifest.json") == frozen["manifest_sha256"],
            "Frozen suite manifest changed",
        )
        sealed[suite] = baf.sha256(path / "FROZEN.json")
        for item in frozen["final"]:
            fit_root = path / "fits" / item["name"]
            baf.require(
                baf.sha256(fit_root / "COMPLETE.json") == item["record_sha256"],
                "Frozen fit record changed",
            )
            record = research_read(fit_root / "COMPLETE.json")
            for name, digest in record["hashes"].items():
                baf.require(
                    baf.sha256(fit_root / name) == digest,
                    "Fit artifact changed before test",
                )
            records.append((suite, record))
    seal = {
        "campaign_sha256": baf.sha256(campaign / "campaign.json"),
        "frozen_suites": sealed,
        "rule": "All requested methods/configurations and validation thresholds frozen before test stage; Base test is previously inspected development data",
    }
    if (root / "SEALED.json").exists():
        baf.require(
            research_read(root / "SEALED.json") == seal, "Evaluation seal changed"
        )
    else:
        research_json(root / "SEALED.json", seal)
    print("ENTIRE_RESEARCH_TABLE_SEALED_BEFORE_TEST", len(records), flush=True)
    rows = research_rows(info["cache"], "test")
    shutil.copy2(Path(info["cache"]) / "test/rows.npz", root / "test_rows.npz")
    all_metrics = []
    scored = {}
    for suite in info["suites"]:
        study = ResearchStudy(
            info["project"], info["cache"], suite, config, campaign / "suites" / suite
        )
        for _, record in [item for item in records if item[0] == suite]:
            stem = suite + "__" + record["name"]
            score_path = root / (stem + "_score.npy")
            scores = research_predict_record(study, record, "test")
            np.save(score_path, scores)
            threshold = research_threshold(record)
            values = baf.evaluate(rows["y"], scores, threshold, config.max_fpr)
            values.update(
                research_fairness(rows["y"], scores, rows["customer_age"], threshold)
            )
            row = {
                "suite": suite,
                "method": record["recipe"]["method"],
                "seed": record["seed"],
                "comparison_group": record["comparison_group"],
                "threshold": threshold,
                "fit_seconds": record["fit_seconds"],
                "mode": config.mode,
                **values,
            }
            all_metrics.append(row)
            scored[(row["method"], row["seed"])] = scores
            print("TEST_METRICS", row, flush=True)
    frame = pd.DataFrame(all_metrics)
    frame.to_csv(root / "test_metrics.csv", index=False)
    metrics = [
        "ap",
        "recall_frozen_threshold",
        "fpr_frozen_threshold",
        "precision_frozen_threshold",
        "older_fpr",
        "younger_fpr",
        "fpr_gap_older_minus_younger",
        "fpr_ratio_min_max",
        "fit_seconds",
    ]
    summary = frame.groupby(["comparison_group", "method"])[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(root / "research_table.csv", index=False)
    reference = {
        "shared_mlp": "ce",
        "normalized_head_mlp": "norm_ce",
        "lsr_latent_mlp": "lsr_nn_none",
        "lsr_raw_lightgbm": "lsr_lgbm_none",
        "xgboost_raw": "xgb_none",
    }
    comparisons = []
    for group, group_rows in frame.groupby("comparison_group"):
        baseline = reference[group]
        methods = group_rows.method.unique()
        baf.require(baseline in methods, f"Missing within-backbone baseline {baseline}")
        for method in methods:
            if method == baseline:
                continue
            paired = [
                (scored[(baseline, seed)], scored[(method, seed)])
                for seed in config.seeds
            ]
            delta = float(
                np.mean(
                    [
                        average_precision_score(rows["y"], b)
                        - average_precision_score(rows["y"], a)
                        for a, b in paired
                    ]
                )
            )
            comparison = {
                "comparison_group": group,
                "reference": baseline,
                "method": method,
                "delta_ap_mean": delta,
                **research_paired_bootstrap(
                    rows["y"], paired, config.bootstrap_repeats
                ),
            }
            comparisons.append(comparison)
            print("PAIRED_BOOTSTRAP", comparison, flush=True)
    pd.DataFrame(comparisons).to_csv(root / "paired_bootstrap.csv", index=False)
    research_json(
        root / "EVALUATION_COMPLETE.json",
        {
            "sealed_sha256": baf.sha256(root / "SEALED.json"),
            "models": len(frame),
            "mode": config.mode,
            "hashes": {
                p.name: baf.sha256(p)
                for p in root.iterdir()
                if p.is_file() and p.name != "EVALUATION_COMPLETE.json"
            },
        },
    )
    print("RESEARCH_TABLE_READY", root)
    return summary


def research_launch_evaluation(campaign):
    """Run final inference/table generation in a saved child notebook, once."""
    import nbformat

    campaign = Path(campaign).resolve()
    info = research_read(campaign / "campaign.json")
    project = Path(info["project"])
    root = campaign / "evaluation"
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    attempt.mkdir()
    code = (
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0, " + repr(str(project / "src")) + ")\n"
        "import gru\n"
        "table = gru.research_evaluate_campaign(Path(" + repr(str(campaign)) + "))\n"
        "print(table.to_string(index=False))\n"
    )
    path = attempt / "evaluation_executed.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "# Actual test inference and comparison table\n\nAll selected models and thresholds must already be frozen."
                ),
                nbformat.v4.new_code_cell(code),
            ],
            metadata={
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                }
            },
        ),
        path,
    )
    command = "import sys; sys.path.insert(0, sys.argv[1]); import baf; baf.execute_weighting_recipe(sys.argv[2])"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    state = {"state": "running"}
    process = None
    try:
        with (attempt / "evaluation.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", command, str(project / "src"), str(path)],
                cwd=project,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(
                    "Evaluation failed; inspect the saved evaluation notebook"
                )
        state["state"] = "completed"
    except BaseException as error:
        state.update(state="failed_or_interrupted", error=repr(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        research_json(attempt / "status.json", state)
    return root


def research_paper_table(campaign):
    """Join real run metrics to paper provenance; do not invent original scores."""
    campaign = Path(campaign)
    info = research_read(campaign / "campaign.json")
    root = campaign / "evaluation"
    table = pd.read_csv(root / "research_table.csv")
    metadata = {
        "focal": ("Focal Loss for Dense Object Detection", 2017, None),
        "cb_ce": (
            "Class-Balanced Loss Based on Effective Number of Samples",
            2019,
            "cb",
        ),
        "cb_focal": (
            "Class-Balanced Loss Based on Effective Number of Samples",
            2019,
            "cb",
        ),
        "drw": (
            "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss",
            2019,
            "ldam",
        ),
        "ldam": (
            "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss",
            2019,
            "ldam",
        ),
        "ldam_drw": (
            "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss",
            2019,
            "ldam",
        ),
        "lre": ("Learning to Reweight Examples for Robust Deep Learning", 2018, "lre"),
        "mwn": (
            "Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting",
            2019,
            "mwn",
        ),
        "lsr_nn_weighted": (
            "Latent Score-Based Reweighting for Robust Classification on Imbalanced Tabular Data",
            2025,
            "lsr",
        ),
        "lsr_lgbm_weighted": (
            "Latent Score-Based Reweighting for Robust Classification on Imbalanced Tabular Data",
            2025,
            "lsr",
        ),
        "xgb_weighted": ("Imbalance-XGBoost", 2020, "imxgb"),
        "xgb_focal": ("Imbalance-XGBoost", 2020, "imxgb"),
    }
    lock = research_read(Path(info["project"]) / "docs/research_sources.json")
    rows = []
    for record in table.to_dict("records"):
        method = record["method"]
        paper, year, tag = metadata.get(
            method, ("Within-backbone control; not an independent paper", None, None)
        )
        provenance = lock[tag] if tag else {}
        record.update(
            paper=paper,
            paper_year=year,
            author_repo=provenance.get("repo", ""),
            author_commit=provenance.get("commit", ""),
            mode=info["mode"],
            implementation_scope=(
                "Budgeted BAF transfer"
                if tag == "lsr"
                else (
                    "BAF transfer / adapted implementation"
                    if method in metadata
                    else "Control"
                )
            ),
            original_paper_score="Not copied; read the paper and report separately",
            split="train0-5 minus balanced meta; validation6; development test7",
            seeds=json.dumps(info["config"]["seeds"]),
            evidence_path=str(campaign / "suites"),
            interpretation="Observed BAF result; no claim of reproducing original dataset scores",
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    paired = pd.read_csv(root / "paired_bootstrap.csv")
    result = result.merge(paired, how="left", on=["comparison_group", "method"])
    result.to_csv(root / "paper_research_table.csv", index=False, encoding="utf-8-sig")
    return result


def research_verify_review(root):
    """Verify the explicit partial package without requiring omitted models."""
    root = Path(root)
    scope = research_read(root / "review_scope.json")
    baf.require(
        scope["kind"] == "PARTIAL_REVIEW_NOT_FULL_PROOF", "Unknown review scope"
    )
    for entry in scope["retained"]:
        baf.require(
            baf.sha256(root / entry["path"]) == entry["sha256"],
            "Retained review file mismatch: " + entry["path"],
        )
    if (root / "FROZEN.json").exists():
        rows = np.load(root / "evidence/validation_rows.npz")
        for item in research_read(root / "FROZEN.json")["final"]:
            folder = root / "fits" / item["name"]
            record = research_read(folder / "COMPLETE.json")
            score = np.load(folder / "validation_score.npy")
            baf.require(
                np.isclose(
                    average_precision_score(rows["y"], score), record["metrics"]["ap"]
                ),
                "Review validation AP mismatch",
            )
    else:
        _research_verify_test_metrics(root)
    print(
        "PARTIAL_REVIEW_VERIFIED; omitted models and training artifacts were not checked",
        len(scope["omitted"]),
    )


def _research_verify_test_metrics(root):
    rows = np.load(root / "test_rows.npz")
    metrics = pd.read_csv(root / "test_metrics.csv")
    for row in metrics.to_dict("records"):
        name = f"{row['suite']}__{row['method']}_c"
        files = list(root.glob(name + f"*_s{row['seed']}_score.npy"))
        baf.require(len(files) == 1, "Ambiguous saved test score")
        values = baf.evaluate(rows["y"], np.load(files[0]), row["threshold"])
        for key in ["ap", "roc_auc", "recall_frozen_threshold", "fpr_frozen_threshold"]:
            baf.require(
                np.isclose(values[key], row[key]), f"Evaluation metric mismatch: {key}"
            )
    return len(metrics)


def research_verify_run(root):
    """Read-only verification from saved scores/labels; no raw CSV or fitting."""
    root = Path(root)
    if (root / "review_scope.json").is_file():
        return research_verify_review(root)
    if (root / "sha256.json").is_file():
        for name, digest in research_read(root / "sha256.json").items():
            baf.require(
                baf.sha256(root / name) == digest, "Full proof hash mismatch: " + name
            )
    if (root / "FROZEN.json").is_file():
        frozen = research_read(root / "FROZEN.json")
        baf.require(
            baf.sha256(root / "manifest.json") == frozen["manifest_sha256"],
            "Frozen manifest mismatch",
        )
        rows = np.load(root / "evidence/validation_rows.npz")
        for item in frozen["final"]:
            folder = root / "fits" / item["name"]
            baf.require(
                baf.sha256(folder / "COMPLETE.json") == item["record_sha256"],
                "Frozen record mismatch",
            )
            record = research_read(folder / "COMPLETE.json")
            for name, digest in record["hashes"].items():
                baf.require(baf.sha256(folder / name) == digest, "Fit file mismatch")
            score = np.load(folder / "validation_score.npy")
            threshold = baf.select_threshold(
                rows["y"],
                score,
                research_read(root / "manifest.json")["config"]["max_fpr"],
            )
            baf.require(
                np.isclose(threshold, research_threshold(record)),
                "Validation threshold mismatch",
            )
            baf.require(
                np.isclose(
                    average_precision_score(rows["y"], score), record["metrics"]["ap"]
                ),
                "Validation AP mismatch",
            )
        print("RESEARCH_FROZEN_SUITE_VERIFIED", len(frozen["final"]))
    else:
        complete = research_read(root / "EVALUATION_COMPLETE.json")
        for name, digest in complete["hashes"].items():
            baf.require(baf.sha256(root / name) == digest, "Evaluation file mismatch")
        print("RESEARCH_TEST_TABLE_VERIFIED", _research_verify_test_metrics(root))


def research_export_proof(root, project):
    """Export full evidence plus an explicitly partial review ZIP in one pass."""
    import nbformat

    root = Path(root).resolve()
    project = Path(project).resolve()
    research_verify_run(root)
    if (root / "sha256.json").exists():
        (root / "sha256.json").unlink()
    verification = root / "verify.ipynb"
    if not (root / "evidence/src/gru.py").exists():
        (root / "evidence/src").mkdir(parents=True, exist_ok=True)
        for name in ["baf.py", "gru.py"]:
            shutil.copy2(project / "src" / name, root / "evidence/src" / name)
    code = 'from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\nif not (ROOT / "evidence").is_dir():\n    raise RuntimeError("Open the extracted run folder before Run All.")\nsys.path.insert(0, str(ROOT / "evidence/src"))\nimport gru\ngru.research_verify_run(ROOT)\n'
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Verify saved weighting research evidence\n\nRead-only; no training or raw CSV required."
            ),
            nbformat.v4.new_code_cell(code),
        ],
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        },
    )
    nbformat.write(notebook, verification)
    command = "import sys; sys.path.insert(0, sys.argv[1]); import baf; baf.execute_weighting_recipe(sys.argv[2])"
    subprocess.run(
        [sys.executable, "-u", "-c", command, str(project / "src"), str(verification)],
        cwd=root,
        check=True,
    )
    hashes = {
        p.relative_to(root).as_posix(): baf.sha256(p)
        for p in root.rglob("*")
        if p.is_file() and p.name not in ["sha256.json", "review_scope.json"]
    }
    research_json(root / "sha256.json", hashes)
    proof_root = project / "results/proofs"
    proof_root.mkdir(parents=True, exist_ok=True)
    unique = root.name + "_" + uuid.uuid4().hex[:8]
    full = proof_root / (unique + "_full_proof.zip")
    review = proof_root / (unique + "_review.zip")
    with zipfile.ZipFile(full, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in [*hashes, "sha256.json"]:
            archive.write(root / name, Path(unique) / name)
    selected_names = set()
    if (root / "FROZEN.json").exists():
        selected_names = {
            item["name"] for item in research_read(root / "FROZEN.json")["final"]
        }
    retained, omitted = [], []
    for name, digest in hashes.items():
        path = Path(name)
        keep = path.suffix in [".json", ".csv", ".md", ".py", ".ipynb", ".log"]
        keep |= name.startswith("evidence/") and path.suffix == ".npz"
        keep |= name == "test_rows.npz" or (
            len(path.parts) == 1 and name.endswith("_score.npy")
        )
        keep |= (
            name.endswith("validation_score.npy")
            and len(path.parts) >= 3
            and path.parts[1] in selected_names
        )
        entry = {"path": name, "sha256": digest, "bytes": (root / name).stat().st_size}
        (retained if keep else omitted).append(entry)
    scope = {
        "kind": "PARTIAL_REVIEW_NOT_FULL_PROOF",
        "verification": "verify.ipynb detects this review scope, checks retained hashes and recomputes saved AP/metrics; omitted models cannot be verified or used for inference",
        "retained": retained,
        "omitted": omitted,
    }
    with zipfile.ZipFile(review, "x", zipfile.ZIP_DEFLATED) as archive:
        for entry in retained:
            archive.write(root / entry["path"], Path(unique) / entry["path"])
        archive.writestr(
            str(Path(unique) / "review_scope.json"),
            json.dumps(scope, indent=2, ensure_ascii=False),
        )
    print("FULL_PROOF", full)
    print("PARTIAL_REVIEW", review)
    return {"full_proof": full, "review": review}
