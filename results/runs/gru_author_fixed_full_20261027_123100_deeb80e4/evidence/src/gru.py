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
            torch.from_numpy(part.fraud_bool.to_numpy(dtype=np.int64)),
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
