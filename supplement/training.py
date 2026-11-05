"""Reusable supplementary BAF fits and small original-dataset implementation checks.

Existing baf.py/gru.py are imported unchanged. No old campaign manifest is rewritten.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import tarfile
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pandas as pd

import reporting as report


def load_modules(root):
    report.setup(root)
    import baf
    import gru
    return baf, gru


def cache_from_campaign(root, campaign=None):
    _, gru = load_modules(root)
    campaign = gru.research_resolve_campaign(root, campaign)
    plan = report.read_json(campaign / "campaign.json")
    cache = Path(plan["cache"])
    if not cache.exists():
        candidates = list((Path(root) / "results/research_data").glob("*/metadata.json"))
        candidates = [p.parent for p in candidates if report.read_json(p).get("data_sha256") == gru.BAF_RESEARCH_SHA256]
        if len(candidates) != 1:
            raise ValueError("Cache path moved. Correct campaign cache path or select one matching cache.")
        cache = candidates[0]
    gru.research_verify_cache(cache)
    for name, checksum in plan["source"].items():
        if report.digest(Path(root) / "src" / name) != checksum:
            raise ValueError(f"Original campaign source differs: {name}. Restore its original source snapshot before supplementary fits.")
    return campaign, plan, cache


def prepare_study(root, run, campaign=None, batch_size=None):
    import torch
    _, gru = load_modules(root)
    campaign, plan, cache = cache_from_campaign(root, campaign)
    config = gru.ResearchConfig(**plan["config"])
    if config.mode != "full":
        raise ValueError("Supplementary BAF runs require the original full-data campaign")
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)
    torch.set_num_threads(config.threads)
    study = SimpleNamespace(
        project=Path(root), output=Path(run), cache=cache, config=config,
        device=gru.research_device(config), train_rows=gru.research_rows(cache, "train"),
        validation_rows=gru.research_rows(cache, "validation"), meta_rows=gru.research_rows(cache, "meta"),
    )
    study.counts = np.bincount(study.train_rows["y"], minlength=2)
    study.feature = lambda part, feature="dense": gru.research_array(cache, part, feature)
    report.save_json(Path(run) / "data_contract.json", {
        "campaign": str(campaign), "cache": str(cache), "metadata": report.read_json(cache / "metadata.json"),
        "cache_complete_sha256": report.digest(cache / "COMPLETE.json"),
        "rule": "Same core train/meta/validation/test cache as notebooks 11–16; no new preprocessing",
        "training_batch_size": config.batch_size, "device": str(study.device),
    })
    return study


def sampled_indices(y, method, ratio=0.1, seed=42):
    """ratio = minority/majority after resampling. Uses only training labels."""
    y = np.asarray(y)
    if not 0 < ratio <= 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Binary labels and ratio in (0,1] required")
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(negative) <= len(positive):
        raise ValueError("Class 1 must be the minority")
    rng = np.random.default_rng(seed)
    if method == "rus":
        target = min(len(negative), int(np.floor(len(positive) / ratio)))
        indices = np.r_[rng.choice(negative, target, replace=False), positive]
    elif method == "ros":
        target = max(len(positive), int(np.ceil(len(negative) * ratio)))
        indices = np.r_[negative, positive, rng.choice(positive, target - len(positive), replace=True)]
    elif method in ["none", "balanced"]:
        indices = np.arange(len(y))
    else:
        raise ValueError(method)
    return indices[rng.permutation(len(indices))]


def _fit(study, recipe, seed, kind):
    import torch
    _, gru = load_modules(study.project)
    name = f"{recipe['method']}_c{recipe['candidate']:02d}_s{seed}"
    folder = study.output / "fits" / name
    marker = folder / "COMPLETE.json"
    if marker.exists():
        record = report.read_json(marker)
        for path, checksum in record["hashes"].items():
            if report.digest(folder / path) != checksum:
                raise ValueError(f"Changed fit: {folder / path}")
        print("FIT_REUSED", name, flush=True)
        return record
    folder.mkdir(parents=True, exist_ok=True)
    print("FIT_BEGIN", name, recipe, flush=True)
    started = time.perf_counter()
    if kind == "ce":
        actual = {**recipe, "method": "ce", "family": "nn", "feature": "dense", "normed": False}
        result = gru.research_fit_neural(study, actual, seed, folder)
        result["comparison_group"] = "meta_matched_batch256"
    else:
        method = recipe["sampling"]
        indices = sampled_indices(study.train_rows["y"], method, recipe["ratio"], seed)
        np.save(folder / "sampled_indices.npy", indices)
        counts = np.bincount(study.train_rows["y"][indices], minlength=2)
        report.save_json(folder / "sampling.json", {
            "method": method, "ratio_requested": recipe["ratio"], "counts": counts.tolist(),
            "rows": len(indices), "unique_rows": len(np.unique(indices)),
            "train_only": True, "inverse_probability_correction": False,
        })
        sampled = SimpleNamespace(**vars(study))
        sampled.train_rows = {key: value[indices] for key, value in study.train_rows.items()}
        sampled.counts = counts
        sampled.feature = lambda part, feature="dense": (
            study.feature(part, feature)[indices] if part == "train" else study.feature(part, feature)
        )
        actual = {**recipe, "method": "lsr_lgbm_balanced" if method == "balanced" else "lsr_lgbm_none"}
        result = gru.research_fit_lgbm(sampled, actual, seed, folder)
        result["comparison_group"] = "sampling_matched_lightgbm"
    score = np.load(folder / "validation_score.npy", allow_pickle=False)
    import baf
    threshold = baf.select_threshold(study.validation_rows["y"], score, 0.05)
    validation = report.metrics(study.validation_rows["y"], score, study.validation_rows["customer_age"], threshold)
    record = {
        "name": name, "recipe": recipe, "seed": int(seed), **result,
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "threshold_is_posinf": bool(np.isposinf(threshold)),
        "validation_ap": validation["ap"], "fit_seconds": time.perf_counter() - started,
    }
    record["hashes"] = {p.relative_to(folder).as_posix(): report.digest(p)
                        for p in folder.rglob("*") if p.is_file() and p != marker}
    report.save_json(marker, record)
    print("FIT_COMPLETE", name, "validation_AP", validation["ap"], flush=True)
    return record


def run_baf(root, run, kind, campaign=None, ratio=0.1):
    """Freeze validation selections before evaluating any new selected model on month 7."""
    study = prepare_study(root, run, campaign, batch_size=256 if kind == "ce" else None)
    seeds = tuple(study.config.seeds)
    if seeds != report.SEEDS:
        raise ValueError("Expected seeds 42,317,2026 from the audited campaign")
    if kind == "ce":
        recipes = [{"method": "ce_batch256", "candidate": i, "lr": float(lr), "family": "nn", "feature": "dense"}
                   for i, lr in enumerate(study.config.learning_rates)]
    elif kind == "sampling":
        recipes = [{"method": "lgbm_" + method, "sampling": method, "ratio": ratio,
                    "candidate": i, "leaves": leaves, "family": "lsr_lgbm", "feature": "dense"}
                   for method in ["none", "balanced", "rus", "ros"] for i, leaves in enumerate([15, 31, 63])]
    else:
        raise ValueError(kind)
    candidates = [_fit(study, recipe, seeds[0], kind) for recipe in recipes]
    selected = {}
    for method in sorted({r["recipe"]["method"] for r in candidates}):
        best = max([r for r in candidates if r["recipe"]["method"] == method],
                   key=lambda r: (r["validation_ap"], -r["recipe"]["candidate"]))
        selected[method] = best["recipe"]
    final = [_fit(study, recipe, seed, kind) for recipe in selected.values() for seed in seeds]
    report.save_json(Path(run) / "FROZEN.json", {
        "selections": selected, "final": [{"name": r["name"], "record_sha256": report.digest(
            Path(run) / "fits" / r["name"] / "COMPLETE.json")} for r in final],
        "rule": "Selected using unweighted month6 AP only; all selected fits complete before test",
    })
    _, gru = load_modules(root)
    rows = gru.research_rows(study.cache, "test")
    output = Path(run) / "evaluation"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "test_rows.npz", **rows)
    records = []
    for record in final:
        score_path = output / (record["name"] + "_score.npy")
        if score_path.exists() and (output / (record["name"] + "_score.json")).exists():
            saved = report.read_json(output / (record["name"] + "_score.json"))
            if report.digest(score_path) != saved["sha256"]:
                raise ValueError("Saved test score changed")
            score = np.load(score_path, allow_pickle=False)
        else:
            score = gru.research_predict_record(study, record, "test")
            np.save(score_path, score)
            report.save_json(output / (record["name"] + "_score.json"), {"sha256": report.digest(score_path)})
        threshold = float("inf") if record["threshold_is_posinf"] else record["threshold"]
        records.append({"suite": "advisor_" + kind, "method": record["recipe"]["method"],
                        "comparison_group": record["comparison_group"], "seed": record["seed"],
                        "threshold": threshold, "fit_seconds": record["fit_seconds"], "mode": "full",
                        **report.metrics(rows["y"], score, rows["customer_age"], threshold)})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "test_metrics.csv", index=False)
    return report.summarize(frame, output)


def verify_sources(root):
    """Verify author files against the lock retained in the original proof."""
    root = Path(root)
    lock = report.read_json(root / "docs/advisor/source_lock.json")
    records = []
    for tag in ["cb", "lre", "mwn"]:
        spec = lock[tag]
        for relative, checksum in spec["files"].items():
            path = root / "docs/advisor/upstream" / tag / relative
            if not path.is_file() or report.digest(path) != checksum:
                raise ValueError(f"Author source missing/changed: {tag}/{relative}")
            records.append({"method": tag, "file": relative, "commit": spec["commit"], "sha256": checksum})
    return pd.DataFrame(records)


def numerical_checks(root, output):
    """Independent analytical and finite-difference checks on the actual port functions."""
    import torch
    from torch import nn
    from torch.func import functional_call
    _, gru = load_modules(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    verify_sources(root).to_csv(output / "source_hash_checks.csv", index=False)
    torch.set_num_threads(2)
    torch.manual_seed(317)
    dtype = torch.float64
    checks = []
    counts = np.array([800, 23])
    for beta in [0, 0.9, 0.9999, 0.999999]:
        expected = (1 - beta) / (1 - np.power(beta, counts)) if beta else np.ones(2)
        expected = expected / expected.sum() * 2
        actual = gru.research_effective_number(counts, beta)
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11):
            raise AssertionError("CB effective-number mismatch")
        checks.append({"check": f"CB_weights_beta_{beta}", "max_abs_error": error, "pass": True})
    logits = torch.randn(13, 2, dtype=dtype, requires_grad=True)
    labels = torch.arange(13) % 2
    weight = torch.tensor(gru.research_effective_number(counts, 0.9999), dtype=dtype)
    reference = -(weight[labels] * logits.log_softmax(1)[torch.arange(13), labels]).mean()
    actual = gru.research_loss(logits, labels, "cb_ce", counts, {"beta": 0.9999}, 0, 20)
    gradient_a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    gradient_b = torch.autograd.grad(reference, logits)[0]
    error = float((gradient_a - gradient_b).abs().max())
    assert error < 1e-10
    checks.append({"check": "CB_CE_loss_and_gradient", "max_abs_error": error, "pass": True})
    target = nn.functional.one_hot(labels, 2).to(dtype)
    sigmoid = logits.sigmoid()
    # Stable BCE times independent sigmoid modulation, as the author's focal loss.
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    modulation = torch.where(target.bool(), (1 - sigmoid).square(), sigmoid.square())
    reference = (weight[labels, None] * modulation * bce).sum() / len(labels)
    actual = gru.research_loss(logits, labels, "cb_focal", counts, {"beta": 0.9999, "gamma": 2.0}, 0, 20)
    error = float((reference - actual).abs().detach())
    assert error < 1e-10
    checks.append({"check": "CB_focal_sigmoid_formula", "max_abs_error": error, "pass": True})
    model = nn.Linear(4, 2).to(dtype)
    x, y = torch.randn(9, 4, dtype=dtype), torch.arange(9) % 2
    mx, my = torch.randn(6, 4, dtype=dtype), torch.arange(6) % 2
    actual, _ = gru.research_lre_weights(model, x, y, mx, my, 0.03)
    meta_gradient = torch.autograd.grad(nn.functional.cross_entropy(model(mx), my), model.parameters())
    alignments = []
    for i in range(len(y)):
        grad = torch.autograd.grad(nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]), model.parameters())
        alignments.append(sum((a * b).sum() for a, b in zip(grad, meta_gradient)))
    reference = torch.stack(alignments).clamp_min(0)
    reference = reference / reference.sum() if reference.sum() else reference
    error = float((actual - reference).abs().max())
    assert error < 1e-9
    checks.append({"check": "LRE_independent_gradient_alignment", "max_abs_error": error, "pass": True})
    # Compare MWN virtual model to the author's unchanged MetaModule.update_params.
    module_path = Path(root) / "docs/advisor/upstream/mwn/resnet.py"
    spec = importlib.util.spec_from_file_location("mwn_original_resnet", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class AuthorLinear(module.MetaModule):
        def __init__(self, base):
            super().__init__()
            self.register_buffer("weight", base.weight.detach().clone().requires_grad_())
            self.register_buffer("bias", base.bias.detach().clone().requires_grad_())
        def named_leaves(self):
            return [("weight", self.weight), ("bias", self.bias)]
        def forward(self, features):
            return nn.functional.linear(features, self.weight, self.bias)
    network = gru.ResearchWeightNet().to(dtype)
    reference_net = deepcopy(network)
    actual_optimizer = torch.optim.SGD(network.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    reference_optimizer = torch.optim.SGD(reference_net.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    author_model = AuthorLinear(model)
    losses = nn.functional.cross_entropy(author_model(x), y, reduction="none")
    weights = reference_net(losses)
    gradients = torch.autograd.grad((losses * weights / weights.sum()).sum(), tuple(author_model.params()), create_graph=True)
    author_model.update_params(lr_inner=0.03, source_params=gradients)
    meta_loss = nn.functional.cross_entropy(author_model(mx), my)
    reference_optimizer.zero_grad()
    meta_loss.backward()
    reference_optimizer.step()
    with torch.no_grad():
        reference = reference_net(nn.functional.cross_entropy(model(x), y, reduction="none"))
        reference /= reference.sum()
    actual, _ = gru.research_mwn_step(model, network, actual_optimizer, x, y, mx, my, 0.03)
    parameter_error = max(float((a - b).abs().max().detach()) for a, b in zip(network.parameters(), reference_net.parameters()))
    error = max(parameter_error, float((actual - reference).abs().max()))
    assert error < 1e-9
    checks.append({"check": "MWN_author_MetaModule_one_step", "max_abs_error": error, "pass": True})
    frame = pd.DataFrame(checks)
    frame.to_csv(output / "numerical_checks.csv", index=False)
    print(frame.to_string(index=False))
    print("NUMERICAL_CHECKS_PASS; this does not reproduce full-paper accuracy")
    return frame


def download_checked(urls, target, md5):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    def valid(path):
        if not path.exists():
            return False
        h = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest() == md5
    if valid(target):
        return target
    errors = []
    for url in urls:
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BAF-academic-verification/1.0"})
            with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as dest:
                total = 0
                last_log = 0
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    dest.write(block)
                    total += len(block)
                    if total - last_log >= 20 * 1024 * 1024:
                        print("DOWNLOAD", target.name, total // (1024 * 1024), "MiB", flush=True)
                        last_log = total
            if not valid(temporary):
                raise ValueError("Dataset MD5 mismatch")
            temporary.replace(target)
            return target
        except Exception as error:
            errors.append(f"{url}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Download failed. Place the verified file at " + str(target) + "\n" + "\n".join(errors))


def original_data(root, dataset, majority=1000, imbalance=100, meta_per_class=20, val_per_class=100):
    """Small long-tailed original benchmark; all meta/val rows come from original train."""
    folder = Path(root) / "data/external/advisor"
    if dataset == "mnist":
        specs = [
            ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 16),
            ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 8),
            ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 16),
            ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 8),
        ]
        arrays = []
        for filename, md5, offset in specs:
            path = download_checked(["https://ossci-datasets.s3.amazonaws.com/mnist/" + filename], folder / filename, md5)
            arrays.append(np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8, offset=offset).copy())
        x, y, test_x, test_y = arrays
        x, test_x = x.reshape(-1, 1, 28, 28), test_x.reshape(-1, 1, 28, 28)
        # The original LRE MNIST imbalance experiment is binary 4 versus 9.
        mask, test_mask = np.isin(y, [4, 9]), np.isin(test_y, [4, 9])
        x, y = x[mask], (y[mask] == 9).astype(np.int64)
        test_x, test_y = test_x[test_mask], (test_y[test_mask] == 9).astype(np.int64)
        mean, std = np.array([0.1307]), np.array([0.3081])
        classes = 2
    elif dataset == "cifar10":
        path = download_checked([
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
        ], folder / "cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a")
        with tarfile.open(path, "r:gz") as archive:
            def batch(name):
                # Read only a known archive member after verification; do not extract archive paths.
                record = pickle.load(archive.extractfile("cifar-10-batches-py/" + name), encoding="bytes")
                return record[b"data"].reshape(-1, 3, 32, 32), np.array(record[b"labels"], dtype=np.int64)
            train = [batch(f"data_batch_{i}") for i in range(1, 6)]
            x, y = np.concatenate([a for a, _ in train]), np.concatenate([b for _, b in train])
            test_x, test_y = batch("test_batch")
        mean, std = np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])
        classes = 10
    else:
        raise ValueError(dataset)
    rng = np.random.default_rng(317)
    selected, meta, validation, counts = [], [], [], []
    for label in range(classes):
        indices = rng.permutation(np.flatnonzero(y == label))
        n = max(1, int(majority * imbalance ** (-label / (classes - 1))))
        if n + meta_per_class + val_per_class > len(indices):
            raise ValueError("Insufficient original training examples")
        counts.append(n)
        meta.extend(indices[:meta_per_class])
        validation.extend(indices[meta_per_class:meta_per_class + val_per_class])
        selected.extend(indices[meta_per_class + val_per_class:meta_per_class + val_per_class + n])
    def normalize(a):
        return ((a.astype(np.float32) / 255 - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
    return {
        "train": (normalize(x[selected]), y[selected]),
        "meta": (normalize(x[meta]), y[meta]),
        "validation": (normalize(x[validation]), y[validation]),
        "test": (normalize(test_x), test_y),
        "indices": {"train": selected, "meta": meta, "validation": validation},
        "counts": counts, "classes": classes, "dataset": dataset,
        "dataset_files": {p.name: report.digest(p) for p in folder.iterdir() if p.is_file() and not p.name.endswith(".part")},
    }


def run_original(root, output, config):
    """Small original-dataset check, not a reproduction of the paper's full training recipe."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    _, gru = load_modules(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    numerical_checks(root, output / "checks")
    torch.set_num_threads(config.get("threads", 4))
    device = torch.device("cuda" if config.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu")
    all_records = []
    for dataset in config.get("datasets", ["mnist", "cifar10"]):
        data = original_data(root, dataset, majority=config.get("majority", 1000),
                             imbalance=config.get("imbalance", 100), meta_per_class=20, val_per_class=100)
        methods = ["ce", "lre"] if dataset == "mnist" else ["ce", "cb_ce", "mwn"]
        classes = data["classes"]
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(data["train"][0].shape[1], 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, classes),
                )
            def forward(self, features):
                return self.network(features)
        def predict(model, values):
            model.eval()
            result = []
            with torch.no_grad():
                for start in range(0, len(values), 256):
                    result.append(model(torch.from_numpy(values[start:start + 256]).to(device)).softmax(1).cpu().numpy())
            return np.concatenate(result)
        for seed in config.get("seeds", [42, 317, 2026]):
            for method in methods:
                folder = output / dataset / f"{method}_s{seed}"
                marker = folder / "COMPLETE.json"
                if marker.exists():
                    record = report.read_json(marker)
                    for name, checksum in record["hashes"].items():
                        if report.digest(folder / name) != checksum:
                            raise ValueError("Changed original-dataset fit")
                    all_records.append(record)
                    print("ORIGINAL_FIT_REUSED", dataset, method, seed)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                gru.set_seed(seed)
                model = SmallCNN().to(device)
                initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
                vnet = gru.ResearchWeightNet().to(device)
                voptimizer = torch.optim.SGD(vnet.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
                loader = DataLoader(TensorDataset(torch.from_numpy(data["train"][0]), torch.from_numpy(data["train"][1])),
                                    batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
                mx, my = torch.from_numpy(data["meta"][0]).to(device), torch.from_numpy(data["meta"][1]).to(device)
                rng = np.random.default_rng(seed)
                best, history, started = -np.inf, [], time.perf_counter()
                print("ORIGINAL_FIT_BEGIN", dataset, method, seed, flush=True)
                for epoch in range(config.get("epochs", 10)):
                    model.train()
                    for features, labels in loader:
                        features, labels = features.to(device), labels.to(device)
                        if method in ["lre", "mwn"]:
                            idx = torch.tensor(rng.choice(len(my), min(64, len(my)), replace=False), device=device)
                            if method == "lre":
                                weights, _ = gru.research_lre_weights(model, features, labels, mx[idx], my[idx], 0.03)
                            else:
                                weights, _ = gru.research_mwn_step(model, vnet, voptimizer, features, labels, mx[idx], my[idx], 0.03)
                            loss = (weights * nn.functional.cross_entropy(model(features), labels, reduction="none")).sum()
                        elif method == "cb_ce":
                            loss = gru.research_loss(model(features), labels, method, np.array(data["counts"]), {"beta": 0.9999}, epoch, config.get("epochs", 10))
                        else:
                            loss = nn.functional.cross_entropy(model(features), labels)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite original-dataset loss")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    probability = predict(model, data["validation"][0])
                    accuracy = accuracy_score(data["validation"][1], probability.argmax(1))
                    history.append({"epoch": epoch + 1, "validation_accuracy": accuracy})
                    if accuracy > best:
                        best = accuracy
                        torch.save(model.state_dict(), folder / "model.pt")
                    report.write_csv(pd.DataFrame(history), folder / "epochs.csv")
                    print("ORIGINAL_EPOCH", dataset, method, seed, epoch + 1, accuracy, flush=True)
                # Fixed method/recipe; test never selects hyperparameters or checkpoint.
                model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
                probability = predict(model, data["test"][0])
                np.savez_compressed(folder / "test_predictions.npz", y=data["test"][1], probability=probability)
                record = {"dataset": dataset, "method": method, "seed": seed,
                          "accuracy": float(accuracy_score(data["test"][1], probability.argmax(1))),
                          "balanced_accuracy": float(balanced_accuracy_score(data["test"][1], probability.argmax(1))),
                          "log_loss": float(log_loss(data["test"][1], probability, labels=list(range(classes)))),
                          "fit_seconds": time.perf_counter() - started, "initial_hash": initial_hash,
                          "device": str(device), "counts": data["counts"],
                          "dataset_files": data["dataset_files"], "split_indices": data["indices"],
                          "scope": "Small original-dataset implementation check; SmallCNN and capped budget differ from full paper"}
                record["hashes"] = {p.name: report.digest(p) for p in folder.iterdir() if p.is_file() and p != marker}
                report.save_json(marker, record)
                all_records.append(record)
        print("ORIGINAL_DATASET_COMPLETE", dataset, flush=True)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ["hashes", "split_indices", "dataset_files", "counts"]} for r in all_records])
    frame.to_csv(output / "original_dataset_metrics.csv", index=False)
    summary = frame.groupby(["dataset", "method"])[["accuracy", "balanced_accuracy", "log_loss", "fit_seconds"]].agg(["mean", "std"])
    summary.to_csv(output / "original_dataset_summary.csv")
    print(summary.to_string())
    return frame
