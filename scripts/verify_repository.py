"""Verify the published evidence without fitting models or downloading data.

Run from any directory: python -B scripts/verify_repository.py
Optional: --imports checks the training imports in the selected interpreter.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_inventory(root, official):
    """Check every published byte; only the inventory's own digest is omitted."""
    root = Path(root).resolve()
    inventory_name = "FILE_INVENTORY.csv"
    with (root / inventory_name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_path = {}
    for row in rows:
        relative = row["path"]
        require(relative not in by_path, f"Duplicate inventory path: {relative}")
        by_path[relative] = row
    declared = {row["path"] for row in rows if row["include_in_github"] == "True"}
    require(declared == set(official), "Inventory official paths differ from GITHUB_FILES.txt")
    hashes_checked = 0
    for relative in official:
        require(relative in by_path, f"Missing inventory row: {relative}")
        row = by_path[relative]
        path = root / relative
        require(path.resolve().is_relative_to(root), f"Official path outside repository: {relative}")
        require(path.is_file(), f"Missing inventoried file: {relative}")
        require(row["category"] == "official", f"Wrong inventory category: {relative}")
        try:
            expected_size = int(row["size_bytes"])
        except (ValueError, TypeError):
            raise ValueError(f"Missing/invalid inventory byte size: {relative}") from None
        actual_size = path.stat().st_size
        require(actual_size == expected_size,
                f"Inventory size mismatch: {relative}: expected {expected_size}, got {actual_size}")
        expected_hash = row["sha256"]
        if relative == inventory_name:
            require(expected_hash == "", "Inventory self SHA-256 must be empty to avoid self-reference")
            continue
        require(len(expected_hash) == 64 and all(c in "0123456789abcdef" for c in expected_hash),
                f"Missing/invalid inventory SHA-256: {relative}")
        actual_hash = digest(path)
        require(actual_hash == expected_hash,
                f"Inventory SHA-256 mismatch: {relative}: expected {expected_hash}, got {actual_hash}")
        hashes_checked += 1
    require(inventory_name in official, "Inventory must itself be an official file")
    return {"inventory_file_sizes_checked": len(official),
            "inventory_file_sha256_checked": hashes_checked,
            "inventory_self_reference": "Own byte size checked; own SHA-256 intentionally empty; all other official SHA-256 values required"}


def check_historical_survey():
    relative = "references/survey_20261026/paper_survey.csv"
    history = read_json(ROOT / "references/survey_20261026/metadata/artifact_hashes.json")
    expected = history["docs/paper_survey.csv"]
    require((ROOT / relative).stat().st_size == 6673, "Historical survey CSV byte size changed")
    require(digest(ROOT / relative) == expected, "Historical survey CSV hash mismatch; retain original CRLF bytes")
    return {"historical_survey_bytes": 6673, "historical_survey_sha256": expected}


def check_metric_rows(folder, expected_count, research=False):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "supplement"))
    import reporting

    table = pd.read_csv(folder / "test_metrics.csv")
    require(len(table) == expected_count, f"Wrong metric count: {folder}")
    require(not table.duplicated(["suite", "method", "seed"]).any(), "Duplicate score identity")
    require(set(table.seed) == {42, 317, 2026}, "Unexpected seeds")
    with np.load(folder / "test_rows.npz", allow_pickle=False) as saved:
        rows = {key: saved[key] for key in saved.files}
    require(len(rows["y"]) == 96843 and int(rows["y"].sum()) == 1428, "Unexpected BAF test support")
    for row in table.to_dict("records"):
        prefix = (row["suite"] + "__") if research else ""
        matches = list(folder.glob(f"{prefix}{row['method']}_c*_s{int(row['seed'])}_score.npy"))
        require(len(matches) == 1, f"Missing/ambiguous score: {row['method']}/{row['seed']}")
        score = np.load(matches[0], allow_pickle=False)
        actual = reporting.metrics(rows["y"], score, rows["customer_age"], row["threshold"])
        for key, value in actual.items():
            if key in row:
                require(np.isclose(value, row[key], atol=1e-10, rtol=0, equal_nan=True),
                        f"Metric mismatch: {matches[0].name}/{key}")
    return table


def check_results():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

    campaign = ROOT / "results/research_campaigns/weighting_research_20261028_214637_b9a2294e"
    original = check_metric_rows(campaign / "evaluation", 60, research=True)
    ce = check_metric_rows(ROOT / "results/advisor_runs/ce_0c139aa5b451/evaluation", 3)
    sampling = check_metric_rows(ROOT / "results/advisor_runs/sampling_1827c85af012/evaluation", 12)
    audited = pd.read_csv(ROOT / "results/advisor_step1/test_metrics_verified.csv")
    keys = ["suite", "comparison_group", "method", "seed"]
    pd.testing.assert_frame_equal(
        original[keys + ["ap", "roc_auc", "threshold"]].sort_values(keys).reset_index(drop=True),
        audited[keys + ["ap", "roc_auc", "threshold"]].sort_values(keys).reset_index(drop=True),
        check_dtype=False, atol=1e-10, rtol=0,
    )
    expected = pd.concat([audited, ce, sampling], ignore_index=True)
    meta = expected[expected.method.isin(["lre", "mwn"])].copy()
    meta["comparison_group"] = "meta_matched_batch256"
    expected = pd.concat([expected, meta], ignore_index=True)
    actual = pd.read_csv(ROOT / "results/advisor_combined/test_metrics_all.csv")
    require(len(actual) == 81, "Expected 75 unique fits plus 6 reused meta comparison rows")
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
                                  actual.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, atol=1e-10, rtol=0)
    sys.path.insert(0, str(ROOT / "supplement"))
    import reporting
    for folder, frame in [(ROOT / "results/advisor_step1", audited),
                          (ROOT / "results/advisor_combined", actual)]:
        numeric = pd.read_csv(folder / "all_metrics_mean_sd.csv")
        grouping = ["suite", "comparison_group", "method"]
        require(not frame.duplicated(grouping + ["seed"]).any(), "Duplicate summary inputs")
        for metric in ["ap", "roc_auc", "recall_at_fpr5_curve", "fpr_frozen_threshold"]:
            grouped = frame.groupby(grouping)[metric].agg(["mean", "std"]).reset_index()
            merged = numeric.merge(grouped, on=grouping, validate="one_to_one")
            for stat in ["mean", "std"]:
                require(np.allclose(merged[metric + "_" + stat], merged[stat], atol=1e-10, rtol=0),
                        f"Summary mismatch: {folder.name}/{metric}/{stat}")

    original_fits = 0
    max_log_loss_difference = 0.0
    for run in sorted((ROOT / "results/advisor_runs").glob("original_checks_*")):
        if not run.is_dir():
            continue
        table = pd.read_csv(run / "original_dataset_metrics.csv")
        require(len(table) == 15, f"Expected 15 small original-data fits: {run.name}")
        for row in table.to_dict("records"):
            folder = run / row["dataset"] / f"{row['method']}_s{int(row['seed'])}"
            record = read_json(folder / "COMPLETE.json")
            with np.load(folder / "test_predictions.npz", allow_pickle=False) as data:
                # The source stores the original labels and probability matrix.
                y, probability = data["y"], data["probability"]
            values = {"accuracy": accuracy_score(y, probability.argmax(axis=1)),
                      "balanced_accuracy": balanced_accuracy_score(y, probability.argmax(axis=1)),
                      "log_loss": log_loss(y, probability, labels=list(range(probability.shape[1])))}
            for key, value in values.items():
                # CPU evidence used sklearn 1.8; current CUDA environment uses
                # 1.9. Float32 log-loss reduction differs by < 8e-7 on these files.
                tolerance = 1e-6 if key == "log_loss" else 1e-12
                if key == "log_loss":
                    max_log_loss_difference = max(max_log_loss_difference, abs(value - row[key]))
                require(np.isclose(value, row[key], atol=tolerance, rtol=0), f"Original-data metric mismatch: {folder}/{key}")
                require(np.isclose(value, record[key], atol=tolerance, rtol=0), f"Fit record mismatch: {folder}/{key}")
            original_fits += 1
    return {"baf_scores_recomputed": 75, "combined_rows": 81,
            "combined_groups": 27, "small_original_dataset_fits_recomputed": original_fits,
            "max_original_log_loss_difference": max_log_loss_difference,
            "original_log_loss_absolute_tolerance": 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report outside the repository; default is read-only")
    args = parser.parse_args()
    official = (ROOT / "docs/repository/GITHUB_FILES.txt").read_text(encoding="utf-8").splitlines()
    require(len(official) == len(set(official)), "Duplicate official paths")
    if args.output:
        require(not args.output.resolve().is_relative_to(ROOT),
                "Write the report outside this repository; rewriting an official file would invalidate its inventory hash")
    inventory_result = check_inventory(ROOT, official)
    code_count = notebook_count = 0
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    transformer = TransformerManager()
    for relative in official:
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT), f"Path outside repository: {relative}")
        require(path.is_file(), f"Missing official file: {relative}")
        if path.suffix == ".py" and not relative.startswith("docs/advisor/upstream/"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            code_count += 1
        elif path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".ipynb":
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    ast.parse(transformer.transform_cell(cell.source), filename=relative)
            notebook_count += 1
    scope = read_json(ROOT / "docs/repository/EVIDENCE_SCOPE.json")
    for relative, expected in scope["retained_sha256"].items():
        require(digest(ROOT / relative) == expected, f"Retained evidence changed: {relative}")
    from check_notebook20_run_selection import check_run_selection
    from check_inventory_integrity import run_checks
    result = {"official_files_present": len(official), **inventory_result,
              "python_files_parsed": code_count,
              "notebooks_validated": notebook_count, "evidence_hashes_checked": len(scope["retained_sha256"]),
              **check_historical_survey(), "notebook20_run_selection": check_run_selection(ROOT),
              "inventory_validator_checks": run_checks(),
              **check_results()}
    if args.imports:
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "supplement")]
        import baf
        import gru
        import reporting
        import training
        for module in [baf, gru, reporting, training]:
            require(Path(module.__file__).resolve().is_relative_to(ROOT), f"Foreign import: {module.__name__}")
        result["training_imports"] = "passed"
    result["status"] = "passed"
    result["scope"] = "All official inventory sizes/SHA-256 (self-inventory digest excepted), historical CSV, notebook20 selector, saved scores/summaries/schemas/imports; no training, downloads, bootstrap rerun, or omitted checkpoints"
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
