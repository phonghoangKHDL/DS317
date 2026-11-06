"""Verify the published evidence without fitting models or downloading data.

Run from any directory: python -B scripts/verify_repository.py
Optional: --imports checks the training imports in the selected interpreter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
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


def git_file_manifest(root):
    """Read exact tracked bytes from the Git index without network access."""
    root = Path(root).resolve()
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"]
    ).decode("utf-8").strip()
    require(Path(top).resolve() == root, "Run in this repository or pass an external --manifest for an index export")
    entries = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"]
    )
    paths = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split()
        require(stage == "0" and mode in {"100644", "100755"},
                "Unresolved conflict, submodule or unsupported file mode in index")
        relative = name.decode("utf-8")
        require(relative not in paths, f"Duplicate indexed path: {relative}")
        paths[relative] = oid
    require(bool(paths), "Empty Git index")
    objects = sorted(set(paths.values()))
    raw = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=("\n".join(objects) + "\n").encode("ascii"),
    )
    position = 0
    by_oid = {}
    for expected_oid in objects:
        end = raw.index(b"\n", position)
        oid, kind, size = raw[position:end].decode("ascii").split()
        require(oid == expected_oid and kind == "blob", "Unexpected Git batch object")
        size = int(size)
        data = raw[end + 1:end + 1 + size]
        require(len(data) == size and raw[end + 1 + size:end + 2 + size] == b"\n", "Truncated Git blob")
        by_oid[oid] = {"size_bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
        position = end + 2 + size
    require(position == len(raw), "Unexpected Git batch trailer")
    return {"kind": "GIT_INDEX_FILE_MANIFEST",
            "files": {relative: by_oid[oid] for relative, oid in sorted(paths.items())}}


def check_manifest(root, manifest):
    """Every official byte is checked; no public inventory/self-hash exception."""
    root = Path(root).resolve()
    require(manifest.get("kind") == "GIT_INDEX_FILE_MANIFEST", "Unsupported manifest kind")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Empty/invalid file manifest")
    for relative, record in files.items():
        path = root / relative
        require(path.resolve().is_relative_to(root), f"Path outside repository: {relative}")
        require(path.is_file(), f"Missing tracked file: {relative}")
        expected_size = record["size_bytes"]
        require(type(expected_size) is int and expected_size >= 0, f"Invalid byte size: {relative}")
        require(path.stat().st_size == expected_size, f"Git size mismatch: {relative}")
        expected_hash = record["sha256"]
        require(isinstance(expected_hash, str) and len(expected_hash) == 64
                and all(c in "0123456789abcdef" for c in expected_hash), f"Invalid SHA-256: {relative}")
        require(digest(path) == expected_hash, f"Git SHA-256 mismatch: {relative}")
    return {"git_file_sizes_checked": len(files), "git_file_sha256_checked": len(files)}


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
    parser.add_argument("--manifest", type=Path, help="External Git-index manifest when verifying an export without .git")
    parser.add_argument("--write-manifest", type=Path, help="Export Git-index hashes outside the repository, then exit")
    args = parser.parse_args()
    if args.output:
        require(not args.output.resolve().is_relative_to(ROOT), "Write the report outside this repository")
    if args.write_manifest:
        require(args.manifest is None, "Do not combine --manifest and --write-manifest")
        require(not args.write_manifest.resolve().is_relative_to(ROOT), "Write the manifest outside this repository")
        manifest = git_file_manifest(ROOT)
        args.write_manifest.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        print(f"Exported Git-index manifest: {len(manifest['files'])} files")
        return
    manifest = read_json(args.manifest) if args.manifest else git_file_manifest(ROOT)
    manifest_result = check_manifest(ROOT, manifest)
    official = sorted(manifest["files"])
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
    from check_notebook20_run_selection import check_run_selection
    result = {"official_files_present": len(official), **manifest_result,
              "python_files_parsed": code_count, "notebooks_validated": notebook_count,
              **check_historical_survey(), "notebook20_run_selection": check_run_selection(ROOT),
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
    result["scope"] = "All indexed file sizes/SHA-256, historical CSV, notebook20 selector, saved scores/summaries/schemas/imports; no training, downloads, bootstrap rerun, or omitted checkpoints"
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
