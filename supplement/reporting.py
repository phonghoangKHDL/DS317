"""Audit saved scores, aggregate BAF metrics, and record executable proof notebooks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

VERSION = "advisor-20261030-v1"
SEEDS = (42, 317, 2026)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def write_csv(frame, path):
    """Publish a complete CSV atomically, including on repeated epoch writes."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    data = frame.to_csv(index=False).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "lightgbm", "nbformat", "ipython"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def setup(root):
    root = Path(root).resolve()
    cache = Path(r"D:\UIT\.cache\baf-research") if os.name == "nt" else root / ".cache"
    for key, folder in {"PIP_CACHE_DIR": "pip", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "misc"}.items():
        target = cache / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(root / "src"))
    return root


def metrics(y, score, age, threshold, cap=0.05):
    y, score, age = np.asarray(y), np.asarray(score), np.asarray(age)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError("Invalid score alignment/values")
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    prediction = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {
        "ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_fpr5_curve": float(tpr[fpr <= cap].max()),
        "recall_frozen_threshold": float(tp / (tp + fn)),
        "fpr_frozen_threshold": float(fp / (fp + tn)),
        "precision_frozen_threshold": float(tp / (tp + fp)) if tp + fp else 0.0,
        "alert_rate_frozen_threshold": float(prediction.mean()),
    }
    for name, mask in [("older", age >= 50), ("younger", age < 50)]:
        a, b, c, d = confusion_matrix(y[mask], prediction[mask], labels=[0, 1]).ravel()
        result[name + "_fpr"] = float(b / (a + b)) if a + b else np.nan
        result[name + "_tpr"] = float(d / (c + d)) if c + d else np.nan
        result[name + "_n"] = int(mask.sum())
        result[name + "_negative_n"] = int(a + b)
        result[name + "_positive_n"] = int(c + d)
    f0, f1 = result["younger_fpr"], result["older_fpr"]
    result["fpr_gap_older_minus_younger"] = f1 - f0
    result["fpr_ratio_min_max"] = min(f0, f1) / max(f0, f1) if max(f0, f1) else 1.0
    return result


def score_audit(source, output):
    """Recompute all 60 selected-model metrics; directories and proof ZIPs supported."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        if archive:
            candidates = [n for n in archive.namelist() if n.endswith("/test_metrics.csv")]
            if len(candidates) != 1:
                raise ValueError("Proof ZIP must contain exactly one test_metrics.csv")
            prefix = candidates[0].rsplit("/", 1)[0] + "/"
            load = lambda name: archive.read(prefix + name)
        else:
            load = lambda name: (source / name).read_bytes()
        original = pd.read_csv(io.BytesIO(load("test_metrics.csv")))
        with np.load(io.BytesIO(load("test_rows.npz")), allow_pickle=False) as data:
            rows = {k: data[k] for k in data.files}
        records, checks = [], []
        names = archive.namelist() if archive else [p.name for p in source.glob("*_score.npy")]
        for row in original.to_dict("records"):
            stem = f"{row['suite']}__{row['method']}_c"
            suffix = f"_s{int(row['seed'])}_score.npy"
            matches = [n for n in names if Path(n).name.startswith(stem) and n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous score: {stem}{suffix}")
            relative = matches[0][len(prefix):] if archive else matches[0]
            raw = load(relative)
            score = np.load(io.BytesIO(raw), allow_pickle=False)
            calculated = metrics(rows["y"], score, rows["customer_age"], float(row["threshold"]))
            for name, value in calculated.items():
                if name in row and not np.isclose(value, float(row[name]), rtol=0, atol=1e-10, equal_nan=True):
                    raise AssertionError(f"Metric mismatch: {row['method']} seed={row['seed']} {name}")
            records.append({**row, **calculated})
            checks.append({"method": row["method"], "seed": int(row["seed"]), "file": relative,
                           "score_sha256": hashlib.sha256(raw).hexdigest(), "verified": True})
        frame = pd.DataFrame(records)
        frame.to_csv(output / "test_metrics_verified.csv", index=False)
        pd.DataFrame(checks).to_csv(output / "score_audit.csv", index=False)
        save_json(output / "provenance.json", {
            "version": VERSION, "source": str(source), "source_sha256": digest(source) if archive else None,
            "test_rows_sha256": hashlib.sha256(load("test_rows.npz")).hexdigest(),
            "n_test": len(rows["y"]), "fraud": int(rows["y"].sum()), "verified_scores": len(checks),
            "metrics_sha256": digest(output / "test_metrics_verified.csv"),
            "scope": "Recomputed from saved predictions; no new model fits or new test data",
        })
        print("SCORE_AUDIT_PASS", len(checks), "models", len(rows["y"]), "test rows")
        return frame
    finally:
        if archive:
            archive.close()


def summarize(frame, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = ["suite", "comparison_group", "method"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate method/seed records")
    columns = [
        "ap", "roc_auc", "recall_at_fpr5_curve", "recall_frozen_threshold",
        "fpr_frozen_threshold", "precision_frozen_threshold", "alert_rate_frozen_threshold",
        "older_fpr", "younger_fpr", "older_tpr", "younger_tpr",
        "fpr_gap_older_minus_younger", "fpr_ratio_min_max", "fit_seconds",
    ]
    columns = [c for c in columns if c in frame]
    numeric = frame.groupby(keys, sort=True)[columns].agg(["mean", "std"])
    numeric.columns = ["_".join(pair) for pair in numeric.columns]
    numeric = numeric.reset_index()
    counts = frame.groupby(keys).size().rename("n_seeds").reset_index()
    numeric = numeric.merge(counts, on=keys)
    numeric.to_csv(output / "all_metrics_mean_sd.csv", index=False)
    display = numeric[keys + ["n_seeds"]].copy()
    for name in columns:
        multiplier = 100 if name not in ["ap", "roc_auc", "fit_seconds", "fpr_ratio_min_max"] else 1
        display[name] = [f"{a * multiplier:.5f} ± {b * multiplier:.5f}" for a, b in
                         zip(numeric[name + "_mean"], numeric[name + "_std"])]
    display.to_csv(output / "research_table_display.csv", index=False, encoding="utf-8-sig")
    group_columns = [c for c in frame if c.endswith(("_n", "_negative_n", "_positive_n"))]
    if group_columns:
        frame[keys + ["seed"] + group_columns].to_csv(output / "group_support.csv", index=False)
    notes = (
        "# Bảng khảo sát thực nghiệm trên BAF\n\n"
        "Các giá trị là trung bình ± sample SD của các seed được ghi ở bảng chi tiết. "
        "AP/ROC-AUC ở thang 0–1; recall, precision, FPR, alert rate ở %; gap FPR ở điểm phần trăm; "
        "ratio FPR ở thang 0–1; thời gian ở giây.\n\n"
        "TPR@FPR≤5% trên ROC test là độ đo benchmark. Recall/FPR frozen dùng ngưỡng đã khóa "
        "trên validation; FPR test có thể khác 5%. Chỉ số nhóm tuổi frozen và các ratio/gap được tính "
        "riêng từng seed rồi lấy trung bình. SD seed không phải CI lấy mẫu test.\n\n"
        "So sánh weighting trong cùng backbone/feature pipeline. CE batch2048 là baseline loss; "
        "đối chiếu meta với CE batch256 sau bổ sung. LSR dùng chung auxiliary seed42. "
        "Các run này không phải tái lập nguyên thiết lập paper, không xác lập SOTA toàn cục.\n\n"
    )
    # Use plain Markdown instead of a dependency on tabulate.
    for group, part in display.groupby("comparison_group", sort=False):
        wanted = ["method", "ap", "recall_at_fpr5_curve", "fpr_frozen_threshold", "older_fpr", "younger_fpr", "fpr_gap_older_minus_younger"]
        notes += f"## {group}\n\n| " + " | ".join(wanted) + " |\n| " + " | ".join(["---"] * len(wanted)) + " |\n"
        for row in part.to_dict("records"):
            notes += "| " + " | ".join(str(row[c]) for c in wanted) + " |\n"
        notes += "\n"
    (output / "RESEARCH_TABLE.md").write_text(notes, encoding="utf-8")
    print("TABLE_READY", len(numeric), "methods", len(frame), "seed records")
    return numeric


def locate_evaluation(root, selected=None):
    if selected:
        return Path(selected).resolve()
    paths = list((Path(root) / "results/research_campaigns").glob("*/evaluation/test_metrics.csv"))
    if len(paths) == 1:
        return paths[0].parent
    if len(paths) > 1:
        raise ValueError("Several evaluations exist. Set EVALUATION_INPUT explicitly.")
    return None


def step1(root, source=None):
    root = Path(root)
    output = root / "results/advisor_step1"
    source = locate_evaluation(root, source)
    if source:
        frame = score_audit(source, output)
    else:
        provenance = read_json(output / "provenance.json")
        if digest(output / "test_metrics_verified.csv") != provenance["metrics_sha256"]:
            raise ValueError("Included audited metric file changed")
        frame = pd.read_csv(output / "test_metrics_verified.csv")
        print("REAGGREGATE_VERIFIED_METRICS; original score audit is in score_audit.csv")
    return summarize(frame, output)


def launch(root, tag, config, cells):
    """Execute real cells in a fresh process; save each output before exporting ZIP."""
    import nbformat
    root = Path(root).resolve()
    for module in ["nbformat", "IPython"]:
        __import__(module)
    source = {str(p.relative_to(root)): digest(p) for p in
              [root / "src/baf.py", root / "src/gru.py", root / "supplement/reporting.py", root / "supplement/training.py"]}
    identity = {"version": VERSION, "tag": tag, "config": config, "source": source, "environment": environment()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    run = root / "results/advisor_runs" / f"{tag}_{signature}"
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / "manifest.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Run identity changed")
    save_json(manifest, identity)
    attempts = run / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    attempt.mkdir()
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        f"# {tag}: actual execution\n\nConfiguration and source hashes: ../../manifest.json"
    )] + [nbformat.v4.new_code_cell(c) for c in cells])
    path = attempt / "executed.ipynb"
    nbformat.write(nb, path)
    command = "import sys; sys.path.insert(0, sys.argv[1]); import baf; baf.execute_weighting_recipe(sys.argv[2])"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BAF_RUN_DIR"] = str(run)
    env["BAF_PROJECT_ROOT"] = str(root)
    print("RUN_DIR", run, flush=True)
    status = {"state": "running", "attempt": str(attempt)}
    try:
        with (attempt / "execution.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-c", command, str(root / "src"), str(path)],
                                       cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError("Execution failed; completed fits retained for resume.")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        status["state"] = "completed"
    except BaseException as error:
        status.update(state="failed_or_interrupted", error=repr(error))
        raise
    finally:
        save_json(attempt / "status.json", status)
    evidence = run / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in source:
        target = evidence / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    result = export_proof(run)
    print("PROOF_ZIP", result)
    return run


def export_proof(run):
    run = Path(run)
    # Include raw scores, histories, executed notebooks and source. Exclude model/data payloads.
    files = [p for p in run.rglob("*") if p.is_file() and p.suffix not in [".pt", ".pkl", ".joblib"]
             and p.name not in ["model.txt", "sha256.json", "PROOF_SCOPE.json", "sampled_indices.npy"] and "__pycache__" not in p.parts]
    omitted = [p for p in run.rglob("*") if p.is_file() and p not in files
               and p.name not in ["sha256.json", "PROOF_SCOPE.json"] and "__pycache__" not in p.parts]
    save_json(run / "PROOF_SCOPE.json", {
        "kind": "compact review; model checkpoints and sampled indices stay in the local run",
        "omitted": {p.relative_to(run).as_posix(): digest(p) for p in omitted},
    })
    files.append(run / "PROOF_SCOPE.json")
    save_json(run / "sha256.json", {p.relative_to(run).as_posix(): digest(p) for p in files})
    archive = run.with_name(run.name + "_review.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files + [run / "sha256.json"]:
            z.write(path, str(Path(run.name) / path.relative_to(run)))
    return archive
