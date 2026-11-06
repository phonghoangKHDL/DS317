"""Exercise notebook20's actual run selector using small temporary fixtures only.

No notebook setup/worker cells, training, model imports, or real run writes.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def check_run_selection(root):
    notebook = json.loads((Path(root) / "notebooks/20_Original_dataset_weighting_checks.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    source = next(s for s in cells if "def select_original_check_run(" in s)
    require_no_hardcoded = "original_checks_582e4a8ada1a" not in source
    if not require_no_hardcoded:
        raise AssertionError("Notebook20 still contains a hardcoded resume run")
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "select_original_check_run")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json,
                 "reporting": SimpleNamespace(read_json=lambda p: json.loads(Path(p).read_text(encoding="utf-8")))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook20_selector", "exec"), namespace)
    select = namespace["select_original_check_run"]
    identity = {"version": "fixture", "tag": "original_checks", "config": {"epochs": 10},
                "source": {"src/baf.py": "fixture_hash"},
                "environment": {"python": "fixture_python", "executable": "fixture_interpreter"}}
    passed = []

    def expected_path(folder, current):
        signature = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:12]
        return folder / "results/advisor_runs" / ("original_checks_" + signature)

    def rejects(label, function):
        try:
            function()
        except (RuntimeError, FileNotFoundError):
            passed.append(label)
        else:
            raise AssertionError("Expected rejection: " + label)

    with TemporaryDirectory(prefix="baf_notebook20_") as name:
        folder = Path(name).resolve()
        expected = expected_path(folder, identity)
        unrelated = folder / "results/advisor_runs/original_checks_unrelated"
        unrelated.mkdir(parents=True)
        unrelated_manifest = unrelated / "manifest.json"
        unrelated_manifest.write_text(json.dumps(identity), encoding="utf-8")
        unrelated_before = unrelated_manifest.read_bytes()
        assert select(folder, identity) == expected and not expected.exists()
        passed.append("default_ignores_unrelated_old_run_and_does_not_write")
        for key in ["config", "source", "environment"]:
            changed = deepcopy(identity)
            changed[key]["fixture_change"] = True
            assert select(folder, changed) == expected_path(folder, changed)
            assert select(folder, changed) != expected
            passed.append("default_changes_path_with_" + key)
        expected.mkdir(parents=True)
        manifest = expected / "manifest.json"
        manifest.write_text(json.dumps(identity), encoding="utf-8")
        manifest_before = manifest.read_bytes()
        assert select(folder, identity) == expected
        assert select(folder, identity, expected) == expected
        passed.extend(["default_accepts_matching_identity", "explicit_accepts_matching_canonical_run"])
        for key in ["config", "source", "environment"]:
            changed = deepcopy(identity)
            changed[key]["fixture_change"] = True
            rejects("explicit_rejects_changed_" + key, lambda: select(folder, changed, expected))
        extra = deepcopy(identity)
        extra["extra_manifest_field"] = True
        manifest.write_text(json.dumps(extra), encoding="utf-8")
        rejects("default_rejects_existing_mismatched_manifest", lambda: select(folder, identity))
        rejects("explicit_rejects_extra_manifest_field", lambda: select(folder, identity, expected))
        manifest.write_bytes(manifest_before)
        rejects("explicit_rejects_missing_manifest", lambda: select(folder, identity, folder / "missing"))
        rejects("explicit_rejects_relocated_run_before_launch", lambda: select(folder, identity, unrelated))
        assert manifest.read_bytes() == manifest_before
        assert unrelated_manifest.read_bytes() == unrelated_before
        passed.append("existing_manifests_unchanged")
    return {"passed": len(passed), "checks": passed,
            "scope": "Extracted selector only; temporary fixtures; no training or real run modifications"}


if __name__ == "__main__":
    print(json.dumps(check_run_selection(Path(__file__).resolve().parents[1]), indent=2))
