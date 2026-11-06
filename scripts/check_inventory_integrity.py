"""Small positive/negative checks for the published inventory validator."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from verify_repository import check_inventory


def run_checks():
    checks = []
    with TemporaryDirectory(prefix="baf_inventory_") as name:
        root = Path(name)
        payload = root / "payload.txt"
        payload.write_bytes(b"exact bytes\r\n")
        inventory = root / "FILE_INVENTORY.csv"
        official = ["FILE_INVENTORY.csv", "payload.txt"]
        rows = [
            {"path": "FILE_INVENTORY.csv", "size_bytes": 0, "sha256": "", "category": "official", "include_in_github": "True"},
            {"path": "payload.txt", "size_bytes": payload.stat().st_size,
             "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(), "category": "official", "include_in_github": "True"},
        ]

        def publish():
            for _ in range(10):
                with inventory.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                size = inventory.stat().st_size
                if rows[0]["size_bytes"] == size:
                    return
                rows[0]["size_bytes"] = size
            raise AssertionError("Inventory size did not stabilize")

        def rejects(label, fragment):
            try:
                check_inventory(root, official)
            except ValueError as error:
                if fragment not in str(error):
                    raise AssertionError(f"Unexpected rejection for {label}: {error}") from error
                checks.append(label)
            else:
                raise AssertionError("Expected inventory rejection: " + label)

        publish()
        result = check_inventory(root, official)
        assert result["inventory_file_sizes_checked"] == 2
        assert result["inventory_file_sha256_checked"] == 1
        checks.append("valid_inventory_and_explicit_self_hash_exception")
        original = payload.read_bytes()
        payload.write_bytes(original.replace(b"\r\n", b"\n"))
        rejects("rejects_CRLF_to_LF_size_change", "size mismatch")
        payload.write_bytes(original.replace(b"exact", b"other"))
        rejects("rejects_same_size_content_change", "SHA-256 mismatch")
        payload.write_bytes(original)
        rows[1]["sha256"] = ""
        publish()
        rejects("rejects_missing_nonself_hash", "invalid inventory SHA-256")
        rows[1]["sha256"] = hashlib.sha256(original).hexdigest()
        rows[0]["sha256"] = "a" * 64
        publish()
        rejects("rejects_fabricated_self_hash", "self SHA-256 must be empty")
        rows[0]["sha256"] = ""
        rows[1]["include_in_github"] = "False"
        publish()
        rejects("rejects_official_list_disagreement", "official paths differ")
        rows[1]["include_in_github"] = "True"
        rows.append(dict(rows[1]))
        publish()
        rejects("rejects_duplicate_inventory_row", "Duplicate inventory path")
        rows.pop()
        publish()
        inventory.write_bytes(inventory.read_bytes() + b"\r\n")
        rejects("checks_inventory_own_byte_size", "size mismatch: FILE_INVENTORY.csv")
    return {"passed": len(checks), "checks": checks, "scope": "Temporary byte fixtures only; no experiment execution"}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
