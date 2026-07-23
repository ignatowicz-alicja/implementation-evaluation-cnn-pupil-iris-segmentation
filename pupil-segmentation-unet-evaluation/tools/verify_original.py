from __future__ import annotations

import csv
import hashlib
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
manifest = repository_root / "provenance" / "original_files_sha256.csv"
author_dir = repository_root / "original_author_code"

errors: list[str] = []
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        path = author_dir / row["relative_path"]
        if not path.is_file():
            errors.append(f"MISSING: {row['relative_path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"CHANGED: {row['relative_path']}")

if errors:
    raise SystemExit("Original-code verification failed:\n" + "\n".join(errors))

print("[OK] All files in original_author_code match the recorded SHA-256 manifest.")
