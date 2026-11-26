"""Validate local corpus provenance and train/validation separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "data" / "corpora" / "original"


def read_paragraphs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def validate(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {manifest_path} at line {error.lineno}, column {error.colno}: {error.msg}. "
            "Check commas, quotation marks, and braces."
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    errors: list[str] = []
    for field in ("author", "created_date", "license_or_permission"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip() or value.lower().startswith("pending"):
            errors.append(f"manifest field '{field}' must be completed")
    if manifest.get("status") != "approved_for_local_experiment":
        errors.append("set manifest status to 'approved_for_local_experiment' after reviewing the text and permission")
    if manifest.get("privacy_reviewed") is not True:
        errors.append("privacy_reviewed must be true after confirming the corpus contains no private personal details")
    if manifest.get("third_party_text_included") is not False:
        errors.append("the starter corpus must contain only original text; third_party_text_included must be false")

    train_path = directory / str(manifest.get("train_file", "corpus.txt"))
    validation_path = directory / str(manifest.get("validation_file", "validation.txt"))
    train = read_paragraphs(train_path) if train_path.exists() else []
    validation = read_paragraphs(validation_path) if validation_path.exists() else []
    if not train:
        errors.append("training corpus is empty; add original paragraphs to corpus.txt")
    if not validation:
        errors.append("validation corpus is empty; add separate original paragraphs to validation.txt")
    overlap = set(train) & set(validation)
    if overlap:
        errors.append(f"train/validation contain {len(overlap)} identical paragraph(s); keep the split separate")

    result: dict[str, object] = {
        "dataset_name": manifest.get("dataset_name"),
        "train_paragraphs": len(train),
        "validation_paragraphs": len(validation),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest() if train_path.exists() else None,
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest() if validation_path.exists() else None,
        "ready": not errors,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    try:
        result = validate(args.directory)
    except (ValueError, OSError) as error:
        print(json.dumps({"ready": False, "errors": [str(error)]}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

