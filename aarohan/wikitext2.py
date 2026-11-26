"""Dataset-specific provenance checks for the third-party WikiText-2 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASET = "Salesforce/wikitext"
EXPECTED_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_wikitext2(directory: Path) -> dict[str, object]:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_path = PROJECT_ROOT / "data" / "wikitext2-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    license_review = review.get("license_review", {})
    errors: list[str] = []

    if manifest.get("dataset_id") != EXPECTED_DATASET:
        errors.append("unexpected WikiText dataset id")
    if manifest.get("dataset_revision") != EXPECTED_REVISION:
        errors.append("dataset revision differs from the reviewed pinned revision")
    if manifest.get("third_party_text_included") is not True:
        errors.append("WikiText-2 must be identified as third-party text")
    if manifest.get("status") != "approved_for_local_experiment":
        errors.append("license review is not approved for local experiments")
    if license_review.get("status") != "reviewed_approved" or review.get("training_allowed") is not True:
        errors.append("trusted-adult license approval is missing")
    if license_review.get("reviewer", {}).get("role") != "trusted_adult":
        errors.append("trusted-adult reviewer role is missing")
    if review.get("privacy_reviewed") is not True or manifest.get("privacy_reviewed") is not True:
        errors.append("privacy_reviewed must be true after trusted-adult review of the corpus for private personal details")
    if manifest.get("review_manifest_sha256") != _sha256(review_path):
        errors.append("review file changed after the prepared corpus manifest was written; rerun prepare_wikitext2.py")

    train_path = directory / str(manifest.get("train_file", "train.txt"))
    validation_path = directory / str(manifest.get("validation_file", "validation.txt"))
    if not train_path.is_file() or not validation_path.is_file():
        errors.append("prepared training or validation text file is missing")
        train, validation = [], []
    else:
        train = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validation = [line.strip() for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if _sha256(train_path) != manifest.get("train_sha256"):
            errors.append("training text hash does not match its manifest")
        if _sha256(validation_path) != manifest.get("validation_sha256"):
            errors.append("validation text hash does not match its manifest")
        overlap = set(train) & set(validation)
        if overlap:
            errors.append(f"train and validation contain {len(overlap)} duplicate paragraph(s)")
        if not train or not validation:
            errors.append("training and validation splits must both contain text")
    if manifest.get("test_split_read_or_written") is not False:
        errors.append("test split must remain untouched")

    return {
        "dataset_name": manifest.get("dataset_name"),
        "train_paragraphs": len(train),
        "validation_paragraphs": len(validation),
        "ready": not errors,
        "errors": errors,
    }
