"""Download pinned WikiText-2 raw Parquet splits after a human license review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"
SUBSET = "wikitext-2-raw-v1"
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_human_review(review_path: Path) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    license_review = review.get("license_review", {})
    if license_review.get("status") != "reviewed_approved":
        raise PermissionError("Dataset license review is not approved; no files were downloaded.")
    if review.get("training_allowed") is not True:
        raise PermissionError("Set training_allowed=true only after the reviewer approves this use.")
    reviewer = license_review.get("reviewer", {})
    if reviewer.get("role") not in {"parent_guardian", "teacher", "rights_reviewer", "trusted_adult"}:
        raise PermissionError("Review must identify a parent/guardian, teacher, rights reviewer, or trusted adult.")
    if not reviewer.get("name") or not reviewer.get("review_date"):
        raise PermissionError("Reviewer name and review_date must be recorded before download.")
    if not license_review.get("decision_note"):
        raise PermissionError("Record the review decision and applicable attribution/share-alike terms.")
    return review


def download(review_path: Path, output_dir: Path) -> dict:
    review_path, output_dir = review_path.resolve(), output_dir.resolve()
    review = check_human_review(review_path)
    try:
        from huggingface_hub import hf_hub_download
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError("Install optional intake dependencies with: python -m pip install -r requirements-data.txt") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        filename = f"{SUBSET}/{split}-00000-of-00001.parquet"
        cached_path = Path(hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            revision=DATASET_REVISION,
        ))
        destination = output_dir / f"{split}.parquet"
        destination.write_bytes(cached_path.read_bytes())
        table = parquet.read_table(destination, columns=["text"])
        split_metadata[split] = {
            "file": destination.name,
            "rows": table.num_rows,
            "sha256": sha256_file(destination),
        }

    record = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "subset": SUBSET,
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "review_manifest_sha256": sha256_file(review_path),
        "reviewer": review["license_review"]["reviewer"],
        "license_review": {
            "dataset_card_section": review["license_review"]["dataset_card_section"],
            "repository_license_tags": review["license_review"]["repository_license_tags"],
            "decision_note": review["license_review"]["decision_note"],
        },
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "splits": split_metadata,
        "test_split_used_for_training": False,
        "processing": "Original Parquet split files copied unchanged; text extraction is a later step.",
    }
    (output_dir / "download-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "data" / "wikitext2-review.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "wikitext2-raw-v1")
    args = parser.parse_args()
    try:
        print(json.dumps(download(args.review, args.output_dir), indent=2))
    except (PermissionError, ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"downloaded": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

