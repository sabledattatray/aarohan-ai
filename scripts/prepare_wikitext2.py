"""Prepare cleaned WikiText-2 train/validation text files; never read test rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "wikitext2-raw-v1"
OUTPUT_DIR = ROOT / "data" / "processed" / "wikitext2"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    manifest_path = source_dir / "download-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_path = ROOT / "data" / "wikitext2-review.json"
    if not review_path.is_file():
        raise ValueError("Dataset review file is missing")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    # The downloader's review manifest records the decision as
    # license_review.status / license_review.training_allowed.
    decision = review.get("license_review", {})
    approved = (
        decision.get("status") == "reviewed_approved"
        and review.get("training_allowed") is True
        and decision.get("reviewer", {}).get("role") == "trusted_adult"
    )
    if not approved:
        raise ValueError("Dataset review is not approved for local training")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install requirements-data.txt") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation"):
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest has no {split} split entry")
        source_path = source_dir / entry["file"]
        if _hash_file(source_path) != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {split} source file")

        texts = pq.read_table(source_path, columns=["text"])["text"].to_pylist()
        seen: set[str] = set()
        cleaned: list[str] = []
        blanks = duplicates = 0
        for value in texts:
            if not isinstance(value, str) or not value.strip():
                blanks += 1
                continue
            text = value.strip()
            if text in seen:
                duplicates += 1
                continue
            seen.add(text)
            cleaned.append(text)
        cross_split_duplicates = 0
        if split == "validation":
            train_text = (output_dir / "train.txt").read_text(encoding="utf-8")
            train_paragraphs = {part.strip() for part in train_text.split("\n\n") if part.strip()}
            filtered = []
            for paragraph in cleaned:
                if paragraph in train_paragraphs:
                    cross_split_duplicates += 1
                else:
                    filtered.append(paragraph)
            cleaned = filtered
        output_path = output_dir / f"{split}.txt"
        output_path.write_text("\n\n".join(cleaned) + "\n", encoding="utf-8")
        summary[split] = {
            "source_rows": len(texts),
            "blank_rows_removed": blanks,
            "duplicate_rows_removed": duplicates,
            "cross_split_duplicates_removed": cross_split_duplicates,
            "paragraphs_written": len(cleaned),
            "output_file": str(output_path),
            "output_sha256": _hash_file(output_path),
        }

    result = {
        "dataset_id": manifest.get("dataset_id"),
        "revision": manifest.get("revision"),
        "splits": summary,
        "test_split_read_or_written": False,
        "test_split_used_for_training": False,
        "source_manifest_sha256": _hash_file(manifest_path),
        "note": "Blank and exact duplicate rows were removed within each split. This does not establish that text has no personal information; human review is still needed before training.",
    }
    (output_dir / "preparation-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep a trainer-facing manifest distinct from the original-text manifest:
    # WikiText is third-party text and must never pass the original-only gate.
    review = json.loads((ROOT / "data" / "wikitext2-review.json").read_text(encoding="utf-8"))
    training_manifest = {
        "dataset_name": "WikiText-2 raw v1",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_revision": manifest.get("revision"),
        "source_url": manifest.get("source_url"),
        "license_or_permission": (
            "Dataset card: CC BY-SA 4.0; repository tags: CC-BY-SA-3.0 and GFDL. "
            "See data/wikitext2-review.json and retain applicable attribution and share-alike notices."
        ),
        "review_manifest_sha256": hashlib.sha256(
            (ROOT / "data" / "wikitext2-review.json").read_bytes()
        ).hexdigest(),
        "status": "approved_for_local_experiment" if review.get("training_allowed") is True else "pending_review",
        "privacy_reviewed": review.get("privacy_reviewed") is True,
        "third_party_text_included": True,
        "train_file": "train.txt",
        "validation_file": "validation.txt",
        "train_sha256": summary["train"]["output_sha256"],
        "validation_sha256": summary["validation"]["output_sha256"],
        "source_manifest_sha256": result["source_manifest_sha256"],
        "test_split_read_or_written": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    try:
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"prepared": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
