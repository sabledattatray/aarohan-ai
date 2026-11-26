"""Prepare the held-out WikiText-2 test stream for one-time final scoring."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from aarohan.tokenizer import load_tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw_dir = ROOT / "data" / "raw" / "wikitext2-raw-v1"
    prepared_dir = ROOT / "data" / "processed" / "wikitext2"
    tokenizer_dir = ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe"
    output_dir = prepared_dir / "test-evaluation"
    download_manifest_path = raw_dir / "download-manifest.json"
    corpus_manifest_path = prepared_dir / "manifest.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    review_path = ROOT / "data" / "wikitext2-review.json"
    required = (download_manifest_path, corpus_manifest_path, tokenizer_path, review_path,
                prepared_dir / "train.txt", prepared_dir / "validation.txt")
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not (
        review.get("license_review", {}).get("status") == "reviewed_approved"
        and review.get("license_review", {}).get("reviewer", {}).get("role") == "trusted_adult"
        and review.get("training_allowed") is True
        and review.get("privacy_reviewed") is True
    ):
        raise SystemExit("Trusted-adult license and privacy review must be recorded before test preparation")

    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    test_entry = download_manifest.get("splits", {}).get("test", {})
    raw_test_path = raw_dir / test_entry.get("file", "")
    if not raw_test_path.is_file() or sha256(raw_test_path) != test_entry.get("sha256"):
        raise SystemExit("Raw test file is missing or its hash differs from the download manifest")
    if corpus_manifest.get("dataset_id") != download_manifest.get("dataset_id") or corpus_manifest.get("dataset_revision") != download_manifest.get("revision"):
        raise SystemExit("Prepared corpus and held-out test are from different dataset revisions")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required; install requirements-data.txt") from exc

    tokenizer = load_tokenizer(tokenizer_path)
    train_paragraphs = {
        item.strip() for item in (prepared_dir / "train.txt").read_text(encoding="utf-8").split("\n\n") if item.strip()
    }
    validation_paragraphs = {
        item.strip() for item in (prepared_dir / "validation.txt").read_text(encoding="utf-8").split("\n\n") if item.strip()
    }
    seen: set[str] = set()
    cleaned: list[str] = []
    blank_rows = duplicate_rows = train_overlap_rows = validation_overlap_rows = 0
    for value in pq.read_table(raw_test_path, columns=["text"])["text"].to_pylist():
        if not isinstance(value, str) or not value.strip():
            blank_rows += 1
            continue
        paragraph = value.strip()
        if paragraph in seen:
            duplicate_rows += 1
            continue
        seen.add(paragraph)
        if paragraph in train_paragraphs:
            train_overlap_rows += 1
            continue
        if paragraph in validation_paragraphs:
            validation_overlap_rows += 1
            continue
        cleaned.append(paragraph)

    output_dir.mkdir(parents=True, exist_ok=True)
    test_text_path = output_dir / "test.txt"
    test_text_path.write_text("\n\n".join(cleaned) + "\n", encoding="utf-8")
    ids = tokenizer.encode(test_text_path.read_text(encoding="utf-8")).ids
    if len(ids) < 2:
        raise SystemExit("Held-out test split did not yield enough tokens")
    test_tensor_path = output_dir / "test.pt"
    torch.save(torch.tensor(ids, dtype=torch.long), test_tensor_path)
    metadata = {
        "dataset_id": download_manifest.get("dataset_id"),
        "dataset_revision": download_manifest.get("revision"),
        "split": "test",
        "test_source_sha256": sha256(test_text_path),
        "test_tensor_sha256": sha256(test_tensor_path),
        "raw_test_file_sha256": sha256(raw_test_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "test_paragraphs": len(cleaned),
        "test_tokens": len(ids),
        "blank_rows_removed": blank_rows,
        "duplicate_rows_removed": duplicate_rows,
        "train_overlap_rows_removed": train_overlap_rows,
        "validation_overlap_rows_removed": validation_overlap_rows,
        "test_split_used_for_training": False,
        "note": "Prepared only after selecting the checkpoint on validation. Use once for final scoring; never train or tune on these tokens.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
