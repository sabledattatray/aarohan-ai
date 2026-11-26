"""Train a separate BPE tokenizer from cleaned WikiText-2 training text only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aarohan.tokenizer import train_bpe


def main() -> None:
    prepared_dir = ROOT / "data" / "processed" / "wikitext2"
    train_path = prepared_dir / "train.txt"
    validation_path = prepared_dir / "validation.txt"
    prep_manifest_path = prepared_dir / "preparation-manifest.json"
    output_dir = ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe"

    for path in (train_path, validation_path, prep_manifest_path):
        if not path.is_file():
            raise SystemExit(f"Missing prepared WikiText-2 file: {path}; run prepare_wikitext2.py first")
    prep = json.loads(prep_manifest_path.read_text(encoding="utf-8"))
    review_path = ROOT / "data" / "wikitext2-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    license_review = review.get("license_review", {})
    if not (
        license_review.get("status") == "reviewed_approved"
        and review.get("training_allowed") is True
        and license_review.get("reviewer", {}).get("role") == "trusted_adult"
    ):
        raise SystemExit("WikiText-2 review is not approved for local training")

    requested_vocab = 8192
    metadata = train_bpe(train_path, output_dir, vocab_size=requested_vocab)
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    summary = {
        **metadata,
        "dataset_id": prep.get("dataset_id"),
        "dataset_revision": prep.get("revision"),
        "preparation_manifest_sha256": hashlib.sha256(prep_manifest_path.read_bytes()).hexdigest(),
        "training_file_sha256": train_hash,
        "validation_file_sha256": validation_hash,
        "validation_used_for_tokenizer_training": False,
        "test_split_read_or_used": False,
        "requested_vocab_size": requested_vocab,
        "note": "Tokenizer fit used only the cleaned training split. Keep this artifact separate from the toy-corpus tokenizer and preserve dataset attribution/share-alike notices.",
    }
    (output_dir / "wikitext2-training-record.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
