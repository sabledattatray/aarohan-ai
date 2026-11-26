"""Encode cleaned WikiText-2 train/validation splits with their saved tokenizer."""

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
    data_dir = ROOT / "data" / "processed" / "wikitext2"
    artifact_dir = ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe"
    tokenizer_path = artifact_dir / "tokenizer.json"
    training_record_path = artifact_dir / "wikitext2-training-record.json"
    prep_manifest_path = data_dir / "preparation-manifest.json"
    train_path = data_dir / "train.txt"
    validation_path = data_dir / "validation.txt"
    for path in (tokenizer_path, training_record_path, prep_manifest_path, train_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    training_record = json.loads(training_record_path.read_text(encoding="utf-8"))
    corpus_manifest_path = data_dir / "manifest.json"
    if not corpus_manifest_path.is_file():
        raise SystemExit(f"Missing dataset manifest: {corpus_manifest_path}; rerun prepare_wikitext2.py")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if training_record.get("validation_used_for_tokenizer_training") is not False or training_record.get("test_split_read_or_used") is not False:
        raise SystemExit("Tokenizer record has unexpected validation/test usage")
    if training_record.get("training_file_sha256") != sha256(train_path):
        raise SystemExit("Training text hash differs from the tokenizer training record")
    # Validation is never used to fit BPE. It may be cleaned after tokenizer
    # training (for example, to remove overlap with train), so pin its current
    # hash in the tokenized-data metadata instead of requiring the old snapshot.

    tokenizer = load_tokenizer(tokenizer_path)
    output_dir = data_dir / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, source in (("train", train_path), ("validation", validation_path)):
        text = source.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        if len(ids) < 2:
            raise SystemExit(f"Not enough tokens in {split} split")
        tensor = torch.tensor(ids, dtype=torch.long)
        output = output_dir / f"{split}.pt"
        torch.save(tensor, output)
        counts[split] = {
            "tokens": len(ids),
            "source_sha256": sha256(source),
            "tensor_sha256": sha256(output),
            "tensor_file": str(output),
        }

    record = {
        "dataset_name": corpus_manifest.get("dataset_name"),
        "dataset_id": training_record.get("dataset_id"),
        "dataset_revision": training_record.get("dataset_revision"),
        "corpus_manifest_sha256": sha256(corpus_manifest_path),
        "train_source_sha256": sha256(train_path),
        "validation_source_sha256": sha256(validation_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "tokenizer_training_record_sha256": sha256(training_record_path),
        "preparation_manifest_sha256": sha256(prep_manifest_path),
        "vocab_size": training_record["vocab_size_actual"],
        "train_tokens": counts["train"]["tokens"],
        "validation_tokens": counts["validation"]["tokens"],
        "train_tensor_sha256": counts["train"]["tensor_sha256"],
        "validation_tensor_sha256": counts["validation"]["tensor_sha256"],
        "splits": counts,
        "validation_used_for_tokenizer_training": False,
        "test_split_read_or_used": False,
        "note": "Separate train/validation token streams. Test split remains held out. Keep attribution and applicable share-alike notices with derived artifacts.",
    }
    record_path = output_dir / "preparation-manifest.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # train_tiny_corpus.py consumes metadata.json; retain the descriptive record too.
    (output_dir / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
