"""Encode validated corpus splits into reproducible PyTorch token-ID tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.tokenizer import load_tokenizer, sha256_file
from scripts.validate_corpus import read_paragraphs, validate


def prepare(corpus_dir: Path, tokenizer_dir: Path, output_dir: Path) -> dict[str, object]:
    corpus_dir = corpus_dir.resolve()
    tokenizer_dir = tokenizer_dir.resolve()
    output_dir = output_dir.resolve()
    result = validate(corpus_dir)
    if not result["ready"]:
        raw_errors = result.get("errors")
        errors = [str(err) for err in raw_errors] if isinstance(raw_errors, list) else []
        raise ValueError("Corpus validation failed: " + "; ".join(errors))

    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    train_path = corpus_dir / manifest["train_file"]
    validation_path = corpus_dir / manifest["validation_file"]
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    train_text = "\n".join(read_paragraphs(train_path))
    validation_text = "\n".join(read_paragraphs(validation_path))
    train_ids = tokenizer.encode(train_text, add_special_tokens=False).ids
    validation_ids = tokenizer.encode(validation_text, add_special_tokens=False).ids
    if not train_ids or not validation_ids:
        raise ValueError("Tokenized training and validation data must both be non-empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    validation_tensor = torch.tensor(validation_ids, dtype=torch.long)
    torch.save(train_tensor, output_dir / "train.pt")
    torch.save(validation_tensor, output_dir / "validation.pt")
    metadata: dict[str, object] = {
        "dataset_name": manifest["dataset_name"],
        "corpus_manifest_sha256": sha256_file(corpus_dir / "manifest.json"),
        "train_source_sha256": sha256_file(train_path),
        "validation_source_sha256": sha256_file(validation_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_tensor_sha256": sha256_file(output_dir / "train.pt"),
        "validation_tensor_sha256": sha256_file(output_dir / "validation.pt"),
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "vocab_size": tokenizer.get_vocab_size(),
        "validation_used_for_tokenizer_training": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpora" / "original")
    parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "tokenized" / "original-toy-v1")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.corpus_dir, args.tokenizer_dir, args.output_dir), indent=2))
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

