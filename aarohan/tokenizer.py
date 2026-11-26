"""Project-owned BPE tokenizer training and versioned loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = [
    "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|system|>",
    "<|user|>", "<|assistant|>", "<|tool|>", "<|separator|>",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_bpe(
    train_path: Path,
    output_dir: Path,
    *,
    vocab_size: int = 256,
    min_frequency: int = 1,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Train BPE only from the specified local training file (never validation)."""
    if not train_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {train_path}")
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must exceed {len(SPECIAL_TOKENS)} special tokens")
    content = train_path.read_text(encoding="utf-8")
    if not any(line.strip() and not line.lstrip().startswith("#") for line in content.splitlines()):
        raise ValueError("Training corpus is empty")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(train_path.resolve())], trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    effective_vocab = tokenizer.get_vocab()
    metadata = {
        "tokenizer_name": "AAROHAN-BPE",
        "version": "0.1.0",
        "algorithm": "ByteLevel BPE",
        "vocab_size_target": vocab_size,
        "vocab_size_note": "ByteLevel's 256 required byte symbols plus special tokens can make the actual vocabulary exceed the requested merge target.",
        "vocab_size_actual": len(effective_vocab),
        "special_tokens": SPECIAL_TOKENS,
        "training_file": train_path.name,
        "training_file_sha256": sha256_file(train_path),
        "provenance_manifest": str(provenance_path) if provenance_path else None,
        "provenance_manifest_sha256": sha256_file(provenance_path) if provenance_path and provenance_path.is_file() else None,
        "validation_data_used_for_training": False,
        "unicode_handling": "UTF-8 input with byte-level pre-tokenization",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"Tokenizer is missing required special tokens: {missing}")
    return tokenizer

