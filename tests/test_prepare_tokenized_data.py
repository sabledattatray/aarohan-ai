import json

import torch

from aarohan.tokenizer import train_bpe
from scripts.prepare_tokenized_data import prepare


def test_prepare_writes_distinct_train_and_validation_tensors(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus.txt").write_text("Training sentence one.\nTraining sentence two.\n", encoding="utf-8")
    (corpus_dir / "validation.txt").write_text("Different validation sentence.\n", encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(json.dumps({
        "dataset_name": "tiny test",
        "author": "test author",
        "created_date": "2026-11-25",
        "license_or_permission": "original test-only text",
        "status": "approved_for_local_experiment",
        "privacy_reviewed": True,
        "third_party_text_included": False,
        "train_file": "corpus.txt",
        "validation_file": "validation.txt",
    }), encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_bpe(corpus_dir / "corpus.txt", tokenizer_dir, vocab_size=300)

    output_dir = tmp_path / "encoded"
    metadata = prepare(corpus_dir, tokenizer_dir, output_dir)
    train_ids = torch.load(output_dir / "train.pt", weights_only=True)
    validation_ids = torch.load(output_dir / "validation.pt", weights_only=True)
    assert train_ids.dtype == torch.long and train_ids.numel() == metadata["train_tokens"]
    assert validation_ids.dtype == torch.long and validation_ids.numel() == metadata["validation_tokens"]
    assert metadata["validation_used_for_tokenizer_training"] is False
    assert (output_dir / "metadata.json").is_file()

