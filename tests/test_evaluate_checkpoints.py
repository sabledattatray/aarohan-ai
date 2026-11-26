import json

import torch

from aarohan.tokenizer import train_bpe
from scripts.evaluate_checkpoints import evaluate_many
from scripts.prepare_tokenized_data import prepare
from scripts.train_tiny_corpus import train


def test_evaluator_scores_checkpoint_on_held_out_stream(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus.txt").write_text("The robot walks beside the river. " * 20, encoding="utf-8")
    (corpus_dir / "validation.txt").write_text("A quiet bird rests on a branch. " * 10, encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(json.dumps({
        "dataset_name": "evaluation test fixture",
        "author": "assistant-generated test fixture",
        "created_date": "2026-11-25",
        "license_or_permission": "test fixture only",
        "status": "approved_for_local_experiment",
        "privacy_reviewed": True,
        "third_party_text_included": False,
        "train_file": "corpus.txt",
        "validation_file": "validation.txt",
    }), encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_bpe(corpus_dir / "corpus.txt", tokenizer_dir, vocab_size=300)
    tokenized_dir = tmp_path / "tokenized"
    prepare(corpus_dir, tokenizer_dir, tokenized_dir)
    checkpoint = tmp_path / "checkpoint.pt"
    train(corpus_dir, tokenizer_dir, tokenized_dir, checkpoint,
          steps=2, context=16, batch_size=2)

    result = evaluate_many([checkpoint], tokenized_dir,
                           tokenizer_dir / "tokenizer.json", "cpu")
    row = result["results"][0]
    expected_targets = torch.load(tokenized_dir / "validation.pt", weights_only=True).numel() - 1
    assert row["validation_tokens_scored"] == expected_targets
    assert row["validation_loss"] > 0
    assert row["validation_perplexity"] > 1
    assert row["validation_loss_delta_vs_first"] == 0

