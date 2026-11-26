import json

import pytest
import torch

from aarohan.tokenizer import train_bpe
from scripts.prepare_tokenized_data import prepare
from scripts.train_tiny_corpus import train
from scripts.evaluate_checkpoints import evaluate_checkpoint


def test_tiny_corpus_training_saves_reloadable_checkpoint(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    training = "The curious robot studies a small garden. " * 20
    validation = "A quiet clock rests beside the blue window. " * 10
    (corpus_dir / "corpus.txt").write_text(training, encoding="utf-8")
    (corpus_dir / "validation.txt").write_text(validation, encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(json.dumps({
        "dataset_name": "tiny generated test",
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

    checkpoint = tmp_path / "checkpoints" / "candidate.pt"
    result = train(corpus_dir, tokenizer_dir, tokenized_dir, checkpoint,
                   steps=2, context=16, batch_size=2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert result["steps"] == 2
    assert result["validation_loss"] > 0
    assert result["checkpoint_reload_max_logit_difference"] == 0.0
    assert result["validation_tokens_scored"] == result["validation_tokens"] - 1
    assert payload["completed_steps"] == 2
    assert payload["optimizer_state"]
    evaluation = evaluate_checkpoint(checkpoint, tokenized_dir,
                                    tokenizer_dir / "tokenizer.json", torch.device("cpu"))
    assert evaluation["validation_loss"] == pytest.approx(result["validation_loss"], rel=1e-6)


def test_training_resume_matches_uninterrupted_run(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus.txt").write_text("The robot follows the path around the garden. " * 20, encoding="utf-8")
    (corpus_dir / "validation.txt").write_text("A lantern glows beside the quiet river. " * 10, encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(json.dumps({
        "dataset_name": "tiny generated resume test",
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

    full_path = tmp_path / "full.pt"
    split_path = tmp_path / "split.pt"
    common = dict(corpus_dir=corpus_dir, tokenizer_dir=tokenizer_dir,
                  tokenized_dir=tokenized_dir, device_name="cpu", seed=91,
                  context=16, batch_size=2)
    train(**common, output=full_path, steps=4)
    train(**common, output=split_path, steps=2)
    train(**common, output=split_path, steps=4, resume_from=split_path)
    full = torch.load(full_path, map_location="cpu", weights_only=False)
    resumed = torch.load(split_path, map_location="cpu", weights_only=False)
    assert resumed["completed_steps"] == 4
    for key, tensor in full["model_state"].items():
        torch.testing.assert_close(tensor, resumed["model_state"][key], rtol=0, atol=0)

