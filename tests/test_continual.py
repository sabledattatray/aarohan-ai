import json

import pytest
import torch

from aarohan.continual import (
    RegressionGate,
    ReplayBuffer,
    RollbackManager,
    train_continual_candidate,
)
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, train_bpe
from scripts.continual_cli import run


def test_replay_buffer_sampling_and_blending(tmp_path) -> None:
    hist_file = tmp_path / "historical.jsonl"
    hist_records = [
        {"id": f"hist_{i}", "messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
        for i in range(10)
    ]
    hist_file.write_text("\n".join(json.dumps(r) for r in hist_records) + "\n", encoding="utf-8")

    buf = ReplayBuffer(hist_file)
    assert len(buf.records) == 10

    # Sample 4
    samples = buf.sample_replay(4)
    assert len(samples) == 4

    # Blend with 2 new records at 50% ratio
    new_records = [
        {"id": "new_1", "messages": [{"role": "user", "content": "N1"}, {"role": "assistant", "content": "A1"}]},
        {"id": "new_2", "messages": [{"role": "user", "content": "N2"}, {"role": "assistant", "content": "A2"}]},
    ]
    blended = buf.blend_datasets(new_records, replay_ratio=0.5)
    assert len(blended) == 4
    assert any(r["id"].startswith("hist_") for r in blended)
    assert any(r["id"].startswith("new_") for r in blended)


def test_regression_gate_passes_and_rejects() -> None:
    gate = RegressionGate(max_allowed_regression_percent=5.0)

    # 1. Passes when loss is equal or improved
    pass_res = gate.evaluate(baseline_loss=5.0, candidate_loss=5.1)
    assert pass_res["passed"] is True
    assert pass_res["status"] == "APPROVED"
    assert pass_res["regression_delta_percent"] == 2.0

    # 2. Rejects when candidate suffers catastrophic forgetting (e.g. +10% regression)
    fail_res = gate.evaluate(baseline_loss=5.0, candidate_loss=5.6)
    assert fail_res["passed"] is False
    assert fail_res["status"] == "REJECTED_REGRESSION_DETECTED"
    assert fail_res["regression_delta_percent"] == 12.0


def test_rollback_manager_deployment_and_reversion(tmp_path) -> None:
    reg_file = tmp_path / "registry.json"
    mgr = RollbackManager(reg_file)

    # Mock checkpoints
    ckpt_a = tmp_path / "model_a.pt"
    ckpt_b = tmp_path / "model_b.pt"
    ckpt_a.write_text("checkpoint A", encoding="utf-8")
    ckpt_b.write_text("checkpoint B", encoding="utf-8")

    # Deploy A
    mgr.deploy_model(ckpt_a, note="Initial production model")
    assert mgr.get_active() == str(ckpt_a.resolve())

    # Deploy B
    mgr.deploy_model(ckpt_b, note="Candidate B update")
    assert mgr.get_active() == str(ckpt_b.resolve())

    # Rollback to A
    rb_res = mgr.rollback(reason="Regression test failure on arithmetic benchmark")
    assert rb_res["status"] == "rolled_back"
    assert rb_res["restored_checkpoint"] == str(ckpt_a.resolve())
    assert mgr.get_active() == str(ckpt_a.resolve())

    # Further rollback without previous history raises RuntimeError
    with pytest.raises(RuntimeError, match="No previous certified checkpoint"):
        mgr.rollback()


def test_continual_training_candidate_and_cli(tmp_path) -> None:
    # 1. Train tiny tokenizer
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("AAROHAN continual learning test.\n" * 20, encoding="utf-8")
    tok_dir = tmp_path / "tok"
    train_bpe(corpus, tok_dir, vocab_size=280)
    tok_path = tok_dir / "tokenizer.json"
    tok = load_tokenizer(tok_path)

    # 2. Build base checkpoint
    cfg = AAROHANConfig(vocab_size=tok.get_vocab_size(), context_length=32, layers=1, heads=2, width=16)
    model = AAROHANLM(cfg)
    base_ckpt = tmp_path / "base.pt"
    torch.save({
        "config": model.config_dict(),
        "model_state": model.state_dict(),
        "tokenizer_sha256": tok_path.name,
    }, base_ckpt)

    # 3. Create historical replay and new data files
    hist_file = tmp_path / "hist.jsonl"
    new_file = tmp_path / "new.jsonl"
    item1 = {"id": "h1", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "h_q"}, {"role": "assistant", "content": "h_a"}]}
    item2 = {"id": "n1", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "n_q"}, {"role": "assistant", "content": "n_a"}]}
    hist_file.write_text(json.dumps(item1) + "\n", encoding="utf-8")
    new_file.write_text(json.dumps(item2) + "\n", encoding="utf-8")

    out_ckpt = tmp_path / "candidate.pt"
    train_res = train_continual_candidate(
        base_checkpoint_path=base_ckpt,
        tokenizer_path=tok_path,
        new_records=[item2],
        replay_records_file=hist_file,
        output_checkpoint_path=out_ckpt,
        epochs=3,
        replay_ratio=0.5,
        device_name="cpu",
    )
    assert out_ckpt.is_file()
    assert train_res["total_blended_samples"] == 2

    # 4. CLI gate evaluate
    gate_eval = run([
        "eval-gate",
        "--base-checkpoint", str(base_ckpt),
        "--candidate-checkpoint", str(out_ckpt),
        "--tokenizer", str(tok_path),
        "--benchmark-file", str(hist_file),
    ])
    assert "passed" in gate_eval
    assert "regression_delta_percent" in gate_eval
