"""Test end-to-end 16-step release journey in accordance with PRD Prompt 16."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import sha256_file, train_bpe
from scripts.verify_release import generate_release_report, run_release_journey


def test_complete_release_journey(tmp_path: Path):
    # 1. Setup mini tokenizer
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "AAROHAN is an autonomous AI system with causal transformer architecture and regression gating.\n"
        "Memory stores verified and unverified facts across 6 cognitive layers.\n",
        encoding="utf-8",
    )
    tok_dir = tmp_path / "tokenizer"
    train_bpe(corpus_file, tok_dir, vocab_size=265)
    tok_path = tok_dir / "tokenizer.json"

    # 2. Setup mini base model checkpoint
    cfg = AAROHANConfig(
        vocab_size=265,
        context_length=32,
        layers=1,
        heads=2,
        width=16,
    )
    base_model = AAROHANLM(cfg)
    base_ckpt = tmp_path / "base_model.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "model_state": base_model.state_dict(),
            "tokenizer_sha256": sha256_file(tok_path),
            "training_note": "Test release baseline",
        },
        base_ckpt,
    )

    # 3. Setup mini replay records
    replay_file = tmp_path / "replay.jsonl"
    replay_data = [
        {
            "messages": [
                {"role": "system", "content": "You are AAROHAN, an independent AI assistant."},
                {"role": "user", "content": "Hello AAROHAN"},
                {"role": "assistant", "content": "Hello! I am an independent AI model."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are AAROHAN, an independent AI assistant."},
                {"role": "user", "content": "What is memory?"},
                {"role": "assistant", "content": "Memory is 6-layer cognitive storage."},
            ]
        },
    ]
    with replay_file.open("w", encoding="utf-8") as f:
        for r in replay_data:
            f.write(json.dumps(r) + "\n")

    # 4. Setup approved research document
    doc_file = tmp_path / "approved_doc.md"
    doc_file.write_text(
        "# Autonomous Systems\n\n"
        "Regression gating protects model weights against catastrophic forgetting.\n"
        "Memory items are verified before promotion to permanent knowledge.\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "release_test.sqlite"
    candidate_ckpt = tmp_path / "candidate_model.pt"
    registry_path = tmp_path / "deployment_registry.json"
    artifacts_dir = tmp_path / "artifacts"
    report_file = tmp_path / "release-report.md"

    # 5. Run full 16-step journey
    journey_res = run_release_journey(
        db_path=db_path,
        tokenizer_path=tok_path,
        base_checkpoint_path=base_ckpt,
        candidate_checkpoint_path=candidate_ckpt,
        registry_path=registry_path,
        replay_file=replay_file,
        document_path=doc_file,
        artifacts_dir=artifacts_dir,
        device_name="cpu",
    )

    assert journey_res["success"] is True
    assert journey_res["journey_steps_completed"] == 16
    assert len(journey_res["steps"]) == 16
    for s in journey_res["steps"]:
        assert s["status"] == "PASSED"

    # 6. Generate and verify report
    report_content = generate_release_report(journey_res, report_file)
    assert report_file.is_file()
    assert "AAROHAN AI Platform — Release Certification Report" in report_content
    assert "Step 16: Record Outcome & Registry Update" in report_content
