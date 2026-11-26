import json

import torch

from aarohan.registry import activate_candidate, register_candidate, rollback_active


def _make_pair(tmp_path, name: str):
    checkpoint = tmp_path / f"{name}.pt"
    evaluation = tmp_path / f"{name}.json"
    torch.save({
        "config": {"vocab_size": 16, "context_length": 8, "layers": 1, "heads": 2, "width": 16, "dropout": 0.0},
        "completed_steps": 10,
        "tokenizer_sha256": "tokenizer-hash",
        "dataset_metadata": {"validation_source_sha256": "validation-hash"},
        "training_note": "test candidate",
    }, checkpoint)
    evaluation.write_text(json.dumps({"results": [{
        "checkpoint": str(checkpoint.resolve()),
        "completed_steps": 10,
        "validation_loss": 2.5,
        "validation_perplexity": 12.18,
        "tokenizer_sha256": "tokenizer-hash",
        "validation_source_sha256": "validation-hash",
    }]}), encoding="utf-8")
    return checkpoint, evaluation


def test_registry_candidate_requires_matching_evidence_and_can_rollback(tmp_path) -> None:
    checkpoint, evaluation = _make_pair(tmp_path, "candidate")
    checkpoint2, evaluation2 = _make_pair(tmp_path, "candidate-2")
    registry_path = tmp_path / "registry.json"
    entry = register_candidate(checkpoint, evaluation, registry_path)
    entry2 = register_candidate(checkpoint2, evaluation2, registry_path)
    assert entry["status"] == "candidate"
    assert register_candidate(checkpoint, evaluation, registry_path)["checkpoint_id"] == entry["checkpoint_id"]

    activate_candidate(entry["checkpoint_id"], registry_path)
    active = activate_candidate(entry2["checkpoint_id"], registry_path)
    assert active["active_checkpoint_id"] == entry2["checkpoint_id"]
    rolled = rollback_active(registry_path)
    assert rolled["active_checkpoint_id"] == entry["checkpoint_id"]


def test_registry_rejects_unmatched_evaluation_evidence(tmp_path) -> None:
    checkpoint, evaluation = _make_pair(tmp_path, "bad-candidate")
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    report["results"][0]["validation_source_sha256"] = "other-validation"
    evaluation.write_text(json.dumps(report), encoding="utf-8")
    try:
        register_candidate(checkpoint, evaluation, tmp_path / "registry.json")
    except ValueError as error:
        assert "validation source" in str(error)
    else:
        raise AssertionError("Registry must reject mismatched evaluation evidence")


def test_registry_refuses_activation_if_checkpoint_changes(tmp_path) -> None:
    checkpoint, evaluation = _make_pair(tmp_path, "mutable-candidate")
    registry_path = tmp_path / "registry.json"
    entry = register_candidate(checkpoint, evaluation, registry_path)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    try:
        activate_candidate(entry["checkpoint_id"], registry_path)
    except ValueError as error:
        assert "changed since registration" in str(error)
    else:
        raise AssertionError("Changed checkpoint must not be activated")

