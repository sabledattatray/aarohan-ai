"""Append-only local registry for candidate model checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "candidates": [], "active_checkpoint_id": None, "activation_history": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("candidates"), list):
        raise ValueError("Unsupported or malformed model registry")
    return registry


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".registry-", suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def register_candidate(checkpoint_path: Path, evaluation_path: Path, registry_path: Path) -> dict[str, Any]:
    checkpoint_path, evaluation_path, registry_path = map(Path.resolve, (checkpoint_path, evaluation_path, registry_path))
    if not checkpoint_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("Checkpoint and evaluation report must both exist")
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    checkpoint_key = str(checkpoint_path)
    result = next((item for item in report.get("results", []) if item.get("checkpoint") == checkpoint_key), None)
    if result is None:
        raise ValueError("Evaluation report does not contain this checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if result.get("tokenizer_sha256") != checkpoint.get("tokenizer_sha256"):
        raise ValueError("Evaluation tokenizer hash does not match the checkpoint")
    data = checkpoint.get("dataset_metadata", {})
    if result.get("validation_source_sha256") != data.get("validation_source_sha256"):
        raise ValueError("Evaluation validation source does not match the checkpoint's dataset metadata")
    digest = sha256_file(checkpoint_path)
    candidate_id = digest[:16]
    registry = _read_registry(registry_path)
    existing = next((item for item in registry["candidates"] if item["checkpoint_sha256"] == digest), None)
    if existing:
        return existing
    entry = {
        "checkpoint_id": candidate_id,
        "checkpoint_path": checkpoint_key,
        "checkpoint_sha256": digest,
        "evaluation_report_path": str(evaluation_path),
        "evaluation_report_sha256": sha256_file(evaluation_path),
        "completed_steps": result.get("completed_steps"),
        "model_config": checkpoint.get("config"),
        "tokenizer_sha256": result["tokenizer_sha256"],
        "validation_source_sha256": result["validation_source_sha256"],
        "validation_loss": result["validation_loss"],
        "validation_perplexity": result["validation_perplexity"],
        "status": "candidate",
        "note": checkpoint.get("training_note", ""),
    }
    registry["candidates"].append(entry)
    _write_atomic(registry_path, registry)
    return entry


def activate_candidate(checkpoint_id: str, registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = _read_registry(registry_path)
    candidate = next((item for item in registry["candidates"] if item["checkpoint_id"] == checkpoint_id), None)
    if candidate is None:
        raise KeyError(f"Unknown checkpoint id: {checkpoint_id}")
    checkpoint_path = Path(candidate["checkpoint_path"])
    evaluation_path = Path(candidate["evaluation_report_path"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise ValueError("Checkpoint is missing or changed since registration; refusing activation")
    if not evaluation_path.is_file() or sha256_file(evaluation_path) != candidate["evaluation_report_sha256"]:
        raise ValueError("Evaluation report is missing or changed since registration; refusing activation")
    previous = registry.get("active_checkpoint_id")
    registry.setdefault("activation_history", []).append({
        "previous_checkpoint_id": previous,
        "new_checkpoint_id": checkpoint_id,
    })
    registry["active_checkpoint_id"] = checkpoint_id
    _write_atomic(registry_path, registry)
    return registry


def rollback_active(registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = _read_registry(registry_path)
    history = registry.get("activation_history", [])
    if not history:
        raise ValueError("No activation history to roll back")
    previous = history[-1].get("previous_checkpoint_id")
    if previous is None:
        raise ValueError("No prior active checkpoint is available")
    history.append({
        "previous_checkpoint_id": registry.get("active_checkpoint_id"),
        "new_checkpoint_id": previous,
        "action": "rollback",
    })
    registry["active_checkpoint_id"] = previous
    _write_atomic(registry_path, registry)
    return registry

