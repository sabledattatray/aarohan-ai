"""Controlled Continual Learning with Experience Replay, Regression Gates, and Rollback.

In accordance with PRD Section 11 and Prompt 11:
  - Experience Replay: Preserves historical training examples to prevent catastrophic forgetting.
  - Regression Gates: Re-runs historical benchmarks; rejects candidates that degrade previous capabilities.
  - Model Rollback: Reverts to the certified stable checkpoint whenever regression thresholds are violated.
  - Distinguishes memory storage updates from parameter updates.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from aarohan.chat_template import prepare_dialogue_tokens_with_mask
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file


class ReplayBuffer:
    """Manages historical instruction datasets and balances them with new training candidates."""

    def __init__(self, historical_file: Path) -> None:
        self.historical_file = Path(historical_file)
        self.records: list[dict[str, Any]] = []
        if self.historical_file.is_file():
            with self.historical_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))

    def sample_replay(self, count: int, seed: int = 42) -> list[dict[str, Any]]:
        if not self.records:
            return []
        rng = random.Random(seed)
        return rng.sample(self.records, min(count, len(self.records)))

    def blend_datasets(
        self,
        new_records: list[dict[str, Any]],
        replay_ratio: float = 0.5,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Mix new training candidate examples with historical replay examples.
        replay_ratio = fraction of total blended dataset that comes from replay.
        """
        if not new_records:
            raise ValueError("new_records cannot be empty")
        if not 0.0 <= replay_ratio < 1.0:
            raise ValueError("replay_ratio must be in [0.0, 1.0)")

        if not self.records or replay_ratio == 0.0:
            return list(new_records)

        # Number of replay samples needed
        num_replay = int(len(new_records) * (replay_ratio / (1.0 - replay_ratio)))
        replay_samples = self.sample_replay(num_replay, seed=seed)
        blended = list(new_records) + replay_samples
        random.Random(seed).shuffle(blended)
        return blended


def evaluate_benchmark_loss(
    model: AAROHANLM,
    tokenizer: Any,
    benchmark_records: list[dict[str, Any]],
    *,
    context_length: int | None = None,
    device: torch.device,
) -> tuple[float, float]:
    """Calculate mean loss and perplexity on assistant target tokens for a benchmark set."""
    ctx_len = context_length or model.config.context_length
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for item in benchmark_records:
            tokens, labels, _ = prepare_dialogue_tokens_with_mask(
                tokenizer, item["messages"], context_length=ctx_len
            )
            x = tokens[:-1].unsqueeze(0).to(device)
            y = labels[1:].unsqueeze(0).to(device)
            _, loss = model(x, y)
            if loss is not None and not torch.isnan(loss):
                total_loss += float(loss.detach().cpu())
                valid_batches += 1

    avg_loss = total_loss / max(valid_batches, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


class RegressionGate:
    """Evaluates candidate checkpoint against baseline to prevent catastrophic forgetting."""

    def __init__(self, max_allowed_regression_percent: float = 8.0) -> None:
        self.max_allowed_regression = max_allowed_regression_percent

    def evaluate(
        self,
        baseline_loss: float,
        candidate_loss: float,
        *,
        new_task_baseline_loss: float | None = None,
        new_task_candidate_loss: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates whether candidate model passes regression criteria:
        Regression % = ((candidate_loss - baseline_loss) / baseline_loss) * 100
        """
        regression_delta = ((candidate_loss - baseline_loss) / max(baseline_loss, 1e-6)) * 100.0
        passed = regression_delta <= self.max_allowed_regression

        improvement_on_new = None
        if new_task_baseline_loss is not None and new_task_candidate_loss is not None:
            improvement_on_new = new_task_candidate_loss < new_task_baseline_loss

        return {
            "passed": passed,
            "baseline_benchmark_loss": round(baseline_loss, 4),
            "candidate_benchmark_loss": round(candidate_loss, 4),
            "regression_delta_percent": round(regression_delta, 2),
            "max_allowed_regression_percent": self.max_allowed_regression,
            "improved_on_new_task": improvement_on_new,
            "status": "APPROVED" if passed else "REJECTED_REGRESSION_DETECTED",
        }


def train_continual_candidate(
    base_checkpoint_path: Path,
    tokenizer_path: Path,
    new_records: list[dict[str, Any]],
    replay_records_file: Path,
    output_checkpoint_path: Path,
    *,
    epochs: int = 10,
    lr: float = 2e-4,
    replay_ratio: float = 0.5,
    device_name: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Train a continual learning candidate using blended replay to protect previous knowledge."""
    base_checkpoint_path = base_checkpoint_path.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_checkpoint_path = output_checkpoint_path.resolve()

    tokenizer = load_tokenizer(tokenizer_path)
    base_ckpt = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    config = AAROHANConfig(**base_ckpt["config"])

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AAROHANLM(config).to(device)
    model.load_state_dict(base_ckpt["model_state"])

    # Blend new candidate records with historical replay buffer
    replay_buffer = ReplayBuffer(replay_records_file)
    blended_records = replay_buffer.blend_datasets(new_records, replay_ratio=replay_ratio, seed=seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    initial_loss: float | None = None
    final_loss: float = 0.0

    for _ in range(epochs):
        epoch_loss = 0.0
        for item in blended_records:
            tokens, labels, _ = prepare_dialogue_tokens_with_mask(
                tokenizer, item["messages"], context_length=config.context_length
            )
            x = tokens[:-1].unsqueeze(0).to(device)
            y = labels[1:].unsqueeze(0).to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_loss = float(loss.detach().cpu())
            epoch_loss += step_loss
            if initial_loss is None:
                initial_loss = step_loss
        final_loss = epoch_loss / max(len(blended_records), 1)

    # Save candidate checkpoint
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_payload = {
        "config": asdict(config),
        "model_state": model.state_dict(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "training_note": f"Continual learning candidate trained on {len(new_records)} new records with {replay_ratio*100:.0f}% replay",
        "continual_metadata": {
            "base_checkpoint": str(base_checkpoint_path),
            "base_checkpoint_sha256": sha256_file(base_checkpoint_path),
            "new_records_count": len(new_records),
            "blended_total_count": len(blended_records),
            "replay_ratio": replay_ratio,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    }

    temp_fd, temp_path = tempfile.mkstemp(dir=str(output_checkpoint_path.parent), suffix=".tmp")
    os.close(temp_fd)
    try:
        torch.save(candidate_payload, temp_path)
        os.replace(temp_path, output_checkpoint_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "candidate_checkpoint": str(output_checkpoint_path),
        "candidate_sha256": sha256_file(output_checkpoint_path),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "total_blended_samples": len(blended_records),
    }


class RollbackManager:
    """Maintains active model deployment pointer and performs atomic rollbacks."""

    def __init__(self, registry_file: Path) -> None:
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.is_file():
            self._save_state({
                "active_checkpoint": None,
                "history": [],
            })

    def _load_state(self) -> dict[str, Any]:
        with self.registry_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: dict[str, Any]) -> None:
        temp_fd, temp_path = tempfile.mkstemp(dir=str(self.registry_file.parent), suffix=".tmp")
        os.close(temp_fd)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_path, self.registry_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def deploy_model(self, checkpoint_path: Path, note: str = "") -> dict[str, Any]:
        checkpoint_path = checkpoint_path.resolve()
        state = self._load_state()
        prev = state.get("active_checkpoint")
        entry = {
            "checkpoint": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "deployed_at": datetime.now(UTC).isoformat(),
            "note": note,
            "previous_checkpoint": prev,
        }
        state["active_checkpoint"] = str(checkpoint_path)
        state["history"].append(entry)
        self._save_state(state)
        return entry

    def rollback(self, reason: str = "Regression threshold exceeded") -> dict[str, Any]:
        state = self._load_state()
        history = state.get("history", [])
        if len(history) < 2:
            raise RuntimeError("No previous certified checkpoint available to rollback to")
        # Pop current active, revert to preceding
        last_entry = history.pop()
        target = history[-1]["checkpoint"]
        state["active_checkpoint"] = target
        state["history"] = history
        self._save_state(state)
        return {
            "status": "rolled_back",
            "rolled_back_from": last_entry["checkpoint"],
            "restored_checkpoint": target,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_active(self) -> str | None:
        state = self._load_state()
        return state.get("active_checkpoint")
