"""Run a bounded synthetic next-token training experiment and save a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

# Direct execution sets sys.path[0] to scripts/, so explicitly add the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.model import AAROHANConfig, AAROHANLM


def make_batch(batch_size: int, sequence_length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # A tiny synthetic repeating-token task validates the training pipeline only.
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(sequence_length + 1, device=device).unsqueeze(0)
    sequence = (starts + offsets) % vocab_size
    return sequence[:, :-1], sequence[:, 1:]


def atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run(steps: int, device_name: str, output: Path, seed: int) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    config = AAROHANConfig(vocab_size=64, context_length=32, layers=2, heads=4, width=64)
    model = AAROHANLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_losses: list[float] = []

    for _ in range(steps):
        model.train()
        x, y = make_batch(16, 32, config.vocab_size, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        x, y = make_batch(16, 32, config.vocab_size, device)
        _, validation_loss = model(x, y)
    assert validation_loss is not None

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config_dict(),
        "completed_steps": steps,
        "seed": seed,
        "device": str(device),
        "dataset": "synthetic repeating-token pipeline smoke test; not language data",
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_save(payload, output)

    # Verify serialization and architecture compatibility without replacing artifacts.
    reloaded = AAROHANLM(config).to(device)
    checkpoint = torch.load(output, map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        original_logits, _ = model(x)
        reloaded_logits, _ = reloaded(x)
    max_reload_difference = float((original_logits - reloaded_logits).abs().max().cpu())

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": model.parameter_count(),
        "steps": steps,
        "initial_training_loss": train_losses[0],
        "final_training_loss": train_losses[-1],
        "validation_loss": float(validation_loss.cpu()),
        "checkpoint": str(output.resolve()),
        "checkpoint_reload_max_logit_difference": max_reload_difference,
        "note": "Synthetic data checks pipeline mechanics only; it is not evidence of language learning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tiny-smoke.pt"))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(json.dumps(run(args.steps, args.device, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

