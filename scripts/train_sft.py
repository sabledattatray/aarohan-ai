"""Supervised Fine-Tuning (SFT) trainer with assistant-target masking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.chat_template import prepare_dialogue_tokens_with_mask
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file
from scripts.validate_instruction_dataset import validate_instruction_file


def load_instruction_dataset(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train_sft(
    train_file: Path,
    val_file: Path,
    tokenizer_path: Path,
    output_checkpoint: Path,
    *,
    base_checkpoint_path: Path | None = None,
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    device_name: str = "cpu",
    seed: int = 42,
    context_length: int = 128,
) -> dict[str, Any]:
    train_file, val_file = train_file.resolve(), val_file.resolve()
    tokenizer_path, output_checkpoint = tokenizer_path.resolve(), output_checkpoint.resolve()

    # Validate instruction files
    train_report = validate_instruction_file(train_file)
    val_report = validate_instruction_file(val_file)

    tokenizer = load_tokenizer(tokenizer_path)
    tok_sha256 = sha256_file(tokenizer_path)

    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base_sha256 = None
    if base_checkpoint_path is not None and base_checkpoint_path.is_file():
        base_checkpoint = torch.load(base_checkpoint_path.resolve(), map_location="cpu", weights_only=False)
        base_sha256 = sha256_file(base_checkpoint_path)
        if base_checkpoint.get("tokenizer_sha256") != tok_sha256:
            raise ValueError("Tokenizer hash mismatch with base checkpoint")
        config = AAROHANConfig(**base_checkpoint["config"])
        # Override context length if needed, or use model's context length
        context_length = min(context_length, config.context_length)
        model = AAROHANLM(config)
        model.load_state_dict(base_checkpoint["model_state"])
    else:
        # Default small config for training from scratch
        config = AAROHANConfig(
            vocab_size=tokenizer.get_vocab_size(),
            context_length=context_length,
            layers=2,
            heads=4,
            width=128,
            dropout=0.0,
        )
        model = AAROHANLM(config)

    model = model.to(device)

    train_data = load_instruction_dataset(train_file)
    val_data = load_instruction_dataset(val_file)

    def prepare_batch(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_tokens = []
        batch_labels = []
        for item in items:
            tokens, labels, _ = prepare_dialogue_tokens_with_mask(
                tokenizer, item["messages"], context_length=context_length
            )
            # Autoregressive shift: input is tokens[:-1], label target is labels[1:]
            x = tokens[:-1]
            y = labels[1:]
            batch_tokens.append(x)
            batch_labels.append(y)
        return torch.stack(batch_tokens).to(device), torch.stack(batch_labels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    initial_loss: float | None = None
    final_loss: float = 0.0

    model.train()
    for epoch in range(epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, len(train_data), batch_size):
            batch_items = train_data[i : i + batch_size]
            x, y = prepare_batch(batch_items)
            _, loss = model(x, y)
            assert loss is not None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_loss = float(loss.detach().cpu())
            epoch_loss += step_loss
            batches += 1
            if initial_loss is None:
                initial_loss = step_loss

        final_loss = epoch_loss / max(batches, 1)

    # Evaluate on held-out validation instructions
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch_items = val_data[i : i + batch_size]
            x, y = prepare_batch(batch_items)
            _, loss = model(x, y)
            if loss is not None and not torch.isnan(loss):
                val_loss_sum += float(loss.detach().cpu())
                val_batches += 1

    val_loss = val_loss_sum / max(val_batches, 1)
    val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

    sft_metadata = {
        "train_file_sha256": train_report["sha256"],
        "val_file_sha256": val_report["sha256"],
        "train_records": train_report["record_count"],
        "val_records": val_report["record_count"],
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "validation_loss": val_loss,
        "validation_perplexity": val_perplexity,
        "base_checkpoint_sha256": base_sha256,
    }

    # Save checkpoint atomically
    checkpoint_payload = {
        "config": model.config_dict(),
        "model_state": model.state_dict(),
        "tokenizer_sha256": tok_sha256,
        "training_note": "Supervised Fine-Tuned (SFT) model on instruction dataset.",
        "sft_metadata": sft_metadata,
    }

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=str(output_checkpoint.parent), suffix=".tmp")
    os.close(temp_fd)
    try:
        torch.save(checkpoint_payload, temp_path)
        os.replace(temp_path, output_checkpoint)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": "completed",
        "output_checkpoint": str(output_checkpoint),
        "output_sha256": sha256_file(output_checkpoint),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "validation_loss": val_loss,
        "validation_perplexity": val_perplexity,
        "sft_metadata": sft_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "instructions" / "train_instructions.jsonl")
    parser.add_argument("--val-file", type=Path, default=PROJECT_ROOT / "data" / "instructions" / "val_instructions.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe" / "tokenizer.json")
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "aarohan-wikitext2-005k.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints" / "aarohan-sft-001.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--context-length", type=int, default=128)
    args = parser.parse_args()

    result = train_sft(
        train_file=args.train_file,
        val_file=args.val_file,
        tokenizer_path=args.tokenizer,
        output_checkpoint=args.output,
        base_checkpoint_path=args.base_checkpoint if args.base_checkpoint.exists() else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        context_length=args.context_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
