"""Evaluate and compare AAROHAN checkpoints on the same held-out token stream."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.model import AAROHANConfig, AAROHANLM
from aarohan.tokenizer import sha256_file


def evaluate_checkpoint(checkpoint_path: Path, tokenized_dir: Path, tokenizer_path: Path, device: torch.device, split: str = "validation") -> dict:
    checkpoint_path, tokenized_dir, tokenizer_path = map(Path.resolve, (checkpoint_path, tokenized_dir, tokenizer_path))
    metadata = json.loads((tokenized_dir / "metadata.json").read_text(encoding="utf-8"))
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if split == "test" and metadata.get("test_split_used_for_training") is not False:
        raise ValueError("Test evaluation metadata must confirm test split was not used for training")
    if metadata.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("Evaluation tokenizer does not match prepared data")
    tensor_key = f"{split}_tensor_sha256"
    source_key = f"{split}_source_sha256"
    tensor_path = tokenized_dir / f"{split}.pt"
    if metadata.get(tensor_key) != sha256_file(tensor_path):
        raise ValueError(f"{split.capitalize()} tensor hash does not match prepared-data metadata")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("tokenizer_sha256") != metadata["tokenizer_sha256"]:
        raise ValueError(f"Checkpoint tokenizer mismatch: {checkpoint_path}")
    checkpoint_data = checkpoint.get("dataset_metadata", {})
    if split == "validation" and checkpoint_data.get("validation_source_sha256") != metadata.get(source_key):
        raise ValueError(f"Checkpoint validation split differs from evaluator data: {checkpoint_path}")
    if split == "test" and checkpoint_data.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError(f"Checkpoint dataset differs from test data: {checkpoint_path}")

    config = AAROHANConfig(**checkpoint["config"])
    model = AAROHANLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = torch.load(tensor_path, map_location="cpu", weights_only=True)
    total_loss = 0.0
    target_count = 0
    with torch.inference_mode():
        for start in range(0, tokens.numel() - 1, config.context_length):
            window = tokens[start : start + config.context_length + 1]
            if window.numel() < 2:
                continue
            inputs = window[:-1].unsqueeze(0).to(device)
            targets = window[1:].unsqueeze(0).to(device)
            logits, _ = model(inputs)
            summed = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1), reduction="sum")
            total_loss += float(summed.cpu())
            target_count += targets.numel()
    if target_count == 0:
        raise ValueError(f"{split.capitalize()} stream has no next-token targets")
    mean_loss = total_loss / target_count
    return {
        "checkpoint": str(checkpoint_path),
        "completed_steps": checkpoint.get("completed_steps"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        f"{split}_tokens_scored": target_count,
        f"{split}_loss": mean_loss,
        f"{split}_perplexity": math.exp(mean_loss),
        f"{split}_source_sha256": metadata[source_key],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "training_note": checkpoint.get("training_note"),
    }


def evaluate_many(checkpoints: list[Path], tokenized_dir: Path, tokenizer_path: Path, device_name: str, split: str = "validation") -> dict:
    if len(checkpoints) < 1:
        raise ValueError("Provide at least one checkpoint")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    results = [evaluate_checkpoint(path, tokenized_dir, tokenizer_path, device, split) for path in checkpoints]
    baseline = results[0]
    loss_key = f"{split}_loss"
    for result in results:
        result[f"{split}_loss_delta_vs_first"] = result[loss_key] - baseline[loss_key]
    return {
        "device": str(device),
        "split": split,
        "results": results,
        "note": f"Loss and perplexity are measured on the supplied {split} split; they do not measure general capability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Checkpoint to evaluate; repeat to compare on the same validation split. First is baseline.")
    parser.add_argument("--tokenized-dir", type=Path, default=PROJECT_ROOT / "data" / "tokenized" / "original-toy-v1")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0" / "tokenizer.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Use test only once after model selection; never use it for training or tuning")
    parser.add_argument("--output-json", type=Path, help="Optional path to save the exact evaluation report")
    args = parser.parse_args()
    try:
        result = evaluate_many(args.checkpoint, args.tokenized_dir, args.tokenizer, args.device, args.split)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    if args.output_json:
        output = args.output_json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                             prefix=".evaluation-", suffix=".tmp", delete=False) as handle:
                temporary_name = handle.name
                json.dump(result, handle, indent=2)
                handle.write("\n")
            os.replace(temporary_name, output)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

