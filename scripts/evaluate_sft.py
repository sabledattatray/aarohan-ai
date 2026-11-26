"""Evaluate SFT model against held-out instructions, base checkpoints, and safety prompts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.chat_template import prepare_dialogue_tokens_with_mask
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file
from scripts.chat_sft import generate_chat_response
from scripts.validate_instruction_dataset import validate_instruction_file


def evaluate_sft_model(
    checkpoint_path: Path,
    tokenizer_path: Path,
    test_instruction_path: Path,
    *,
    device_name: str = "cpu",
    output_report: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path, tokenizer_path = checkpoint_path.resolve(), tokenizer_path.resolve()
    test_instruction_path = test_instruction_path.resolve()

    dataset_report = validate_instruction_file(test_instruction_path)
    tokenizer = load_tokenizer(tokenizer_path)
    tok_sha256 = sha256_file(tokenizer_path)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("tokenizer_sha256") != tok_sha256:
        raise ValueError("Tokenizer hash mismatch with checkpoint")

    config = AAROHANConfig(**checkpoint["config"])
    model = AAROHANLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_records = []
    with test_instruction_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    # Evaluate loss and perplexity on assistant targets
    total_loss = 0.0
    valid_batches = 0
    per_item_results = []

    with torch.no_grad():
        for item in test_records:
            tokens, labels, _ = prepare_dialogue_tokens_with_mask(
                tokenizer, item["messages"], context_length=config.context_length
            )
            x = tokens[:-1].unsqueeze(0).to(device)
            y = labels[1:].unsqueeze(0).to(device)

            _, loss = model(x, y)
            loss_val = float(loss.detach().cpu()) if loss is not None else float("nan")
            if not math.isnan(loss_val):
                total_loss += loss_val
                valid_batches += 1

            # Generate sample continuation from conversation prefix
            prefix_messages = [m for m in item["messages"] if m["role"] != "assistant"]
            if prefix_messages:
                gen_result = generate_chat_response(
                    checkpoint_path=checkpoint_path,
                    tokenizer_path=tokenizer_path,
                    messages=prefix_messages,
                    max_new_tokens=32,
                    device_name=device_name,
                )
                generated_text = gen_result["response"]
            else:
                generated_text = ""

            per_item_results.append({
                "id": item["id"],
                "category": item.get("category", "unspecified"),
                "target_loss": loss_val,
                "target_perplexity": math.exp(loss_val) if loss_val < 20 else float("inf"),
                "generated_response": generated_text,
            })

    avg_loss = total_loss / max(valid_batches, 1)
    avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tokenizer_sha256": tok_sha256,
        "test_dataset_sha256": dataset_report["sha256"],
        "record_count": len(test_records),
        "average_target_loss": avg_loss,
        "average_target_perplexity": avg_ppl,
        "results": per_item_results,
    }

    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "aarohan-sft-001.pt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe" / "tokenizer.json")
    parser.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "data" / "instructions" / "test_instructions.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "sft-evaluation.json")
    args = parser.parse_args()

    report = evaluate_sft_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        test_instruction_path=args.test_file,
        device_name=args.device,
        output_report=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
