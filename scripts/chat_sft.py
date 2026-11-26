"""Generate responses using an AAROHAN Supervised Fine-Tuned (SFT) checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.chat_template import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    validate_dialogue,
)
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file


def stream_chat_response(
    checkpoint_path: Path,
    tokenizer_path: Path,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 64,
    device_name: str = "cpu",
    temperature: float = 0.7,
    top_k: int | None = 20,
    top_p: float | None = 0.9,
) -> Iterator[dict[str, Any]]:
    """Yield token-by-token generation events using native AAROHAN model weights."""
    validate_dialogue(messages)
    checkpoint_path, tokenizer_path = checkpoint_path.resolve(), tokenizer_path.resolve()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_tok_hash = checkpoint.get("tokenizer_sha256")
    actual_tok_hash = sha256_file(tokenizer_path)
    if expected_tok_hash != actual_tok_hash:
        raise ValueError("Tokenizer hash mismatch with checkpoint")

    tokenizer = load_tokenizer(tokenizer_path)
    config = AAROHANConfig(**checkpoint["config"])
    model = AAROHANLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    prompt_str = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            prompt_str += f"{SYSTEM_TOKEN}\n{content}\n"
        elif role == "user":
            prompt_str += f"{USER_TOKEN}\n{content}\n"
        elif role == "assistant":
            prompt_str += f"{ASSISTANT_TOKEN}\n{content}{EOS_TOKEN}\n"

    prompt_str += f"{ASSISTANT_TOKEN}\n"
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_tokens: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[:, -config.context_length :]
            logits, _ = model(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = int(next_token[0, 0].item())

            if next_id == eos_id:
                break

            generated_tokens.append(next_id)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            token_text = tokenizer.decode([next_id], skip_special_tokens=True)
            yield {
                "type": "token",
                "token": token_text,
                "token_id": next_id,
                "generated_count": len(generated_tokens),
            }

    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    warning = (
        "This is an independently trained experimental SFT model checkpoint. "
        "Generated text may be incomplete or inaccurate and should not be used for critical decisions."
    )
    yield {
        "type": "done",
        "response": full_response,
        "generated_tokens": len(generated_tokens),
        "prompt_tokens": len(prompt_ids),
        "checkpoint": str(checkpoint_path),
        "warning": warning,
    }


def generate_chat_response(
    checkpoint_path: Path,
    tokenizer_path: Path,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 64,
    device_name: str = "cpu",
    temperature: float = 0.7,
    top_k: int | None = 20,
    top_p: float | None = 0.9,
) -> dict[str, Any]:
    validate_dialogue(messages)
    checkpoint_path, tokenizer_path = checkpoint_path.resolve(), tokenizer_path.resolve()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_tok_hash = checkpoint.get("tokenizer_sha256")
    actual_tok_hash = sha256_file(tokenizer_path)
    if expected_tok_hash != actual_tok_hash:
        raise ValueError("Tokenizer hash mismatch with checkpoint")

    tokenizer = load_tokenizer(tokenizer_path)
    config = AAROHANConfig(**checkpoint["config"])
    model = AAROHANLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    # Format conversation prompt up to the assistant turn
    prompt_str = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            prompt_str += f"{SYSTEM_TOKEN}\n{content}\n"
        elif role == "user":
            prompt_str += f"{USER_TOKEN}\n{content}\n"
        elif role == "assistant":
            prompt_str += f"{ASSISTANT_TOKEN}\n{content}{EOS_TOKEN}\n"

    # Add assistant prefix for model continuation
    prompt_str += f"{ASSISTANT_TOKEN}\n"

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Autoregressive generation with early stop on EOS_TOKEN
    generated_tokens: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[:, -config.context_length :]
            logits, _ = model(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = int(next_token[0, 0].item())

            if next_id == eos_id:
                break

            generated_tokens.append(next_id)
            token_ids = torch.cat((token_ids, next_token), dim=1)

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    warning = (
        "This is an independently trained experimental SFT model checkpoint. "
        "Generated text may be incomplete or inaccurate and should not be used for critical decisions."
    )

    return {
        "messages": messages,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_tokens),
        "response": response_text,
        "checkpoint": str(checkpoint_path),
        "training_note": checkpoint.get("training_note", "No training note recorded."),
        "warning": warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "aarohan-sft-001.pt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe" / "tokenizer.json")
    parser.add_argument("--query", default="What is an independent AI model?")
    parser.add_argument("--system", default="You are AAROHAN, an independent AI assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.query},
    ]

    try:
        result = generate_chat_response(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            device_name=args.device,
            temperature=args.temperature,
        )
    except Exception as err:
        print(json.dumps({"ok": False, "error": str(err)}, indent=2))
        raise SystemExit(1) from err

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
