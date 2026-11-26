"""Generate a local continuation from an AAROHAN checkpoint and tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.model import AAROHANConfig, AAROHANLM
from aarohan.tokenizer import load_tokenizer, sha256_file


def generate(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompt: str,
    *,
    max_new_tokens: int = 40,
    device_name: str = "cpu",
    temperature: float = 0.8,
    top_k: int | None = 20,
    top_p: float | None = None,
) -> dict[str, object]:
    if not prompt.strip():
        raise ValueError("Prompt must contain non-whitespace text")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    checkpoint_path, tokenizer_path = checkpoint_path.resolve(), tokenizer_path.resolve()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_hash = checkpoint.get("tokenizer_sha256")
    actual_hash = sha256_file(tokenizer_path)
    if expected_hash != actual_hash:
        raise ValueError("Tokenizer hash does not match the checkpoint; use the tokenizer trained with this model")
    tokenizer = load_tokenizer(tokenizer_path)
    config = AAROHANConfig(**checkpoint["config"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration")

    model = AAROHANLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs")
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(token_ids, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_k=top_k, top_p=top_p)
    generated_ids = generated[0].tolist()
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    dataset_metadata = checkpoint.get("dataset_metadata", {})
    dataset_id = dataset_metadata.get("dataset_id")
    if dataset_id == "Salesforce/wikitext":
        warning = "This small checkpoint was trained on WikiText-2; its continuation may be incoherent or inaccurate and is not a reliable answer."
    elif "toy" in str(checkpoint.get("training_note", "")).lower():
        warning = "This checkpoint was trained on a tiny AI-generated toy corpus; output is a pipeline demonstration, not a reliable answer."
    else:
        warning = "This is a small experimental checkpoint; generated text may be inaccurate and is not a reliable answer."
    return {
        "prompt": prompt,
        "generated_text": full_text,
        "generated_token_count": len(generated_ids) - len(prompt_ids),
        "checkpoint": str(checkpoint_path),
        "tokenizer_sha256": actual_hash,
        "training_note": checkpoint.get("training_note", "No training note recorded."),
       "warning": warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "toy-corpus-smoke.pt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0" / "tokenizer.json")
    parser.add_argument("--prompt", default="A small robot")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float)
    args = parser.parse_args()
    try:
        result = generate(args.checkpoint, args.tokenizer, args.prompt,
                          max_new_tokens=args.max_new_tokens, device_name=args.device,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

