"""Train a small AAROHAN model on validated tokenized toy text."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import sha256_file
from aarohan.wikitext2 import validate_wikitext2
from scripts.validate_corpus import validate


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


def batchify(tokens: torch.Tensor, batch_size: int, context: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    usable = tokens.numel() - 1
    if usable < context:
        raise ValueError(f"Need at least {context + 1} tokens for context length {context}; got {tokens.numel()}")
    starts = torch.randint(0, usable - context + 1, (batch_size,))
    windows = torch.stack([tokens[start : start + context + 1] for start in starts.tolist()])
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def train(
    corpus_dir: Path,
    tokenizer_dir: Path,
    tokenized_dir: Path,
    output: Path,
    *,
    steps: int = 100,
    device_name: str = "cpu",
    seed: int = 1729,
    context: int = 32,
    batch_size: int = 8,
    resume_from: Path | None = None,
    layers: int = 2,
    heads: int = 2,
    width: int = 64,
) -> dict[str, object]:
    if steps < 1 or context < 1 or batch_size < 1 or layers < 1 or heads < 1 or width < 1:
        raise ValueError("steps, context, batch_size, layers, heads, and width must be positive")
    corpus_dir, tokenizer_dir, tokenized_dir = map(Path.resolve, (corpus_dir, tokenizer_dir, tokenized_dir))
    corpus_manifest_path = corpus_dir / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("dataset_id") == "Salesforce/wikitext":
        readiness = validate_wikitext2(corpus_dir)
    else:
        readiness = validate(corpus_dir)
    if not readiness["ready"]:
        raw_errors = readiness.get("errors")
        errors = [str(err) for err in raw_errors] if isinstance(raw_errors, list) else []
        raise ValueError("Corpus validation failed: " + "; ".join(errors))

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    metadata_path = tokenized_dir / "metadata.json"
    if not tokenizer_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Tokenizer or prepared-data metadata is missing; run the earlier pipeline steps first")
    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_meta.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("Prepared data was built with a different tokenizer; rerun prepare_tokenized_data.py")
    if dataset_meta.get("corpus_manifest_sha256") != sha256_file(corpus_manifest_path):
        raise ValueError("Prepared data was built from a different corpus manifest; rerun prepare_tokenized_data.py")
    manifest = corpus_manifest
    if dataset_meta.get("train_source_sha256") != sha256_file(corpus_dir / manifest["train_file"]):
        raise ValueError("Training source changed after tokenization; rerun prepare_tokenized_data.py")
    if dataset_meta.get("validation_source_sha256") != sha256_file(corpus_dir / manifest["validation_file"]):
        raise ValueError("Validation source changed after tokenization; rerun prepare_tokenized_data.py")
    if dataset_meta.get("train_tensor_sha256") != sha256_file(tokenized_dir / "train.pt"):
        raise ValueError("Training tensor hash does not match prepared-data metadata")
    if dataset_meta.get("validation_tensor_sha256") != sha256_file(tokenized_dir / "validation.pt"):
        raise ValueError("Validation tensor hash does not match prepared-data metadata")

    train_tokens = torch.load(tokenized_dir / "train.pt", map_location="cpu", weights_only=True)
    validation_tokens = torch.load(tokenized_dir / "validation.pt", map_location="cpu", weights_only=True)
    if train_tokens.ndim != 1 or validation_tokens.ndim != 1:
        raise ValueError("Expected one-dimensional token streams")
    if train_tokens.numel() != dataset_meta["train_tokens"] or validation_tokens.numel() != dataset_meta["validation_tokens"]:
        raise ValueError("Token tensor counts do not match prepared-data metadata")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = AAROHANConfig(
        vocab_size=int(dataset_meta["vocab_size"]), context_length=context,
        layers=layers, heads=heads, width=width,
    )
    model = AAROHANLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    completed_steps = 0
    if resume_from is not None:
        resume_path = resume_from.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        if previous.get("tokenizer_sha256") != dataset_meta["tokenizer_sha256"]:
            raise ValueError("Resume checkpoint tokenizer does not match prepared data")
        if previous.get("config") != model.config_dict():
            raise ValueError("Resume checkpoint model configuration does not match requested settings")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        completed_steps = int(previous["completed_steps"])
        if completed_steps >= steps:
            raise ValueError(f"Checkpoint already has {completed_steps} steps; target steps must be greater")
        random.setstate(previous["python_rng_state"])
        torch.set_rng_state(previous["torch_rng_state"])
        saved_cuda_rng = previous.get("cuda_rng_state_all")
        if device.type == "cuda" and saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)

    initial_loss: float | None = None
    final_loss = 0.0
    for step in range(completed_steps, steps):
        model.train()
        x, y = batchify(train_tokens, batch_size, context, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = final_loss

    model.eval()
    validation_loss_sum = 0.0
    validation_token_count = 0
    with torch.no_grad():
        # Score every next-token target, including the final shorter window.
        for start in range(0, validation_tokens.numel() - 1, context):
            window = validation_tokens[start : start + context + 1]
            if window.numel() < 2:
                continue
            _, loss = model(window[:-1].unsqueeze(0).to(device), window[1:].unsqueeze(0).to(device))
            assert loss is not None
            target_count = window.numel() - 1
            validation_loss_sum += float(loss.cpu()) * target_count
            validation_token_count += target_count
    if validation_token_count == 0:
        raise ValueError("Validation stream did not yield a complete window")
    validation_loss = validation_loss_sum / validation_token_count

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config_dict(),
        "completed_steps": steps,
        "seed": seed,
        "device": str(device),
        "tokenizer_sha256": dataset_meta["tokenizer_sha256"],
        "dataset_metadata": dataset_meta,
       "training_note": (
           "WikiText-2 local research run; results describe this dataset and configuration only."
           if dataset_meta.get("dataset_id") == "Salesforce/wikitext"
           else "Tiny AI-generated toy corpus integration experiment; not evidence of general language ability."
       ),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_save(checkpoint, output)

    reloaded = AAROHANLM(config).to(device)
    saved = torch.load(output, map_location=device, weights_only=False)
    reloaded.load_state_dict(saved["model_state"])
    sample_x, _ = batchify(train_tokens, 1, context, device)
    with torch.no_grad():
        original_logits, _ = model(sample_x)
        restored_logits, _ = reloaded(sample_x)
    reload_difference = float((original_logits - restored_logits).abs().max().cpu())

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": model.parameter_count(),
        "steps": steps,
        "context_length": context,
        "batch_size": batch_size,
        "train_tokens": train_tokens.numel(),
        "validation_tokens": validation_tokens.numel(),
        "validation_tokens_scored": validation_token_count,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "validation_loss": validation_loss,
        "checkpoint": str(output.resolve()),
        "checkpoint_reload_max_logit_difference": reload_difference,
       "note": (
           "WikiText-2 next-token training experiment; this small run does not demonstrate general language ability."
           if dataset_meta.get("dataset_id") == "Salesforce/wikitext"
           else "Toy AI-generated data validates the training pipeline only; metrics do not demonstrate useful language ability."
       ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpora" / "original")
    parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0")
    parser.add_argument("--tokenized-dir", type=Path, default=PROJECT_ROOT / "data" / "tokenized" / "original-toy-v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints" / "toy-corpus-smoke.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--resume-from", type=Path, help="Resume from a previous checkpoint; --steps is the new total step target")
    args = parser.parse_args()
    try:
        result = train(args.corpus_dir, args.tokenizer_dir, args.tokenized_dir, args.output,
                       steps=args.steps, device_name=args.device, seed=args.seed,
                       context=args.context, batch_size=args.batch_size,
                       resume_from=args.resume_from, layers=args.layers,
                       heads=args.heads, width=args.width)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

