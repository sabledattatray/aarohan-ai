"""Benchmark AAROHAN training-step throughput and peak device memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.model import AAROHANConfig, AAROHANLM


BENCHMARK_CONFIGS = {
    "tiny_pipeline": AAROHANConfig(vocab_size=265, context_length=32, layers=2, heads=2, width=64),
    "aarohan_0_01_candidate": AAROHANConfig(vocab_size=4096, context_length=256, layers=4, heads=6, width=288),
}


def benchmark_one(config: AAROHANConfig, *, device_name: str = "cuda", batch_size: int = 2,
                  warmup_steps: int = 2, measured_steps: int = 10, seed: int = 1729) -> dict:
    if batch_size < 1 or warmup_steps < 0 or measured_steps < 1:
        raise ValueError("batch_size/measured_steps must be positive and warmup_steps nonnegative")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = AAROHANLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)

    def step() -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Benchmark produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach().cpu())

    try:
        for _ in range(warmup_steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        last_loss = 0.0
        for _ in range(measured_steps):
            last_loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        tokens = batch_size * config.context_length * measured_steps
        result = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "config": config.__dict__,
            "parameters": model.parameter_count(),
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": measured_steps / elapsed,
            "training_tokens_per_second": tokens / elapsed,
            "last_loss": last_loss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "status": "completed",
        }
        return result
    except torch.cuda.OutOfMemoryError as error:
        return {
            "device": str(device),
            "config": config.__dict__,
            "parameters": model.parameter_count(),
            "status": "out_of_memory",
            "error": str(error).splitlines()[0],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    finally:
        del model, optimizer, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_benchmarks(*, device_name: str = "cuda", batch_size: int = 2,
                   warmup_steps: int = 2, measured_steps: int = 10) -> dict:
    results = {
        name: benchmark_one(config, device_name=device_name, batch_size=batch_size,
                            warmup_steps=warmup_steps, measured_steps=measured_steps)
        for name, config in BENCHMARK_CONFIGS.items()
    }
    return {
        "benchmark_time_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
        "note": "Random-token training-step benchmark only; it measures hardware throughput and memory, not model quality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "experiments" / "hardware-benchmark.json")
    args = parser.parse_args()
    report = run_benchmarks(device_name=args.device, batch_size=args.batch_size,
                            warmup_steps=args.warmup_steps, measured_steps=args.steps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

