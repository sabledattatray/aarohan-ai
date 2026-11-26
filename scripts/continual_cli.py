"""CLI for continual learning cycles, regression gate evaluation, and rollback management."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.continual import (
    RegressionGate,
    RollbackManager,
    evaluate_benchmark_loss,
    train_continual_candidate,
)
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer

DEFAULT_REGISTRY = PROJECT_ROOT / "checkpoints" / "deployment_registry.json"


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-file", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)

    # 1. Train Candidate
    t_cand = commands.add_parser("train-candidate", help="Train a continual learning candidate with replay")
    t_cand.add_argument("--base-checkpoint", type=Path, required=True)
    t_cand.add_argument("--tokenizer", type=Path, required=True)
    t_cand.add_argument("--new-data", type=Path, required=True)
    t_cand.add_argument("--replay-data", type=Path, required=True)
    t_cand.add_argument("--output", type=Path, required=True)
    t_cand.add_argument("--epochs", type=int, default=10)
    t_cand.add_argument("--replay-ratio", type=float, default=0.5)
    t_cand.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    # 2. Evaluate Regression Gate
    eval_gate = commands.add_parser("eval-gate", help="Evaluate candidate against baseline on historical benchmark")
    eval_gate.add_argument("--base-checkpoint", type=Path, required=True)
    eval_gate.add_argument("--candidate-checkpoint", type=Path, required=True)
    eval_gate.add_argument("--tokenizer", type=Path, required=True)
    eval_gate.add_argument("--benchmark-file", type=Path, required=True)
    eval_gate.add_argument("--max-regression", type=float, default=8.0)
    eval_gate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    # 3. Deploy
    deploy = commands.add_parser("deploy", help="Deploy or promote a certified model checkpoint")
    deploy.add_argument("--checkpoint", type=Path, required=True)
    deploy.add_argument("--note", default="Promoted model")

    # 4. Rollback
    rb = commands.add_parser("rollback", help="Rollback active model to the previous stable checkpoint")
    rb.add_argument("--reason", default="Regression threshold exceeded")

    # 5. Status
    commands.add_parser("status", help="Get current active model status")

    args = parser.parse_args(argv)
    mgr = RollbackManager(args.registry_file)

    if args.command == "train-candidate":
        new_records = []
        with args.new_data.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    new_records.append(json.loads(line))
        return train_continual_candidate(
            base_checkpoint_path=args.base_checkpoint,
            tokenizer_path=args.tokenizer,
            new_records=new_records,
            replay_records_file=args.replay_data,
            output_checkpoint_path=args.output,
            epochs=args.epochs,
            replay_ratio=args.replay_ratio,
            device_name=args.device,
        )

    if args.command == "eval-gate":
        tok = load_tokenizer(args.tokenizer)
        bench = []
        with args.benchmark_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bench.append(json.loads(line))

        dev = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

        # Evaluate base model
        base_data = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        m_base = AAROHANLM(AAROHANConfig(**base_data["config"])).to(dev)
        m_base.load_state_dict(base_data["model_state"])
        base_loss, base_ppl = evaluate_benchmark_loss(m_base, tok, bench, device=dev)

        # Evaluate candidate model
        cand_data = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=False)
        m_cand = AAROHANLM(AAROHANConfig(**cand_data["config"])).to(dev)
        m_cand.load_state_dict(cand_data["model_state"])
        cand_loss, cand_ppl = evaluate_benchmark_loss(m_cand, tok, bench, device=dev)

        gate = RegressionGate(max_allowed_regression_percent=args.max_regression)
        eval_result = gate.evaluate(base_loss, cand_loss)
        eval_result["baseline_perplexity"] = round(base_ppl, 2)
        eval_result["candidate_perplexity"] = round(cand_ppl, 2)
        return eval_result

    if args.command == "deploy":
        return mgr.deploy_model(args.checkpoint, note=args.note)

    if args.command == "rollback":
        return mgr.rollback(reason=args.reason)

    if args.command == "status":
        return {"active_checkpoint": mgr.get_active()}

    return {"error": "Unhandled command"}


def main() -> None:
    try:
        result = run()
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
