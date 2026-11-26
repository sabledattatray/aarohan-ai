"""Experimental local query flow showing memory evidence separately from model text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.memory import MemoryStore
from scripts.generate_local import generate


def run_query(
    query: str,
    checkpoint: Path,
    tokenizer: Path,
    memory_database: Path,
    *,
    memory_limit: int = 3,
    verified_only: bool = False,
    device: str = "cpu",
    max_new_tokens: int = 40,
    evidence_only: bool = False,
    source_contains: str | None = None,
) -> dict[str, object]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    memories = MemoryStore(memory_database).search(
        query, limit=memory_limit, verified_only=verified_only, source_contains=source_contains
    )
    model_result = None if evidence_only else generate(
        checkpoint, tokenizer, query, max_new_tokens=max_new_tokens, device_name=device
    )
    return {
        "query": query,
        "answer_mode": "retrieved_evidence_only" if evidence_only else "ungrounded_model_continuation",
        "source_filter": source_contains,
        "retrieved_memory": [
            {
                "content": record["content"],
                "source": record["source"],
                "verification_status": record["verification_status"],
                "confidence": record["confidence"],
            }
            for record in memories
        ],
        "model_output": None if model_result is None else {
            "text": model_result["generated_text"],
            "checkpoint": model_result["checkpoint"],
            "training_note": model_result["training_note"],
            "warning": model_result["warning"],
        },
        "note": (
            "Only retrieved memory is shown. Check each source and verification status; no answer was generated."
            if evidence_only
            else "Memory is shown as retrieved evidence; it is not inserted into the model prompt and is not presented as verified unless marked verified. Model text is an ungrounded continuation, not an answer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "toy-corpus-smoke.pt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0" / "tokenizer.json")
    parser.add_argument("--memory-database", type=Path, default=PROJECT_ROOT / "data" / "memory" / "aarohan.sqlite")
    parser.add_argument("--memory-limit", type=int, default=3)
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--evidence-only", action="store_true",
                        help="Show retrieved memory without generating a model continuation")
    parser.add_argument("--source-contains", help="Only retrieve memory whose source contains this text")
    args = parser.parse_args()
    try:
        result = run_query(args.query, args.checkpoint, args.tokenizer, args.memory_database,
                           memory_limit=args.memory_limit, verified_only=args.verified_only,
                          device=args.device, max_new_tokens=args.max_new_tokens,
                          evidence_only=args.evidence_only, source_contains=args.source_contains)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

