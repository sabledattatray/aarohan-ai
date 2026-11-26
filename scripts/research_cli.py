"""CLI for autonomous research planner, learning tasks, and knowledge-gap detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.memory import MemoryStore
from aarohan.research import BoundedResearchEngine, KnowledgeGapDetector, LearningQueue

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "memory" / "aarohan.sqlite"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "research"


def run(argv: list[str] | None = None, *, database_path: Path | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=database_path or DEFAULT_DATABASE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    commands = parser.add_subparsers(dest="command", required=True)

    # 1. Task Create
    t_create = commands.add_parser("task-create", help="Create a learning task in the queue")
    t_create.add_argument("--title", required=True)
    t_create.add_argument("--query", required=True)
    t_create.add_argument("--gap-type", default="manual_learning_goal")
    t_create.add_argument("--priority", type=float, default=0.5)
    t_create.add_argument("--objective", default="")
    t_create.add_argument("--max-docs", type=int, default=5)

    # 2. Task List
    t_list = commands.add_parser("task-list", help="List learning tasks")
    t_list.add_argument("--status", choices=("pending", "approved", "in_progress", "completed", "rejected", "failed"))

    # 3. Gap Detection
    commands.add_parser("gap-detect", help="Detect missing relationships and propose learning tasks")

    # 4. Research Run
    r_run = commands.add_parser("research-run", help="Execute bounded research for a task")
    r_run.add_argument("--task-id", required=True)
    r_run.add_argument("--docs", nargs="+", type=Path, required=True, help="Approved source documents")
    r_run.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)

    args = parser.parse_args(argv)
    store = MemoryStore(args.database)
    queue = LearningQueue(store)

    if args.command == "task-create":
        return queue.create_task(
            args.title,
            args.query,
            gap_type=args.gap_type,
            priority=args.priority,
            objective=args.objective,
            max_budget_docs=args.max_docs,
        )

    if args.command == "task-list":
        return {"tasks": queue.list_tasks(status=args.status)}

    if args.command == "gap-detect":
        detector = KnowledgeGapDetector(store, queue)
        gaps = detector.detect_missing_graph_relationships()
        return {"detected_gaps": len(gaps), "proposed_tasks": gaps}

    if args.command == "research-run":
        engine = BoundedResearchEngine(store, queue, artifacts_dir=args.artifacts_dir)
        return engine.execute_task(args.task_id, args.docs)

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
