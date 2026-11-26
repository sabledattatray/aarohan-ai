"""Use AAROHAN's local persistent memory from PowerShell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.memory import MemoryStore


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "memory" / "aarohan.sqlite"


def run(argv: Sequence[str] | None = None, *, database_path: Path | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=database_path or DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="Store a memory (unverified unless a reviewer is supplied)")
    add.add_argument("--content", required=True)
    add.add_argument("--source")
    add.add_argument("--type", default="episodic", choices=("episodic", "semantic", "procedural", "learning"))
    add.add_argument("--verified-by", help="Only use after a person has checked the source")
    add.add_argument("--privacy", default="non_sensitive", choices=("non_sensitive", "personal", "sensitive"))

    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--verified-only", action="store_true")
    search.add_argument("--source-contains", help="Restrict results to a source filename or source label")

    get = commands.add_parser("get")
    get.add_argument("memory_id")

    correct = commands.add_parser("correct")
    correct.add_argument("memory_id")
    correct.add_argument("--content", required=True)
    correct.add_argument("--source", required=True)
    correct.add_argument("--corrected-by", required=True)

    delete = commands.add_parser("delete")
    delete.add_argument("memory_id")

    args = parser.parse_args(argv)
    store = MemoryStore(args.database)
    if args.command == "add":
        verified = bool(args.verified_by)
        return store.add(args.content, source=args.source, memory_type=args.type,
                         verification_status="verified" if verified else "unverified",
                         verified_by=args.verified_by, privacy_classification=args.privacy)
    if args.command == "search":
        return {"results": store.search(args.query, limit=args.limit, verified_only=args.verified_only,
                                         source_contains=args.source_contains)}
    if args.command == "get":
        return {"memory": store.get(args.memory_id)}
    if args.command == "correct":
        return store.correct(args.memory_id, args.content, source=args.source, corrected_by=args.corrected_by)
    return {"deleted": store.delete(args.memory_id), "memory_id": args.memory_id}


def main() -> None:
    try:
        result = run()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

