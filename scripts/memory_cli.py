"""Use AAROHAN's local persistent 6-layer memory and Knowledge Graph from CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.memory import VALID_MEMORY_TYPES, VALID_RELATION_TYPES, MemoryStore

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "memory" / "aarohan.sqlite"


def run(argv: Sequence[str] | None = None, *, database_path: Path | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=database_path or DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)

    # 1. Add Memory
    add = commands.add_parser("add", help="Store a memory record")
    add.add_argument("--content", required=True)
    add.add_argument("--source")
    add.add_argument("--type", default="episodic", choices=tuple(sorted(VALID_MEMORY_TYPES)))
    add.add_argument("--verified-by", help="Only use after a person has checked the source")
    add.add_argument("--privacy", default="non_sensitive", choices=("non_sensitive", "personal", "sensitive"))
    add.add_argument("--importance", type=float, default=0.5)

    # 2. Search
    search = commands.add_parser("search", help="Search memory via keyword, vector, or hybrid retrieval")
    search.add_argument("query")
    search.add_argument("--mode", default="keyword", choices=("keyword", "vector", "hybrid"))
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--verified-only", action="store_true")
    search.add_argument("--type", choices=tuple(sorted(VALID_MEMORY_TYPES)))
    search.add_argument("--source-contains", help="Restrict results to a source filename or label")

    # 3. Get
    get = commands.add_parser("get")
    get.add_argument("memory_id")

    # 4. Correct
    correct = commands.add_parser("correct")
    correct.add_argument("memory_id")
    correct.add_argument("--content", required=True)
    correct.add_argument("--source", required=True)
    correct.add_argument("--corrected-by", required=True)

    # 5. Delete
    delete = commands.add_parser("delete")
    delete.add_argument("memory_id")

    # 6. Knowledge Graph: Entity Add
    entity_add = commands.add_parser("entity-add", help="Add an entity to the Knowledge Graph (Layer 5)")
    entity_add.add_argument("name")
    entity_add.add_argument("--type", default="concept")
    entity_add.add_argument("--description", default="")

    # 7. Knowledge Graph: Entity List
    entity_list = commands.add_parser("entity-list", help="List entities in the Knowledge Graph")
    entity_list.add_argument("--type")

    # 8. Knowledge Graph: Relation Add
    relation_add = commands.add_parser("relation-add", help="Create a relationship between two entities")
    relation_add.add_argument("source")
    relation_add.add_argument("target")
    relation_add.add_argument("--relation", required=True, choices=tuple(sorted(VALID_RELATION_TYPES)))
    relation_add.add_argument("--confidence", type=float, default=1.0)
    relation_add.add_argument("--source-doc")

    # 9. Knowledge Graph: Graph Query
    graph_query = commands.add_parser("graph-query", help="Explore relations for an entity")
    graph_query.add_argument("entity")
    graph_query.add_argument("--direction", default="both", choices=("out", "in", "both"))

    # 10. Knowledge Graph: Path Finder
    graph_path = commands.add_parser("graph-path", help="Find relational path between two entities")
    graph_path.add_argument("start")
    graph_path.add_argument("end")
    graph_path.add_argument("--max-depth", type=int, default=3)

    # 11. Consolidate Memory
    consolidate = commands.add_parser("consolidate", help="Run memory consolidation across recent episodic records")
    consolidate.add_argument("--recent-limit", type=int, default=50)

    # 12. Check Contradictions
    check_contra = commands.add_parser("check-contradiction", help="Check proposed content against existing verified memory")
    check_contra.add_argument("content")

    args = parser.parse_args(argv)
    store = MemoryStore(args.database)

    if args.command == "add":
        verified = bool(args.verified_by)
        return store.add(
            args.content,
            source=args.source,
            memory_type=args.type,
            verification_status="verified" if verified else "unverified",
            verified_by=args.verified_by,
            privacy_classification=args.privacy,
            importance=args.importance,
        )

    if args.command == "search":
        if args.mode == "vector":
            results = store.vector_search(args.query, limit=args.limit, memory_type=args.type, verified_only=args.verified_only)
        elif args.mode == "hybrid":
            results = store.hybrid_search(args.query, limit=args.limit, memory_type=args.type, verified_only=args.verified_only)
        else:
            results = store.search(
                args.query,
                limit=args.limit,
                memory_type=args.type,
                verified_only=args.verified_only,
                source_contains=args.source_contains,
            )
        return {"mode": args.mode, "query": args.query, "results": results}

    if args.command == "get":
        return {"memory": store.get(args.memory_id)}

    if args.command == "correct":
        return store.correct(args.memory_id, args.content, source=args.source, corrected_by=args.corrected_by)

    if args.command == "delete":
        return {"deleted": store.delete(args.memory_id), "memory_id": args.memory_id}

    if args.command == "entity-add":
        return store.add_entity(args.name, entity_type=args.type, description=args.description)

    if args.command == "entity-list":
        return {"entities": store.list_entities(entity_type=args.type)}

    if args.command == "relation-add":
        return store.add_relation(
            args.source,
            args.target,
            relation_type=args.relation,
            confidence=args.confidence,
            source=args.source_doc,
        )

    if args.command == "graph-query":
        return {"entity": args.entity, "relations": store.get_relations(args.entity, direction=args.direction)}

    if args.command == "graph-path":
        return {"start": args.start, "end": args.end, "paths": store.find_path(args.start, args.end, max_depth=args.max_depth)}

    if args.command == "consolidate":
        return store.consolidate(recent_limit=args.recent_limit)

    if args.command == "check-contradiction":
        return {"content": args.content, "contradictions": store.detect_contradictions(args.content)}

    return {"error": "Unhandled command"}


def main() -> None:
    try:
        result = run()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
