"""Register evaluated checkpoints and explicitly activate or roll them back."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.registry import activate_candidate, register_candidate, rollback_active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "checkpoints" / "registry.json")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("register")
    add.add_argument("--checkpoint", type=Path, required=True)
    add.add_argument("--evaluation", type=Path, required=True)
    activate = commands.add_parser("activate", help="Explicitly set the local active checkpoint pointer")
    activate.add_argument("--id", required=True)
    commands.add_parser("rollback", help="Restore the previous local active checkpoint pointer")
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = register_candidate(args.checkpoint, args.evaluation, args.registry)
        elif args.command == "activate":
            result = activate_candidate(args.id, args.registry)
        else:
            result = rollback_active(args.registry)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

