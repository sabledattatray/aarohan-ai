"""Validate instruction dataset format, role ordering, content, and report checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.chat_template import validate_dialogue


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_instruction_file(file_path: Path) -> dict[str, object]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Instruction file not found: {file_path}")

    records = []
    categories: dict[str, int] = {}
    total_messages = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_num} in {file_path}: {err}") from err

            if "id" not in record or not isinstance(record["id"], str):
                raise ValueError(f"Missing or invalid 'id' at line {line_num}")
            if "messages" not in record or not isinstance(record["messages"], list):
                raise ValueError(f"Missing or invalid 'messages' list at line {line_num}")

            validate_dialogue(record["messages"])
            category = record.get("category", "unspecified")
            categories[category] = categories.get(category, 0) + 1
            total_messages += len(record["messages"])
            records.append(record)

    if not records:
        raise ValueError(f"Instruction file {file_path} contains no valid records")

    return {
        "file": str(file_path.resolve()),
        "record_count": len(records),
        "total_messages": total_messages,
        "categories": categories,
        "sha256": sha256_file(file_path),
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Paths to instruction jsonl files")
    args = parser.parse_args()

    results = []
    for p in args.paths:
        res = validate_instruction_file(p)
        results.append(res)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
