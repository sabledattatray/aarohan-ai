"""Verify and summarize the downloaded WikiText-2 Parquet splits.

This is a read-only check. It does not approve privacy or licensing, and it
never prints corpus text. The test split is only inspected, never used for fit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "wikitext2-raw-v1"
MANIFEST_PATH = DATA_DIR / "download-manifest.json"
SPLITS = ("train", "validation", "test")
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(directory: Path = DATA_DIR) -> dict:
    manifest_path = directory / "download-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Download manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install requirements-data.txt") from exc

    results = {}
    for split in SPLITS:
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest has no '{split}' split entry")
        path = directory / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {split}: expected {entry.get('sha256')}, got {actual_hash}")

        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
        lengths = [len(text) for text in texts if isinstance(text, str)]
        results[split] = {
            "rows": len(texts),
            "manifest_rows_match": len(texts) == entry.get("rows"),
            "sha256_verified": True,
            "blank_rows": sum(not isinstance(text, str) or not text.strip() for text in texts),
            "duplicate_rows": len(texts) - len(set(texts)),
            "characters_min": min(lengths, default=0),
            "characters_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "characters_max": max(lengths, default=0),
            "email_like_rows": sum(bool(EMAIL_RE.search(text)) for text in texts if isinstance(text, str)),
        }
        if not results[split]["manifest_rows_match"]:
            raise ValueError(f"Row count mismatch for {split}: manifest says {entry.get('rows')}, found {len(texts)}")

    return {
        "dataset_id": manifest.get("dataset_id"),
        "revision": manifest.get("revision"),
        "splits": results,
        "test_split_used_for_training": False,
        "privacy_review": "Not established by this script. Email-like counts are only a simple signal; review the dataset card and sample rows yourself before training.",
        "note": "Read-only summary. No text excerpts are printed. Fit tokenizer/model on train only; use validation for evaluation and keep test held out.",
    }


def main() -> None:
    try:
        print(json.dumps(inspect(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
