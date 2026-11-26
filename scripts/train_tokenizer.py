"""Train AAROHAN's BPE tokenizer on a validated local corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.tokenizer import train_bpe
from scripts.validate_corpus import validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpora" / "original")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tokenizers" / "aarohan-bpe-0.1.0")
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args()

    corpus_dir = args.corpus_dir.resolve()
    readiness = validate(corpus_dir)
    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2))
        raise SystemExit("Corpus validation failed; tokenizer training was not started.")
    metadata = train_bpe(
        corpus_dir / "corpus.txt",
        args.output_dir,
        vocab_size=args.vocab_size,
        provenance_path=corpus_dir / "manifest.json",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

