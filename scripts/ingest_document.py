"""Ingest a small local TXT/Markdown document as unverified memory chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aarohan.memory import MemoryStore

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 100
ALLOWED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


class _VisibleTextParser(HTMLParser):
    _BLOCK_TAGS = frozenset({"address", "article", "br", "div", "footer", "h1", "h2", "h3", "h4", "li", "main", "p", "section", "tr"})
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._skip_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in self._SKIP_TAGS:
            self._skip_tag = tag
            self._skip_depth = 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def extract_text(path: Path, raw: bytes) -> str:
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() not in {".html", ".htm"}:
        return text
    parser = _VisibleTextParser()
    parser.feed(text)
    parser.close()
    return re.sub(r"[ \t]+", " ", " ".join(parser.parts)).strip()


def extract_segments(path: Path, raw: bytes) -> list[tuple[str, int | None]]:
    if path.suffix.lower() != ".pdf":
        text = extract_text(path, raw)
        return [(chunk, None) for chunk in chunk_text(text)]
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("Install PDF support with: python -m pip install -r requirements-data.txt") from error
    reader = PdfReader(path, strict=False)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")
    segments: list[tuple[str, int | None]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        segments.extend((chunk, page_number) for chunk in chunk_text(page_text))
    if not segments:
        raise ValueError("PDF contains no extractable text; scanned/image-only PDFs need OCR, which is not supported")
    return segments


def chunk_text(text: str, *, chunk_chars: int = 1000, overlap: int = 100) -> list[str]:
    if chunk_chars < 1 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk_chars must be positive and overlap must be in [0, chunk_chars)")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_chars, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end == len(paragraph):
                    break
                start = end - overlap
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > chunk_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest_document(
    path: Path,
    database: Path,
    *,
    chunk_chars: int = 1000,
    overlap: int = 100,
    privacy_classification: str = "non_sensitive",
) -> dict[str, object]:
    path = path.resolve(strict=True)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only .txt and Markdown files are supported; documents are stored as text and never executed")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document exceeds the {MAX_DOCUMENT_BYTES}-byte limit")
    if privacy_classification == "sensitive":
        raise ValueError("Sensitive documents are blocked until encrypted storage and access controls exist")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    segments = extract_segments(path, raw)
    chunks_with_pages = [
        (chunk, page_number)
        for text, page_number in segments
        for chunk in chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
    ]
    chunks = [chunk for chunk, _ in chunks_with_pages]
    if not chunks:
        raise ValueError("Document contains no text")

    store = MemoryStore(database)
    records = []
    for index, (chunk, page_number) in enumerate(chunks_with_pages, start=1):
        page_source = f"; page:{page_number}" if page_number is not None else ""
        source = f"document:{path.name}; sha256:{digest}{page_source}; chunk:{index}/{len(chunks)}"
        records.append(store.add(
            chunk,
            source=source,
            memory_type="semantic",
            verification_status="unverified",
            privacy_classification=privacy_classification,
        ))
    return {
        "document_name": path.name,
        "document_sha256": digest,
        "chunks_created": len(records),
        "memory_ids": [record["id"] for record in records],
        "verification_status": "unverified",
        "note": "Content was stored as plain text only; no instructions in the document were executed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "memory" / "aarohan.sqlite")
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--privacy", choices=("non_sensitive", "personal", "sensitive"), default="non_sensitive")
    args = parser.parse_args()
    try:
        result = ingest_document(args.path, args.database, chunk_chars=args.chunk_chars,
                                 overlap=args.overlap, privacy_classification=args.privacy)
    except (ValueError, OSError, UnicodeDecodeError, RuntimeError) as error:
        print(json.dumps({"ingested": False, "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

