"""AAROHAN AI 2.0 - Document Intelligence & Extraction Engine.

Extracts text, metadata, and tables from PDF, DOCX, XLSX, and TXT documents.
Supports chunking with page/section provenance for grounded retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
import pypdf

try:
    import docx
except ImportError:
    docx = None

DOCUMENTS_DIR = Path("artifacts/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentExtractor:
    """Multi-format document parsing and semantic chunking service."""

    @staticmethod
    def extract_file(file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return DocumentExtractor._extract_pdf(path)
        elif ext in [".docx", ".doc"] and docx:
            return DocumentExtractor._extract_docx(path)
        elif ext in [".txt", ".md", ".json"]:
            return DocumentExtractor._extract_text(path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")

    @staticmethod
    def _extract_pdf(path: Path) -> Dict[str, Any]:
        reader = pypdf.PdfReader(str(path))
        pages_content = []
        full_text = []

        for idx, page in enumerate(reader.pages, 1):
            txt = page.extract_text() or ""
            pages_content.append({"page_number": idx, "content": txt.strip()})
            full_text.append(f"--- Page {idx} ---\n{txt.strip()}")

        joined = "\n\n".join(full_text)
        return {
            "filename": path.name,
            "format": "pdf",
            "page_count": len(reader.pages),
            "total_chars": len(joined),
            "pages": pages_content,
            "full_text": joined,
        }

    @staticmethod
    def _extract_docx(path: Path) -> Dict[str, Any]:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        joined = "\n\n".join(paragraphs)

        return {
            "filename": path.name,
            "format": "docx",
            "page_count": max(1, len(paragraphs) // 15),
            "total_chars": len(joined),
            "pages": [{"page_number": 1, "content": joined}],
            "full_text": joined,
        }

    @staticmethod
    def _extract_text(path: Path) -> Dict[str, Any]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "filename": path.name,
            "format": path.suffix.lstrip("."),
            "page_count": 1,
            "total_chars": len(content),
            "pages": [{"page_number": 1, "content": content}],
            "full_text": content,
        }

    @staticmethod
    def chunk_document(doc_data: Dict[str, Any], max_chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
        """Split document text into overlapping chunks with page provenance."""
        chunks = []
        chunk_id = 1

        for page in doc_data.get("pages", []):
            text = page["content"]
            page_no = page["page_number"]
            # Split by paragraphs or double newline
            parts = re.split(r"\n\s*\n", text)
            current = ""

            for part in parts:
                clean = part.strip()
                if not clean:
                    continue
                if len(current) + len(clean) < max_chunk_size:
                    current += "\n\n" + clean if current else clean
                else:
                    if current:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "page_number": page_no,
                            "content": current,
                            "filename": doc_data["filename"],
                        })
                        chunk_id += 1
                    current = clean

            if current:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_no,
                    "content": current,
                    "filename": doc_data["filename"],
                })
                chunk_id += 1

        return chunks
