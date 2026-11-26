"""
AAROHAN AI — Retrieval-Augmented Generation (RAG) Engine.

Provides:
- Ingestion of multi-format documents (PDF, DOCX, TXT, MD) with source provenance.
- Semantic chunk management with deduplication via SHA-256.
- Hybrid search (dense vector + FTS5 BM25) with configurable alpha ranking.
- Indirect prompt-injection defense for retrieved contexts.
- Context relevance scoring and outdated/superseded record filtering.
- Contradiction detection between retrieved documents.
- Context assembly for language model prompt augmentation.
- Quantitative retrieval evaluation metrics (Recall@K, Precision@K, MRR).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from aarohan.documents.extractor import DocumentExtractor
from aarohan.memory import MemoryStore

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = re.compile(
    r"(?i)\b("
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"you\s+are\s+now|"
    r"system\s+prompt:|"
    r"disregard\s+(?:all\s+)?prior|"
    r"override\s+(?:system|safety)|"
    r"developer\s+mode\s+enabled"
    r")\b"
)


class RAGPipeline:
    """End-to-end grounded document retrieval and context augmentation engine."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.store = memory_store

    @staticmethod
    def sanitize_chunk(text: str) -> tuple[str, bool]:
        """
        Detects and neutralizes indirect prompt-injection instructions embedded in untrusted retrieved documents.
        Returns the sanitized text and a boolean flag indicating if injection was detected.
        """
        if not INJECTION_PATTERNS.search(text):
            return text, False

        sanitized_lines = []
        injection_found = False
        for line in text.splitlines():
            if INJECTION_PATTERNS.search(line):
                injection_found = True
                logger.warning("Indirect prompt injection pattern stripped from context: '%s'", line.strip())
                continue
            sanitized_lines.append(line)

        cleaned = "\n".join(sanitized_lines).strip()
        return cleaned, injection_found

    def ingest_document(
        self,
        file_path: Path | str,
        *,
        user_scope: str = "default_user",
        chunk_size: int = 800,
        overlap: int = 150,
        verification_status: str = "unverified",
    ) -> list[dict[str, Any]]:
        """
        Parses document, splits into structured chunks with page provenance,
        computes SHA-256 digests, and indexes them into memory.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {path}")

        doc_data = DocumentExtractor.extract_file(str(path))
        chunks = DocumentExtractor.chunk_document(doc_data, max_chunk_size=chunk_size, overlap=overlap)

        stored_chunks: list[dict[str, Any]] = []
        doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        for c in chunks:
            raw_content = c["content"]
            # Sanitize upon ingest as first line of defense
            safe_content, _ = self.sanitize_chunk(raw_content)
            if not safe_content:
                continue

            page_num = c.get("page_number", 1)
            chunk_num = c.get("chunk_id", 1)
            source_label = f"doc:{path.name}:p{page_num}:c{chunk_num}"

            record = self.store.add(
                safe_content,
                source=source_label,
                memory_type="semantic",
                confidence=0.85 if verification_status == "verified" else 0.65,
                verification_status=verification_status,
                user_scope=user_scope,
                metadata={
                    "filename": path.name,
                    "page_number": page_num,
                    "chunk_id": chunk_num,
                    "doc_sha256": doc_hash,
                },
            )
            stored_chunks.append(record)

        return stored_chunks

    def retrieve_context(
        self,
        query: str,
        *,
        user_scope: str = "default_user",
        limit: int = 5,
        min_relevance: float = 0.20,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieves relevant document chunks using hybrid, vector, or keyword search.
        Applies relevance scoring, indirect injection sanitization, and contradiction checks.
        """
        if search_mode == "vector":
            candidates = self.store.vector_search(
                query,
                limit=limit * 2,
                user_scope=user_scope,
                include_global=include_global,
            )
        elif search_mode == "keyword":
            candidates = self.store.search(
                query,
                limit=limit * 2,
                user_scope=user_scope,
                include_global=include_global,
            )
        else:
            candidates = self.store.hybrid_search(
                query,
                limit=limit * 2,
                alpha=alpha,
                user_scope=user_scope,
                include_global=include_global,
            )

        sanitized_results: list[dict[str, Any]] = []

        for item in candidates:
            # Check relevance score
            relevance = item.get("hybrid_score") or item.get("similarity") or item.get("importance", 0.5)
            if relevance < min_relevance:
                continue

            # Strip indirect prompt injection attempts
            clean_content, injection_detected = self.sanitize_chunk(item["content"])
            if not clean_content:
                continue

            res_item = dict(item)
            res_item["content"] = clean_content
            res_item["injection_detected"] = injection_detected
            res_item["relevance_score"] = round(float(relevance), 4)

            # Check for contradiction with existing verified knowledge
            contradictions = self.store.detect_contradictions(clean_content)
            res_item["contradictions"] = contradictions

            sanitized_results.append(res_item)
            if len(sanitized_results) >= limit:
                break

        return sanitized_results

    def build_context_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        *,
        max_chars: int = 2500,
    ) -> str:
        """Assembles retrieved chunks into a structured, citation-ready context block."""
        if not retrieved_chunks:
            return ""

        context_parts = ["[GROUNDED RETRIEVED CONTEXT]"]
        total_chars = 0

        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "internal_knowledge")
            score = chunk.get("relevance_score", 0.0)
            text = chunk.get("content", "").strip()

            snippet = f"\n[Source {idx} | Ref: {source} | Relevance: {score}]\n{text}\n"
            if total_chars + len(snippet) > max_chars:
                break

            context_parts.append(snippet)
            total_chars += len(snippet)

        context_parts.append("[END CONTEXT]\n")
        return "\n".join(context_parts)

    @staticmethod
    def evaluate_retrieval(
        retrieved_chunk_ids: list[str],
        ground_truth_chunk_ids: list[str],
        *,
        top_k: int = 5,
    ) -> dict[str, float]:
        """Computes objective retrieval metrics: Precision@K, Recall@K, and Mean Reciprocal Rank (MRR)."""
        if not ground_truth_chunk_ids:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

        top_retrieved = retrieved_chunk_ids[:top_k]
        hits = [cid for cid in top_retrieved if cid in ground_truth_chunk_ids]

        precision = len(hits) / max(1, len(top_retrieved))
        recall = len(hits) / max(1, len(ground_truth_chunk_ids))

        # Reciprocal Rank of first relevant item
        rr = 0.0
        for rank, cid in enumerate(top_retrieved, 1):
            if cid in ground_truth_chunk_ids:
                rr = 1.0 / rank
                break

        return {
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "mrr": round(rr, 4),
        }
