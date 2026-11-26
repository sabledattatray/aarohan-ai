"""Persistent six-layer cognitive memory architecture for AAROHAN AI.

Layers:
  1. Working Memory: Ephemeral active session context & temporary reasoning scratchpad.
  2. Episodic Memory: Events, conversation turns, attempted tasks, and user corrections.
  3. Semantic Memory: Concepts, verified definitions, extracted facts, and rules.
  4. Procedural Memory: Reusable workflows, recipes, and verified instructions.
  5. Knowledge Graph: Structured entities and relational triples (subject-predicate-object).
  6. Learning Memory: Training benchmarks, loss metrics, knowledge gaps, and model evaluations.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

VALID_MEMORY_TYPES = frozenset({"working", "episodic", "semantic", "procedural", "learning"})
VALID_PRIVACY_CLASSES = frozenset({"non_sensitive", "personal", "sensitive"})
VALID_VERIFICATION_STATUSES = frozenset({"unverified", "verified", "superseded"})
VALID_RELATION_TYPES = frozenset({
    "is_a", "part_of", "used_in", "trained_on", "implements", "evaluates",
    "depends_on", "contradicts", "causes", "derived_from",
})
VALID_LINK_TYPES = frozenset({"supports", "contradicts", "derived_from", "supersedes", "relates_to"})


def compute_dense_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Deterministic offline dense embedding using multi-gram hashing and L2 normalization.
    Ensures 100% reproducible semantic similarity search without external model dependencies.
    """
    vec = np.zeros(dim, dtype=np.float32)
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return vec

    # Unigrams and bigrams
    words = re.findall(r"\b\w+\b", normalized)
    tokens: list[str] = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    # Character 3-grams for morphological similarity
    for i in range(len(normalized) - 2):
        tokens.append(normalized[i : i + 3])

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def embedding_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes | None, dim: int = 128) -> np.ndarray:
    if not blob:
        return np.zeros(dim, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


class MemoryStore:
    """Six-layer persistent memory store with SQLite, FTS5, vector search, and Knowledge Graph."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = True
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            # 1. Core multi-layer memory table
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence REAL NOT NULL,
                    verification_status TEXT NOT NULL,
                    verified_by TEXT,
                    privacy_classification TEXT NOT NULL,
                    importance REAL NOT NULL,
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    superseded_by TEXT,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    metadata_json TEXT
                )
            """)

            # Migration check: ensure newer columns exist if db was created in earlier prompt
            cursor = connection.execute("PRAGMA table_info(memories)")
            columns = {row["name"] for row in cursor.fetchall()}
            if columns:
                if "embedding" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
                if "last_accessed_at" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT")
                if "metadata_json" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN metadata_json TEXT")

            # 2. Knowledge Graph: Entities
            connection.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 3. Knowledge Graph: Relations
            connection.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
                    target_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                )
            """)

            # 4. Memory-to-Memory relational links (corroboration, contradictions)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    source_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    target_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    link_type TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_memory_id, target_memory_id, link_type)
                )
            """)

            # 5. Full-text search index (FTS5)
            try:
                connection.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(memory_id UNINDEXED, content, source, tokenize='unicode61')
                """)
                connection.execute("""
                    INSERT INTO memory_fts(memory_id, content, source)
                    SELECT id, content, COALESCE(source, '') FROM memories
                    WHERE id NOT IN (SELECT memory_id FROM memory_fts)
                """)
            except sqlite3.OperationalError:
                self._fts_available = False

    @staticmethod
    def _digest(content: str) -> str:
        normalized = " ".join(content.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        if "embedding" in data and isinstance(data["embedding"], bytes):
            data["embedding_dim"] = len(data["embedding"]) // 4
            del data["embedding"]
        return data

    # -------------------------------------------------------------------------
    # Core Memory Operations
    # -------------------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        source: str | None = None,
        memory_type: str = "episodic",
        confidence: float = 0.5,
        verification_status: str = "unverified",
        verified_by: str | None = None,
        privacy_classification: str = "non_sensitive",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type '{memory_type}'. Valid types: {sorted(VALID_MEMORY_TYPES)}")
        if privacy_classification not in VALID_PRIVACY_CLASSES:
            raise ValueError(f"privacy_classification must be one of {sorted(VALID_PRIVACY_CLASSES)}")
        if privacy_classification == "sensitive":
            raise ValueError("Sensitive memories are blocked until encrypted storage and access controls exist")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ValueError("confidence and importance must be in [0, 1]")
        if verification_status not in VALID_VERIFICATION_STATUSES:
            raise ValueError(f"verification_status must be one of {sorted(VALID_VERIFICATION_STATUSES)}")
        if verification_status == "verified" and (not source or not verified_by):
            raise ValueError("Verified memories require a source and verified_by")

        digest = self._digest(content)
        embedding_blob = embedding_to_blob(compute_dense_embedding(content))
        now = datetime.now(UTC).isoformat()
        memory_id = str(uuid.uuid4())
        meta_str = json.dumps(metadata) if metadata else None

        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM memories WHERE content_sha256 = ?", (digest,)).fetchone()
            if existing:
                if verification_status == "verified" and existing["verification_status"] != "verified":
                    connection.execute("""
                        UPDATE memories SET source=?, confidence=?, verification_status='verified',
                            verified_by=?, updated_at=?, metadata_json=COALESCE(?, metadata_json)
                        WHERE id=?
                    """, (source, confidence, verified_by, now, meta_str, existing["id"]))
                    existing = connection.execute("SELECT * FROM memories WHERE id = ?", (existing["id"],)).fetchone()
                return self._record(existing)

            connection.execute("""
                INSERT INTO memories(
                    id, memory_type, content, source, confidence,
                    verification_status, verified_by, privacy_classification,
                    importance, embedding, created_at, updated_at, last_accessed_at,
                    superseded_by, content_sha256, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """, (
                memory_id, memory_type, content, source, confidence,
                verification_status, verified_by, privacy_classification,
                importance, embedding_blob, now, now, now, digest, meta_str,
            ))

            if self._fts_available:
                connection.execute(
                    "INSERT INTO memory_fts(memory_id, content, source) VALUES (?, ?, ?)",
                    (memory_id, content, source or ""),
                )
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record(row)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                now = datetime.now(UTC).isoformat()
                connection.execute("UPDATE memories SET last_accessed_at = ? WHERE id = ?", (now, memory_id))
                return self._record(row)
            return None

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_type: str | None = None,
        verified_only: bool = False,
        source_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword-based search using SQLite FTS5 with fallback to LIKE query."""
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []

        type_sql = f" AND m.memory_type = '{memory_type}'" if memory_type else ""
        ver_sql = " AND m.verification_status = 'verified'" if verified_only else ""

        with self._connect() as connection:
            if self._fts_available:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                params: list[Any] = [expression]
                filter_sql = f"{ver_sql}{type_sql}"
                if source_contains:
                    filter_sql += " AND m.source LIKE ?"
                    params.append(f"%{source_contains}%")
                params.append(limit)
                try:
                    rows = connection.execute(f"""
                        SELECT m.* FROM memory_fts f JOIN memories m ON m.id = f.memory_id
                        WHERE memory_fts MATCH ?{filter_sql}
                        ORDER BY bm25(memory_fts), m.importance DESC, m.created_at DESC LIMIT ?
                    """, params).fetchall()
                    return [self._record(row) for row in rows]
                except sqlite3.OperationalError:
                    pass

            predicates = " OR ".join("content LIKE ?" for _ in terms)
            fallback_params: list[Any] = [f"%{term}%" for term in terms]
            filter_sql = f"{ver_sql}{type_sql}"
            if source_contains:
                filter_sql += " AND source LIKE ?"
                fallback_params.append(f"%{source_contains}%")
            fallback_params.append(limit)

            rows = connection.execute(f"""
                SELECT * FROM memories WHERE ({predicates}){filter_sql}
                ORDER BY importance DESC, created_at DESC LIMIT ?
            """, fallback_params).fetchall()
            return [self._record(row) for row in rows]

    def vector_search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float = 0.1,
        memory_type: str | None = None,
        verified_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic vector similarity search using offline dense embeddings."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vec = compute_dense_embedding(query)
        if np.linalg.norm(query_vec) < 1e-9:
            return []

        where_clauses: list[str] = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if memory_type:
            where_clauses.append("memory_type = ?")
            params.append(memory_type)
        if verified_only:
            where_clauses.append("verification_status = 'verified'")

        where_sql = " AND ".join(where_clauses)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {where_sql} ORDER BY created_at DESC",
                params,
            ).fetchall()

        results: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            record = self._record(row)
            doc_vec = blob_to_embedding(row["embedding"])
            similarity = float(np.dot(query_vec, doc_vec))
            if similarity >= min_similarity:
                record["similarity"] = round(similarity, 4)
                results.append((similarity, record))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:limit]]

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 5,
        alpha: float = 0.5,
        memory_type: str | None = None,
        verified_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Combined hybrid search blending keyword relevance and semantic vector similarity."""
        vector_matches = self.vector_search(query, limit=limit * 2, memory_type=memory_type, verified_only=verified_only)
        keyword_matches = self.search(query, limit=limit * 2, memory_type=memory_type, verified_only=verified_only)

        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vector_matches):
            mid = item["id"]
            records[mid] = item
            v_score = item.get("similarity", 0.5)
            scores[mid] = scores.get(mid, 0.0) + alpha * v_score

        for rank, item in enumerate(keyword_matches):
            mid = item["id"]
            records[mid] = item
            k_score = 1.0 / (rank + 1.0)
            scores[mid] = scores.get(mid, 0.0) + (1.0 - alpha) * k_score

        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:limit]
        out: list[dict[str, Any]] = []
        for mid in sorted_ids:
            rec = dict(records[mid])
            rec["hybrid_score"] = round(scores[mid], 4)
            out.append(rec)
        return out

    def correct(self, memory_id: str, new_content: str, *, source: str, corrected_by: str) -> dict[str, Any]:
        old = self.get(memory_id)
        if old is None:
            raise KeyError(f"Unknown memory id: {memory_id}")
        replacement = self.add(
            new_content,
            source=source,
            memory_type=old["memory_type"],
            confidence=1.0,
            verification_status="verified",
            verified_by=corrected_by,
            privacy_classification=old["privacy_classification"],
            importance=old["importance"],
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE memories SET verification_status='superseded', updated_at=?, superseded_by=? WHERE id=?",
                (now, replacement["id"], memory_id),
            )
        self.link_memories(memory_id, replacement["id"], link_type="supersedes", notes="Correction update")
        return replacement

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            if self._fts_available:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Knowledge Graph Operations (Layer 5)
    # -------------------------------------------------------------------------

    def add_entity(self, name: str, *, entity_type: str = "concept", description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Entity name cannot be empty")
        now = datetime.now(UTC).isoformat()
        entity_id = str(uuid.uuid4())
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM knowledge_entities WHERE name = ?", (name,)).fetchone()
            if existing:
                if description and not existing["description"]:
                    connection.execute(
                        "UPDATE knowledge_entities SET description=?, updated_at=? WHERE id=?",
                        (description, now, existing["id"]),
                    )
                    existing = connection.execute("SELECT * FROM knowledge_entities WHERE id = ?", (existing["id"],)).fetchone()
                return dict(existing)
            connection.execute("""
                INSERT INTO knowledge_entities(id, name, entity_type, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_id, name, entity_type, description, now, now))
            row = connection.execute("SELECT * FROM knowledge_entities WHERE id = ?", (entity_id,)).fetchone()
            return dict(row)

    def get_entity(self, name_or_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_entities WHERE id = ? OR name = ?",
                (name_or_id, name_or_id),
            ).fetchone()
            return dict(row) if row else None

    def list_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if entity_type:
                rows = connection.execute(
                    "SELECT * FROM knowledge_entities WHERE entity_type = ? ORDER BY name ASC",
                    (entity_type,),
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM knowledge_entities ORDER BY name ASC").fetchall()
            return [dict(r) for r in rows]

    def add_relation(
        self,
        source_name_or_id: str,
        target_name_or_id: str,
        relation_type: str,
        *,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> dict[str, Any]:
        if relation_type not in VALID_RELATION_TYPES:
            raise ValueError(f"Invalid relation_type '{relation_type}'. Valid types: {sorted(VALID_RELATION_TYPES)}")
        src = self.get_entity(source_name_or_id) or self.add_entity(source_name_or_id)
        tgt = self.get_entity(target_name_or_id) or self.add_entity(target_name_or_id)
        relation_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO knowledge_relations(id, source_entity_id, target_entity_id, relation_type, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity_id, target_entity_id, relation_type)
                DO UPDATE SET confidence=excluded.confidence, source=COALESCE(excluded.source, knowledge_relations.source)
            """, (relation_id, src["id"], tgt["id"], relation_type, confidence, source, now))
            row = connection.execute("""
                SELECT r.*, s.name as source_name, t.name as target_name
                FROM knowledge_relations r
                JOIN knowledge_entities s ON s.id = r.source_entity_id
                JOIN knowledge_entities t ON t.id = r.target_entity_id
                WHERE r.source_entity_id = ? AND r.target_entity_id = ? AND r.relation_type = ?
            """, (src["id"], tgt["id"], relation_type)).fetchone()
            return dict(row)

    def get_relations(self, entity_name_or_id: str, direction: str = "both") -> list[dict[str, Any]]:
        entity = self.get_entity(entity_name_or_id)
        if not entity:
            return []
        eid = entity["id"]
        params: tuple[Any, ...]
        with self._connect() as connection:
            if direction == "out":
                query = """
                    SELECT r.*, s.name as source_name, t.name as target_name
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    WHERE r.source_entity_id = ?
                """
                params = (eid,)
            elif direction == "in":
                query = """
                    SELECT r.*, s.name as source_name, t.name as target_name
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    WHERE r.target_entity_id = ?
                """
                params = (eid,)
            else:
                query = """
                    SELECT r.*, s.name as source_name, t.name as target_name
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    WHERE r.source_entity_id = ? OR r.target_entity_id = ?
                """
                params = (eid, eid)
            rows = connection.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def find_path(self, start: str, end: str, max_depth: int = 3) -> list[list[str]]:
        """Find relationship paths between two concepts in the knowledge graph."""
        src = self.get_entity(start)
        dst = self.get_entity(end)
        if not src or not dst:
            return []
        if src["id"] == dst["id"]:
            return [[src["name"]]]

        visited = {src["id"]}
        queue: list[tuple[str, list[str]]] = [(src["id"], [src["name"]])]
        paths: list[list[str]] = []

        while queue:
            curr_id, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for rel in self.get_relations(curr_id, direction="out"):
                next_id = rel["target_entity_id"]
                next_name = rel["target_name"]
                if next_id == dst["id"]:
                    paths.append(path + [rel["relation_type"], next_name])
                elif next_id not in visited and len(path) < max_depth:
                    visited.add(next_id)
                    queue.append((next_id, path + [rel["relation_type"], next_name]))

        return paths

    # -------------------------------------------------------------------------
    # Memory Links & Contradiction Detection
    # -------------------------------------------------------------------------

    def link_memories(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        if link_type not in VALID_LINK_TYPES:
            raise ValueError(f"Invalid link_type '{link_type}'. Valid types: {sorted(VALID_LINK_TYPES)}")
        link_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO memory_links(id, source_memory_id, target_memory_id, link_type, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_memory_id, target_memory_id, link_type)
                DO UPDATE SET notes=COALESCE(excluded.notes, memory_links.notes)
            """, (link_id, source_id, target_id, link_type, notes, now))
            row = connection.execute("SELECT * FROM memory_links WHERE id = ?", (link_id,)).fetchone()
            return dict(row) if row else {"id": link_id, "status": "linked"}

    def detect_contradictions(self, content: str, threshold: float = 0.75) -> list[dict[str, Any]]:
        """
        Check if newly proposed content conflicts with existing verified or high-confidence memories.
        Detects opposing claims or negation patterns between semantically related items.
        """
        matches = self.vector_search(content, limit=5, min_similarity=threshold)
        contradictions = []
        negation_markers = {"not", "never", "cannot", "no longer", "false", "disproved"}
        content_words = set(re.findall(r"\b\w+\b", content.lower()))

        for item in matches:
            existing_words = set(re.findall(r"\b\w+\b", item["content"].lower()))
            overlap = len(content_words & existing_words)
            if overlap >= 3:
                has_new_negation = bool(content_words & negation_markers)
                has_old_negation = bool(existing_words & negation_markers)
                if has_new_negation != has_old_negation:
                    contradictions.append({
                        "conflicting_memory_id": item["id"],
                        "conflicting_content": item["content"],
                        "similarity": item.get("similarity", 0.0),
                        "reason": "Polarity mismatch detected on shared subject matter",
                    })

        return contradictions

    # -------------------------------------------------------------------------
    # Memory Consolidation Engine (PRD Section 9.6)
    # -------------------------------------------------------------------------

    def consolidate(self, *, recent_limit: int = 50) -> dict[str, Any]:
        """
        Scheduled consolidation engine:
        1. Reviews recent episodic records.
        2. Identifies recurring concepts and extracts candidate knowledge entities.
        3. Detects duplicates and links related memories.
        4. Preserves verification status: unverified claims remain unverified.
        """
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT * FROM memories
                WHERE memory_type = 'episodic' AND verification_status != 'superseded'
                ORDER BY created_at DESC LIMIT ?
            """, (recent_limit,)).fetchall()

        episodics = [self._record(r) for r in rows]
        new_entities_created = 0
        new_links_created = 0
        recurring_concepts: dict[str, int] = {}

        for mem in episodics:
            words = [w.capitalize() for w in re.findall(r"\b[A-Za-z]{4,}\b", mem["content"])]
            for w in words:
                recurring_concepts[w] = recurring_concepts.get(w, 0) + 1

        # Promising concepts that appear repeatedly become candidate entities
        for concept, count in recurring_concepts.items():
            if count >= 2:
                existing = self.get_entity(concept)
                if not existing:
                    self.add_entity(concept, entity_type="concept", description=f"Extracted during consolidation from {count} episodic memories")
                    new_entities_created += 1

        # Find closely related memories and establish links
        for i, mem1 in enumerate(episodics):
            for mem2 in episodics[i + 1 :]:
                if mem1["id"] == mem2["id"]:
                    continue
                v1 = compute_dense_embedding(mem1["content"])
                v2 = compute_dense_embedding(mem2["content"])
                sim = float(np.dot(v1, v2))
                if sim >= 0.85:
                    self.link_memories(mem1["id"], mem2["id"], link_type="relates_to", notes=f"Consolidated semantic similarity {sim:.2f}")
                    new_links_created += 1

        return {
            "episodic_records_inspected": len(episodics),
            "recurring_concepts_identified": len([c for c, cnt in recurring_concepts.items() if cnt >= 2]),
            "new_entities_created": new_entities_created,
            "new_links_created": new_links_created,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # -------------------------------------------------------------------------
    # Learning Memory Operations (Layer 6)
    # -------------------------------------------------------------------------

    def record_learning_event(
        self,
        event_type: str,
        title: str,
        details: dict[str, Any],
        *,
        importance: float = 0.8,
    ) -> dict[str, Any]:
        """Record training milestones, evaluation failures, or knowledge gaps in Layer 6."""
        content = f"[{event_type.upper()}] {title}: " + json.dumps(details, ensure_ascii=False)
        return self.add(
            content,
            source="learning_engine",
            memory_type="learning",
            confidence=1.0,
            verification_status="verified",
            verified_by="aarohan_orchestrator",
            importance=importance,
            metadata={"event_type": event_type, "details": details},
        )
