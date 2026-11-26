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
import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from aarohan.cognitive_memory import CognitiveMemoryEngine

logger = logging.getLogger("aarohan.memory")

VALID_MEMORY_TYPES = frozenset({"working", "episodic", "semantic", "procedural", "reflective", "learning"})
VALID_PRIVACY_CLASSES = frozenset({"non_sensitive", "personal", "sensitive"})
VALID_VERIFICATION_STATUSES = frozenset({"unverified", "verified", "superseded"})
VALID_RELATION_TYPES = frozenset({
    "is_a", "part_of", "used_in", "trained_on", "implements", "evaluates",
    "depends_on", "contradicts", "causes", "derived_from",
    "developed", "uses", "contains", "created_by", "developed_by", "resides_in", "associated_with",
})
VALID_RELATION_CLASSES = frozenset({"verified", "inferred"})
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

    def __init__(self, database_path: Path | str | None = None, db_path: Path | str | None = None) -> None:
        chosen_path = database_path or db_path or Path("data/memory.db")
        self.database_path = Path(chosen_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = True
        self.cognitive = CognitiveMemoryEngine(self.database_path)
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            # 1. Core multi-layer memory table with user isolation and TTL support
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
                    user_scope TEXT DEFAULT 'default_user',
                    expires_at TEXT DEFAULT NULL,
                    fact_type TEXT DEFAULT 'fact',
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
                if "user_scope" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN user_scope TEXT DEFAULT 'default_user'")
                if "expires_at" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT DEFAULT NULL")
                if "fact_type" not in columns:
                    connection.execute("ALTER TABLE memories ADD COLUMN fact_type TEXT DEFAULT 'fact'")

            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_scope ON memories(user_scope)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")

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
                    relation_class TEXT DEFAULT 'verified',
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                )
            """)

            # Migration check: knowledge_entities
            cursor_ent = connection.execute("PRAGMA table_info(knowledge_entities)")
            ent_columns = {row["name"] for row in cursor_ent.fetchall()}
            if ent_columns:
                if "canonical_name" not in ent_columns:
                    connection.execute("ALTER TABLE knowledge_entities ADD COLUMN canonical_name TEXT DEFAULT NULL")
                if "aliases_json" not in ent_columns:
                    connection.execute("ALTER TABLE knowledge_entities ADD COLUMN aliases_json TEXT DEFAULT '[]'")

            # Migration check: knowledge_relations
            cursor_rel = connection.execute("PRAGMA table_info(knowledge_relations)")
            rel_columns = {row["name"] for row in cursor_rel.fetchall()}
            if rel_columns:
                if "relation_class" not in rel_columns:
                    connection.execute("ALTER TABLE knowledge_relations ADD COLUMN relation_class TEXT DEFAULT 'verified'")
                if "valid_from" not in rel_columns:
                    connection.execute("ALTER TABLE knowledge_relations ADD COLUMN valid_from TEXT DEFAULT NULL")
                if "valid_until" not in rel_columns:
                    connection.execute("ALTER TABLE knowledge_relations ADD COLUMN valid_until TEXT DEFAULT NULL")

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
    def _digest(content: str, user_scope: str = "default_user") -> str:
        normalized = " ".join(content.casefold().split())
        return hashlib.sha256(f"{user_scope}::{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        if "embedding" in data and isinstance(data["embedding"], bytes):
            data["embedding_dim"] = len(data["embedding"]) // 4
            del data["embedding"]
        if "metadata_json" in data:
            try:
                data["metadata"] = json.loads(data["metadata_json"]) if data["metadata_json"] else {}
            except Exception:
                data["metadata"] = {}
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
        user_scope: str = "default_user",
        ttl_seconds: int | None = None,
        expires_at: str | None = None,
        fact_type: str = "fact",
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

        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        if ttl_seconds is not None and ttl_seconds > 0:
            computed_expires = (now_dt + timedelta(seconds=ttl_seconds)).isoformat()
        else:
            computed_expires = expires_at

        digest = self._digest(content, user_scope=user_scope)
        embedding_blob = embedding_to_blob(compute_dense_embedding(content))
        memory_id = str(uuid.uuid4())
        meta_str = json.dumps(metadata) if metadata else None

        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM memories WHERE content_sha256 = ?", (digest,)).fetchone()
            if existing:
                if verification_status == "verified" and existing["verification_status"] != "verified":
                    connection.execute("""
                        UPDATE memories SET source=?, confidence=?, verification_status='verified',
                            verified_by=?, updated_at=?, expires_at=?, metadata_json=COALESCE(?, metadata_json)
                        WHERE id=?
                    """, (source, confidence, verified_by, now, computed_expires, meta_str, existing["id"]))
                    existing = connection.execute("SELECT * FROM memories WHERE id = ?", (existing["id"],)).fetchone()
                return self._record(existing)

            connection.execute("""
                INSERT INTO memories(
                    id, memory_type, content, source, confidence,
                    verification_status, verified_by, privacy_classification,
                    importance, embedding, created_at, updated_at, last_accessed_at,
                    superseded_by, content_sha256, user_scope, expires_at, fact_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
            """, (
                memory_id, memory_type, content, source, confidence,
                verification_status, verified_by, privacy_classification,
                importance, embedding_blob, now, now, now, digest,
                user_scope, computed_expires, fact_type, meta_str,
            ))

            if self._fts_available:
                connection.execute(
                    "INSERT INTO memory_fts(memory_id, content, source) VALUES (?, ?, ?)",
                    (memory_id, content, source or ""),
                )
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record(row)

    def get(self, memory_id: str, *, user_scope: str | None = None) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                rec = self._record(row)
                if user_scope and rec.get("user_scope") not in (user_scope, "global", "system", "default_user", None):
                    return None
                now = datetime.now(UTC).isoformat()
                connection.execute("UPDATE memories SET last_accessed_at = ? WHERE id = ?", (now, memory_id))
                return rec
            return None

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_type: str | None = None,
        verified_only: bool = False,
        source_contains: str | None = None,
        user_scope: str | None = None,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        """Keyword-based search using SQLite FTS5 with fallback to LIKE query."""
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []

        now_iso = datetime.now(UTC).isoformat()
        type_sql = f" AND m.memory_type = '{memory_type}'" if memory_type else ""
        ver_sql = " AND m.verification_status = 'verified'" if verified_only else ""
        expiry_sql = f" AND (m.expires_at IS NULL OR m.expires_at > '{now_iso}')"

        user_sql = ""
        user_params: list[Any] = []
        if user_scope:
            if include_global:
                user_sql = " AND (m.user_scope = ? OR m.user_scope IN ('global', 'system', 'default_user') OR m.user_scope IS NULL)"
            else:
                user_sql = " AND m.user_scope = ?"
            user_params.append(user_scope)

        with self._connect() as connection:
            if self._fts_available:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                params: list[Any] = [expression]
                filter_sql = f"{ver_sql}{type_sql}{expiry_sql}{user_sql}"
                params.extend(user_params)
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

            fallback_type_sql = f" AND memory_type = '{memory_type}'" if memory_type else ""
            fallback_ver_sql = " AND verification_status = 'verified'" if verified_only else ""
            fallback_expiry_sql = f" AND (expires_at IS NULL OR expires_at > '{now_iso}')"
            fallback_user_sql = ""
            fallback_user_params: list[Any] = []
            if user_scope:
                if include_global:
                    fallback_user_sql = " AND (user_scope = ? OR user_scope IN ('global', 'system', 'default_user') OR user_scope IS NULL)"
                else:
                    fallback_user_sql = " AND user_scope = ?"
                fallback_user_params.append(user_scope)

            predicates = " OR ".join("content LIKE ?" for _ in terms)
            fallback_params: list[Any] = [f"%{term}%" for term in terms]
            filter_sql = f"{fallback_ver_sql}{fallback_type_sql}{fallback_expiry_sql}{fallback_user_sql}"
            fallback_params.extend(fallback_user_params)
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
        user_scope: str | None = None,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic vector similarity search using offline dense embeddings."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vec = compute_dense_embedding(query)
        if np.linalg.norm(query_vec) < 1e-9:
            return []

        now_iso = datetime.now(UTC).isoformat()
        where_clauses: list[str] = [
            "embedding IS NOT NULL",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        params: list[Any] = [now_iso]

        if user_scope:
            if include_global:
                where_clauses.append("(user_scope = ? OR user_scope IN ('global', 'system', 'default_user') OR user_scope IS NULL)")
            else:
                where_clauses.append("user_scope = ?")
            params.append(user_scope)

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
        user_scope: str | None = None,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        """Combined hybrid search blending keyword relevance and semantic vector similarity."""
        vector_matches = self.vector_search(
            query,
            limit=limit * 2,
            memory_type=memory_type,
            verified_only=verified_only,
            user_scope=user_scope,
            include_global=include_global,
        )
        keyword_matches = self.search(
            query,
            limit=limit * 2,
            memory_type=memory_type,
            verified_only=verified_only,
            user_scope=user_scope,
            include_global=include_global,
        )

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

    def correct(
        self,
        memory_id: str,
        new_content: str,
        *,
        source: str,
        corrected_by: str,
        user_scope: str = "default_user",
    ) -> dict[str, Any]:
        old = self.get(memory_id, user_scope=user_scope)
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
            user_scope=user_scope,
            fact_type=old.get("fact_type", "fact"),
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE memories SET verification_status='superseded', updated_at=?, superseded_by=? WHERE id=?",
                (now, replacement["id"], memory_id),
            )
        self.link_memories(memory_id, replacement["id"], link_type="supersedes", notes="Correction update")
        return replacement

    def delete(self, memory_id: str, *, user_scope: str | None = None) -> bool:
        with self._connect() as connection:
            if user_scope:
                mem = connection.execute("SELECT user_scope FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if not mem or (mem["user_scope"] != user_scope and mem["user_scope"] not in ("global", "system", "default_user", None)):
                    return False
            if self._fts_available:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Knowledge Graph Operations (Layer 5)
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """Normalizes entity names: trims extra whitespace and standardizes casing for canonical matching."""
        return " ".join(name.strip().split())

    def add_entity(
        self,
        name: str,
        *,
        entity_type: str = "concept",
        description: str = "",
        aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_entity_name(name)
        if not normalized:
            raise ValueError("Entity name cannot be empty")
        now = datetime.now(UTC).isoformat()
        entity_id = str(uuid.uuid4())
        aliases_list = [self.normalize_entity_name(a) for a in (aliases or []) if a.strip()]
        aliases_json = json.dumps(list(set(aliases_list)), ensure_ascii=False)

        with self._connect() as connection:
            # Case-insensitive duplicate prevention
            existing = connection.execute(
                "SELECT * FROM knowledge_entities WHERE LOWER(name) = LOWER(?)",
                (normalized,),
            ).fetchone()
            if not existing and aliases_list:
                for alias in aliases_list:
                    existing = connection.execute(
                        "SELECT * FROM knowledge_entities WHERE LOWER(name) = LOWER(?) OR aliases_json LIKE ?",
                        (alias, f"%\"{alias}\"%"),
                    ).fetchone()
                    if existing:
                        break

            if existing:
                existing_dict = dict(existing)
                updated_desc = description if (description and not existing_dict.get("description")) else existing_dict.get("description", "")
                existing_aliases = json.loads(existing_dict.get("aliases_json") or "[]")
                merged_aliases = list(set(existing_aliases + aliases_list))
                connection.execute(
                    "UPDATE knowledge_entities SET description=?, aliases_json=?, updated_at=? WHERE id=?",
                    (updated_desc, json.dumps(merged_aliases, ensure_ascii=False), now, existing_dict["id"]),
                )
                row = connection.execute("SELECT * FROM knowledge_entities WHERE id = ?", (existing_dict["id"],)).fetchone()
                return dict(row)

            connection.execute("""
                INSERT INTO knowledge_entities(id, name, entity_type, description, canonical_name, aliases_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, normalized, entity_type, description, normalized.lower(), aliases_json, now, now))
            row = connection.execute("SELECT * FROM knowledge_entities WHERE id = ?", (entity_id,)).fetchone()
            return dict(row)

    def get_entity(self, name_or_id: str) -> dict[str, Any] | None:
        target = self.normalize_entity_name(name_or_id) if isinstance(name_or_id, str) else name_or_id
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_entities WHERE id = ? OR LOWER(name) = LOWER(?)",
                (target, target),
            ).fetchone()
            if not row:
                row = connection.execute(
                    "SELECT * FROM knowledge_entities WHERE aliases_json LIKE ?",
                    (f"%\"{target}\"%",),
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
        relation_class: str = "verified",
        valid_from: str | None = None,
        valid_until: str | None = None,
    ) -> dict[str, Any]:
        if relation_type not in VALID_RELATION_TYPES:
            raise ValueError(f"Invalid relation_type '{relation_type}'. Valid types: {sorted(VALID_RELATION_TYPES)}")
        if relation_class not in VALID_RELATION_CLASSES:
            raise ValueError(f"Invalid relation_class '{relation_class}'. Valid: {sorted(VALID_RELATION_CLASSES)}")

        src = self.get_entity(source_name_or_id) or self.add_entity(source_name_or_id)
        tgt = self.get_entity(target_name_or_id) or self.add_entity(target_name_or_id)

        # Self-referencing loop constraint
        if src["id"] == tgt["id"] and relation_type in ("contradicts", "developed"):
            raise ValueError(f"Invalid self-referencing relationship '{relation_type}' for entity '{src['name']}'")

        relation_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO knowledge_relations(id, source_entity_id, target_entity_id, relation_type, confidence, source, relation_class, valid_from, valid_until, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity_id, target_entity_id, relation_type)
                DO UPDATE SET
                    confidence=excluded.confidence,
                    source=COALESCE(excluded.source, knowledge_relations.source),
                    relation_class=excluded.relation_class,
                    valid_from=COALESCE(excluded.valid_from, knowledge_relations.valid_from),
                    valid_until=excluded.valid_until
            """, (relation_id, src["id"], tgt["id"], relation_type, confidence, source, relation_class, valid_from, valid_until, now))
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

    def find_path(self, start: str, end: str, max_depth: int = 4, direction: str = "out") -> list[list[str]]:
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
            if (len(path) - 1) // 2 >= max_depth:
                continue
            for rel in self.get_relations(curr_id, direction=direction):
                if rel["source_entity_id"] == curr_id:
                    next_id = rel["target_entity_id"]
                    next_name = rel["target_name"]
                    edge_str = rel["relation_type"]
                else:
                    next_id = rel["source_entity_id"]
                    next_name = rel["source_name"]
                    edge_str = f"<-({rel['relation_type']})-"

                if next_id == dst["id"]:
                    paths.append(path + [edge_str, next_name])
                elif next_id not in visited and (len(path) - 1) // 2 < max_depth:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge_str, next_name]))

        return paths

    def extract_entities_and_relations(
        self,
        text: str,
        *,
        source: str = "text_extraction",
        relation_class: str = "inferred",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Extracts verified or inferred entities and explicit relationships from natural language statements.
        Does NOT invent connections merely because entities co-occur in the same sentence.
        Requires explicit relational verbs (developed, created, uses, contains, is a, depends on).
        """
        extracted_entities: list[dict[str, Any]] = []
        extracted_relations: list[dict[str, Any]] = []

        patterns = [
            (r"([A-Z][\w\s.-]+?)\s+(?:developed|created|built|authored)\s+([A-Z][\w\s.-]+)", "developed", 0.95),
            (r"([A-Z][\w\s.-]+?)\s+(?:uses|utilizes|relies on|is powered by)\s+([A-Z][\w\s.-]+)", "uses", 0.90),
            (r"([A-Z][\w\s.-]+?)\s+(?:contains|includes|consists of|embodies)\s+([A-Z][\w\s.-]+)", "contains", 0.90),
            (r"([A-Z][\w\s.-]+?)\s+(?:is a|is an|serves as a|acts as a)\s+([A-Z][\w\s.-]+)", "is_a", 0.85),
            (r"([A-Z][\w\s.-]+?)\s+(?:depends on|requires|prerequisite is)\s+([A-Z][\w\s.-]+)", "depends_on", 0.85),
            (r"([A-Z][\w\s.-]+?)\s+(?:contradicts|conflicts with|opposes)\s+([A-Z][\w\s.-]+)", "contradicts", 0.85),
            (r"([A-Z][\w\s.-]+?)\s+(?:is part of|subsystem of)\s+([A-Z][\w\s.-]+)", "part_of", 0.90),
        ]

        for pat, rel_type, base_conf in patterns:
            for match in re.finditer(pat, text, flags=re.IGNORECASE):
                src_raw = match.group(1).strip()
                tgt_raw = match.group(2).strip()
                src_clean = re.sub(r"^[^\w]+|[^\w]+$", "", src_raw)
                tgt_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tgt_raw)

                if len(src_clean) < 2 or len(tgt_clean) < 2:
                    continue
                if src_clean.lower() in ("it", "they", "this", "that", "which") or tgt_clean.lower() in ("it", "they", "this", "that", "which"):
                    continue

                e1 = self.add_entity(src_clean)
                e2 = self.add_entity(tgt_clean)
                extracted_entities.extend([e1, e2])

                actual_conf = base_conf if relation_class == "verified" else min(base_conf, 0.70)
                rel = self.add_relation(
                    e1["name"],
                    e2["name"],
                    rel_type,
                    confidence=actual_conf,
                    source=source,
                    relation_class=relation_class,
                )
                extracted_relations.append(rel)

        return extracted_entities, extracted_relations

    def detect_graph_contradictions(self, entity_name_or_id: str | None = None) -> list[dict[str, Any]]:
        """
        Detects contradictions in the knowledge graph:
        1. Explicit 'contradicts' edges between entities.
        2. Mutually exclusive relations (e.g. source -> target has both 'is_a' and 'contradicts').
        """
        contradictions = []
        with self._connect() as connection:
            if entity_name_or_id:
                ent = self.get_entity(entity_name_or_id)
                eid = ent["id"] if ent else None
                rows = connection.execute("""
                    SELECT r.*, s.name as source_name, t.name as target_name
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    WHERE r.relation_type = 'contradicts' AND (r.source_entity_id = ? OR r.target_entity_id = ?)
                """, (eid, eid)).fetchall()
            else:
                rows = connection.execute("""
                    SELECT r.*, s.name as source_name, t.name as target_name
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    WHERE r.relation_type = 'contradicts'
                """).fetchall()

            for r in rows:
                contradictions.append({
                    "type": "explicit_contradiction",
                    "source": r["source_name"],
                    "target": r["target_name"],
                    "confidence": r["confidence"],
                    "source_ref": r["source"],
                    "relation_class": r["relation_class"] if "relation_class" in r.keys() else "verified",
                })

            conflict_rows = connection.execute("""
                SELECT r1.source_entity_id, r1.target_entity_id, s.name as source_name, t.name as target_name,
                       r1.relation_type as rel1, r2.relation_type as rel2
                FROM knowledge_relations r1
                JOIN knowledge_relations r2 ON r1.source_entity_id = r2.source_entity_id AND r1.target_entity_id = r2.target_entity_id
                JOIN knowledge_entities s ON s.id = r1.source_entity_id
                JOIN knowledge_entities t ON t.id = r1.target_entity_id
                WHERE r1.relation_type != r2.relation_type
                  AND (r1.relation_type = 'contradicts' OR r2.relation_type = 'contradicts')
            """).fetchall()

            for cr in conflict_rows:
                contradictions.append({
                    "type": "mutually_exclusive_relations",
                    "source": cr["source_name"],
                    "target": cr["target_name"],
                    "relations": [cr["rel1"], cr["rel2"]],
                    "details": f"Entity '{cr['source_name']}' has conflicting relations '{cr['rel1']}' and '{cr['rel2']}' with '{cr['target_name']}'",
                })

        return contradictions

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

    # -------------------------------------------------------------------------
    # Cognitive Lifecycle, Expiration, and Conflict Resolution (Phase 4)
    # -------------------------------------------------------------------------

    def purge_expired(self, *, user_scope: str | None = None) -> int:
        """Purges memories that have exceeded their TTL / expiration timestamp."""
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            if user_scope:
                expired_rows = connection.execute(
                    "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ? AND user_scope = ?",
                    (now_iso, user_scope),
                ).fetchall()
            else:
                expired_rows = connection.execute(
                    "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now_iso,),
                ).fetchall()

            if not expired_rows:
                return 0

            expired_ids = [row["id"] for row in expired_rows]
            placeholders = ",".join("?" for _ in expired_ids)

            if self._fts_available:
                connection.execute(f"DELETE FROM memory_fts WHERE memory_id IN ({placeholders})", expired_ids)

            cursor = connection.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", expired_ids)
            purged = cursor.rowcount
            logger.info("Purged %d expired memories", purged)
            return purged

    @staticmethod
    def classify_fact_or_assumption(text: str) -> tuple[str, float]:
        """Distinguishes factual assertions from speculative assumptions based on linguistic markers."""
        lower = text.lower()
        assumption_markers = [
            "i think", "maybe", "probably", "might", "could be", "perhaps",
            "guess", "assume", "suppose", "not sure", "unconfirmed", "unverified",
            "believe",
        ]
        if any(marker in lower for marker in assumption_markers):
            return "assumption", 0.40

        fact_markers = [
            "i am", "my name is", "i live", "i stay", "i work", "confirmed",
            "rule:", "deadline is", "is equal to", "always", "never", "speed of light",
        ]
        if any(marker in lower for marker in fact_markers) or re.search(r"\b\d+[\d,.]*\b", text):
            return "fact", 0.95

        return "fact", 0.80

    def resolve_conflict(
        self,
        memory_id_to_supersede: str,
        new_content: str,
        *,
        source: str = "user_override",
        verified_by: str = "user_override",
        user_scope: str = "default_user",
        reason: str = "Explicit factual override",
    ) -> dict[str, Any]:
        """Resolves conflicting memory records by superseding the older record with proven newer facts."""
        old = self.get(memory_id_to_supersede, user_scope=user_scope)
        if old is None:
            raise KeyError(f"Unknown memory id: {memory_id_to_supersede}")

        replacement = self.add(
            new_content,
            source=source,
            memory_type=old.get("memory_type", "semantic"),
            confidence=1.0,
            verification_status="verified",
            verified_by=verified_by,
            privacy_classification=old.get("privacy_classification", "non_sensitive"),
            importance=old.get("importance", 0.8),
            user_scope=user_scope,
            fact_type="fact",
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE memories SET verification_status='superseded', updated_at=?, superseded_by=? WHERE id=?",
                (now, replacement["id"], memory_id_to_supersede),
            )
        self.link_memories(memory_id_to_supersede, replacement["id"], link_type="supersedes", notes=reason)
        return replacement

    def extract_and_remember(
        self,
        user_message: str,
        *,
        user_scope: str = "default_user",
        source: str = "user_conversation_turn",
    ) -> list[dict[str, Any]]:
        """Extracts meaningful user preferences, instructions, and personal facts from natural language."""
        stored: list[dict[str, Any]] = []
        msg_lower = user_message.lower().strip()

        # 1. User preferences (e.g. "I prefer dark mode", "I prefer concise code")
        pref_match = re.search(r"\bi\s+(?:prefer|like|love|favor)\s+(.+?)(?:\.|$|\band\b)", msg_lower)
        if pref_match:
            pref_content = f"User preference ({user_scope}): prefers {pref_match.group(1).strip()}"
            stored.append(
                self.add(
                    pref_content,
                    source=source,
                    memory_type="semantic",
                    confidence=0.90,
                    verification_status="verified",
                    verified_by="user_statement",
                    user_scope=user_scope,
                    fact_type="preference",
                    importance=0.85,
                )
            )

        # 2. Explicit instructions (e.g. "Always respond in bullet points", "Never use emojis")
        instr_match = re.search(r"\b(always|never|make sure to|do not)\s+(.+?)(?:\.|$)", msg_lower)
        if instr_match and not any(kw in msg_lower for kw in ["what", "how", "why"]):
            instr_content = f"User instruction ({user_scope}): {instr_match.group(1).strip()} {instr_match.group(2).strip()}"
            stored.append(
                self.add(
                    instr_content,
                    source=source,
                    memory_type="procedural",
                    confidence=0.95,
                    verification_status="verified",
                    verified_by="user_instruction",
                    user_scope=user_scope,
                    fact_type="instruction",
                    importance=0.90,
                )
            )

        # 3. Personal facts (e.g. "I live in Badlapur East", "I work at ABC")
        fact_match = re.search(r"\bi\s+(?:live at|live in|stay at|stay in|work at|work in|am based in)\s+(.+?)(?:\.|$)", msg_lower)
        if fact_match:
            fact_content = f"Personal location/fact ({user_scope}): {fact_match.group(0).strip()}"
            stored.append(
                self.add(
                    fact_content,
                    source=source,
                    memory_type="semantic",
                    confidence=0.95,
                    verification_status="verified",
                    verified_by="user_statement",
                    user_scope=user_scope,
                    fact_type="fact",
                    importance=0.95,
                )
            )

        return stored
