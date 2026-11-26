"""Persistent local episodic memory with source and verification metadata."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """SQLite-backed memory baseline; generated claims remain unverified by default."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = True
        with self._connect() as connection:
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded_by TEXT,
                    content_sha256 TEXT NOT NULL UNIQUE
                )
            """)
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _digest(content: str) -> str:
        normalized = " ".join(content.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

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
    ) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if privacy_classification not in {"non_sensitive", "personal", "sensitive"}:
            raise ValueError("privacy_classification must be non_sensitive, personal, or sensitive")
        if privacy_classification == "sensitive":
            raise ValueError("Sensitive memories are blocked until encrypted storage and access controls exist")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ValueError("confidence and importance must be in [0, 1]")
        if verification_status not in {"unverified", "verified"}:
            raise ValueError("verification_status must be unverified or verified")
        if verification_status == "verified" and (not source or not verified_by):
            raise ValueError("Verified memories require a source and verified_by")
        digest = self._digest(content)
        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM memories WHERE content_sha256 = ?", (digest,)).fetchone()
            if existing:
                if verification_status == "verified" and existing["verification_status"] != "verified":
                    connection.execute("""
                        UPDATE memories SET source=?, confidence=?, verification_status='verified',
                            verified_by=?, updated_at=? WHERE id=?
                    """, (source, confidence, verified_by, now, existing["id"]))
                    existing = connection.execute("SELECT * FROM memories WHERE id = ?", (existing["id"],)).fetchone()
                return self._record(existing)
            connection.execute("""
                INSERT INTO memories(id, memory_type, content, source, confidence,
                    verification_status, verified_by, privacy_classification,
                    importance, created_at, updated_at, superseded_by, content_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, (memory_id, memory_type, content, source, confidence,
                  verification_status, verified_by, privacy_classification,
                  importance, now, now, digest))
            if self._fts_available:
                connection.execute("INSERT INTO memory_fts(memory_id, content, source) VALUES (?, ?, ?)",
                                   (memory_id, content, source or ""))
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record(row)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._record(row) if row else None

    def search(self, query: str, *, limit: int = 10, verified_only: bool = False,
               source_contains: str | None = None) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        with self._connect() as connection:
            if self._fts_available:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                filter_sql = " AND m.verification_status = 'verified'" if verified_only else ""
                params: list[Any] = [expression]
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
            params: list[Any] = [f"%{term}%" for term in terms]
            verification = " AND verification_status = 'verified'" if verified_only else ""
            if source_contains:
                verification += " AND source LIKE ?"
                params.append(f"%{source_contains}%")
            rows = connection.execute(
                f"SELECT * FROM memories WHERE ({predicates}){verification} ORDER BY importance DESC, created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [self._record(row) for row in rows]

    def correct(self, memory_id: str, new_content: str, *, source: str, corrected_by: str) -> dict[str, Any]:
        old = self.get(memory_id)
        if old is None:
            raise KeyError(f"Unknown memory id: {memory_id}")
        replacement = self.add(new_content, source=source, memory_type=old["memory_type"],
                               confidence=1.0, verification_status="verified", verified_by=corrected_by,
                               privacy_classification=old["privacy_classification"],
                               importance=old["importance"])
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("UPDATE memories SET verification_status='superseded', updated_at=?, superseded_by=? WHERE id=?",
                               (now, replacement["id"], memory_id))
        return replacement

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            if self._fts_available:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

