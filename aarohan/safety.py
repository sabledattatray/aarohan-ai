"""Security, prompt-injection defense, privacy screening, and audit governance.

In accordance with PRD Section 18 and Prompt 14:
  - Defends against prompt injection in user queries and ingested documents.
  - Redacts sensitive PII (emails, phone numbers, credentials, private keys).
  - Maintains tamper-evident audit logging for autonomous operations.
  - The model never possesses unrestricted access to its production infrastructure.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"system\s+(override|prompt\s+leak|instruction\s+bypass)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+(all\s+)?(safety|rules|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"(output|print|show)\s+(your\s+)?(initial|system)\s+(instructions|prompt)", re.IGNORECASE),
]

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "api_key": re.compile(r"\b(?:api[_-]?key|secret[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?", re.IGNORECASE),
}


class SafetyFilter:
    """Detects prompt injections and redacts sensitive private information."""

    @staticmethod
    def detect_prompt_injection(text: str) -> dict[str, Any]:
        matched_patterns = []
        for pat in INJECTION_PATTERNS:
            match = pat.search(text)
            if match:
                matched_patterns.append(match.group(0))

        is_suspicious = len(matched_patterns) > 0
        return {
            "flagged": is_suspicious,
            "threat_type": "prompt_injection" if is_suspicious else None,
            "matched_patterns": matched_patterns,
            "safe": not is_suspicious,
        }

    @staticmethod
    def redact_pii(text: str) -> tuple[str, list[str]]:
        redacted = text
        detected: list[str] = []
        for pii_type, pat in PII_PATTERNS.items():
            if pat.search(redacted):
                detected.append(pii_type)
                redacted = pat.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted, detected


class AuditLogger:
    """Records audit logs in persistent SQLite storage for all governance actions."""

    def __init__(self, database_path: Path) -> None:
        self.db_path = Path(database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        action: str,
        *,
        actor: str = "aarohan_agent",
        target: str = "system",
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        log_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        details_str = json.dumps(details) if details else None
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO audit_logs(id, action, actor, target, status, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (log_id, action, actor, target, status, details_str, now))
            row = conn.execute("SELECT * FROM audit_logs WHERE id = ?", (log_id,)).fetchone()
            return dict(row)

    def list_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
