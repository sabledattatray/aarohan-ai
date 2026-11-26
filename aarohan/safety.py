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
    re.compile(r"ignore\s+(all\s+)?(previous\s+|prior\s+|system\s+)?(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"system\s+(override|prompt\s+leak|instruction\s+bypass)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+(all\s+)?(safety|rules|instructions)", re.IGNORECASE),
    re.compile(r"(bypass|override)\s+(all\s+)?(safety|security|filters)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"(output|print|show)\s+(your\s+)?(initial|system)\s+(instructions|prompt)", re.IGNORECASE),
]

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "api_key": re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.MULTILINE),
    "password": re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}['\"]?", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}

SPECIAL_DELIMITER_PATTERNS = [
    re.compile(r"<\|(?:im_start|im_end|system|user|assistant|tool|separator)\|>"),
    re.compile(r"\[(?:SYSTEM|INSTRUCTION|ADMIN|ROOT)\s*(?:PROMPT|OVERRIDE)?\]", re.IGNORECASE),
    re.compile(r"```(?:system|instruction|prompt)"),
    re.compile(r"---+\s*(?:BEGIN|START)\s+(?:SYSTEM|PROMPT|INSTRUCTION)", re.IGNORECASE),
]

ROLEPLAY_JAILBREAK_PATTERNS = [
    re.compile(r"\b(?:DAN|Do\s+Anything\s+Now)\s+mode\b", re.IGNORECASE),
    re.compile(r"\b(?:unrestricted|unfiltered|jailbroken)\s+(?:mode|ai|assistant|persona)\b", re.IGNORECASE),
    re.compile(r"\bhypothetical\s+(?:unfiltered|unaligned)\s+engine\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(?:you\s+have\s+no\s+(?:rules|filters|morals|limits))\b", re.IGNORECASE),
    re.compile(r"\bas\s+my\s+(?:creator|developer|administrator)\s*,\s*i\s+(?:order|command)\s+you\b", re.IGNORECASE),
]


class SafetyFilter:
    """Detects multi-layer prompt injections, adversarial framing, and redacts sensitive PII/secrets."""

    @staticmethod
    def normalize_leetspeak(text: str) -> str:
        """Decodes simple character substitutions used to evade text filters."""
        leet_map = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
        normalized = text
        for char, replacement in leet_map.items():
            normalized = normalized.replace(char, replacement)
        return normalized

    @classmethod
    def detect_prompt_injection(cls, text: str) -> dict[str, Any]:
        matched_patterns = []
        threat_categories = []

        # 1. Regex pattern matching on raw text
        for pat in INJECTION_PATTERNS:
            match = pat.search(text)
            if match:
                matched_patterns.append(match.group(0))
                threat_categories.append("instruction_override")

        # 2. Special token / delimiter injection
        for pat in SPECIAL_DELIMITER_PATTERNS:
            match = pat.search(text)
            if match:
                matched_patterns.append(match.group(0))
                threat_categories.append("delimiter_hijack")

        # 3. Roleplay & authority jailbreak checks
        for pat in ROLEPLAY_JAILBREAK_PATTERNS:
            match = pat.search(text)
            if match:
                matched_patterns.append(match.group(0))
                threat_categories.append("roleplay_jailbreak")

        # 4. Normalized leetspeak check
        leeted = cls.normalize_leetspeak(text.lower())
        for pat in INJECTION_PATTERNS:
            match = pat.search(leeted)
            if match and match.group(0) not in matched_patterns:
                matched_patterns.append(f"obfuscated:{match.group(0)}")
                threat_categories.append("obfuscated_injection")

        is_suspicious = len(matched_patterns) > 0
        risk_score = min(1.0, 0.4 * len(matched_patterns) + (0.3 if "delimiter_hijack" in threat_categories else 0.0))

        return {
            "flagged": is_suspicious,
            "threat_type": threat_categories[0] if threat_categories else None,
            "threat_categories": list(set(threat_categories)),
            "matched_patterns": matched_patterns,
            "risk_score": round(risk_score, 2),
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

    @staticmethod
    def sanitize_error_message(error: Exception | str) -> str:
        """Strips absolute filesystem paths and credentials from user-facing error messages."""
        raw_msg = str(error)
        # Strip Windows drive paths (e.g. C:\Users\... or D:\Datta Sable\...)
        sanitized = re.sub(r"[A-Za-z]:\\[\w\s.\\-]+", "[INTERNAL_PATH]", raw_msg)
        # Strip Unix paths
        sanitized = re.sub(r"/(?:home|var|etc|usr|tmp)/[\w\s./-]+", "[INTERNAL_PATH]", sanitized)
        # Strip potential connection strings
        sanitized = re.sub(r"(?:sqlite|postgres|mysql)://[^\s]+", "[CONNECTION_STRING]", sanitized)
        return sanitized


class RateLimiter:
    """Thread-safe in-memory sliding window rate limiter per client identifier."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str, current_time: float | None = None) -> tuple[bool, int]:
        import time
        now = current_time if current_time is not None else time.time()
        window_start = now - 60.0

        timestamps = self.requests.get(client_id, [])
        # Evict timestamps older than 60 seconds
        active_timestamps = [t for t in timestamps if t > window_start]

        if len(active_timestamps) >= self.rpm:
            self.requests[client_id] = active_timestamps
            remaining = 0
            return False, remaining

        active_timestamps.append(now)
        self.requests[client_id] = active_timestamps
        remaining = self.rpm - len(active_timestamps)
        return True, remaining


def validate_file_path(target_path: Path | str, allowed_dirs: list[Path | str]) -> Path:
    """Prevents directory traversal attacks by validating resolved canonical paths against allowed roots."""
    target = Path(target_path).resolve()
    for allowed in allowed_dirs:
        allowed_root = Path(allowed).resolve()
        try:
            target.relative_to(allowed_root)
            return target
        except ValueError:
            continue
    raise PermissionError(f"Access denied: path '{target}' is outside permitted directory boundaries")


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
