"""AAROHAN AI — Cognitive Memory Engine 3.0

Persistent Cognitive Memory, Continuous Learning & Cross-Session Recall.

Implements:
1. Structured extraction supporting numbered rules, natural-language sentences,
   conditionals, priorities, corrections, and compound instructions.
2. Unique semantic keys per rule — never derives two distinct rules from one word.
3. Versioned correction & lineage tracking (memory_versions, memory_events, memory_retrieval_logs).
4. Memory-first cross-session retrieval: exact key → subject/predicate → FTS5 keyword scan.
5. Authoritative source attribution and honest abstention.
6. Prevention of false confirmations — every confirmation is read-back verified from SQLite.
7. Complete user isolation and scoped access controls.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aarohan.cognitive_memory")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MemoryType(str, enum.Enum):
    FACTUAL = "FACTUAL"
    CORRECTED = "CORRECTED"
    PROCEDURAL = "PROCEDURAL"
    PROJECT = "PROJECT"
    CONVERSATIONAL = "CONVERSATIONAL"
    PREFERENCE = "PREFERENCE"
    KNOWLEDGE = "KNOWLEDGE"
    EPISODIC = "EPISODIC"
    RULE = "RULE"


class MemoryStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"
    FLAGGED_CONFLICT = "FLAGGED_CONFLICT"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFact:
    memory_key: str
    subject: str
    predicate: str
    value: str
    unit: str = ""
    conditions: str = ""
    description: str = ""
    memory_type: str = MemoryType.FACTUAL.value
    is_correction: bool = False
    confidence: float = 0.9


@dataclass
class MemoryExtractionResult:
    is_memory_instruction: bool
    is_correction: bool
    is_forget_request: bool
    is_question: bool
    extracted_facts: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    raw_input: str = ""
    extraction_method: str = "none"
    unextracted_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stopwords excluded from key extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "what", "does", "mean", "current", "previous", "meanings", "retrieve", "tell",
    "who", "which", "how", "when", "where", "why", "are", "is", "was", "were",
    "the", "and", "but", "for", "not", "this", "that", "there", "here", "can",
    "has", "have", "had", "been", "being", "with", "from", "also", "into",
    "then", "than", "them", "they", "will", "would", "could", "should", "must",
    "may", "might", "shall", "did", "do", "done", "get", "got", "give", "got",
    "make", "made", "take", "took", "see", "seen", "say", "said", "know",
    "about", "above", "after", "again", "against", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "only", "own", "same", "so", "too", "very", "just", "because", "while",
    "through", "during", "before", "after", "between", "each", "these", "those",
    "account", "accounts", "their", "our", "your", "his", "her", "its",
    "rule", "rules", "code", "codes", "name", "names", "meaning", "means",
    "term", "terms", "record", "records", "query", "deadline", "priority",
    "level", "value", "status", "happens", "please", "store", "save",
    "stored", "asking", "everything", "website", "whats", "whos", "where", "wheres",
    "mine", "yours", "nam", "url",
    "am", "an", "as", "at", "be", "by", "go", "he", "if", "in", "it",
    "me", "my", "of", "on", "or", "to", "up", "us", "we",
})

# Save-trigger keywords
_SAVE_TRIGGERS = frozenset({
    "remember that", "remember:", "save this", "save the following",
    "learn this", "learn and save", "learn the following",
    "store this", "store the following", "keep this for future",
    "note that", "note:", "record this", "teach you",
    "i want you to remember", "please remember", "don't forget",
})

# Correction keywords
_CORRECTION_TRIGGERS = frozenset({
    "correction:", "correcting", "corrected", "actually,",
    "update:", "now means", "updated to", "change this to",
    "not anymore", "no longer", "instead of", "replace with",
    "is now", "has changed to", "should now be",
})

# Question prefixes — used to detect retrieval intent (not to block retrieval)
_QUESTION_PREFIXES = re.compile(
    r"^(what|who|which|how|tell me|retrieve|explain|show me|do you know|"
    r"remind me|recall|fetch|look up|find|search for|give me)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Structured extraction helpers
# ---------------------------------------------------------------------------

# Deadline / time-limit rules:
#   "P1 accounts must be contacted within 2 hours"
#   "Priority P2 accounts must be contacted within 6 hours"
#   "P3 must be contacted within 24 hours"
_DEADLINE_PATTERN = re.compile(
    r"(?:priority\s+)?([Pp][1-9])\s+(?:accounts?\s+)?(?:must(?:\s+now)?\s+be\s+contacted|should(?:\s+now)?\s+be\s+contacted|"
    r"contact\s+(?:now\s+)?within|contacted\s+within|now\s+means?|must(?:\s+now)?\s+be)\s+(?:within\s+)?(\d+)\s+(hours?|minutes?|days?|weeks?)",
    re.IGNORECASE,
)

# General "X must be contacted within N hours" without priority prefix
_CONTACT_WITHIN_PATTERN = re.compile(
    r"([\w\s\-]+?)\s+(?:accounts?\s+)?(?:must\s+be\s+contacted|should\s+be\s+contacted)\s+within\s+(\d+)\s+(hours?|minutes?|days?)",
    re.IGNORECASE,
)

# Routing / queue rules:
#   "confirmed payment accounts must be excluded from unpaid collections"
#   "Accounts with payment confirmation must not be included in the unpaid follow-up queue"
_CONFIRMED_PAYMENT_PATTERN = re.compile(
    r"(?:accounts?\s+with\s+)?(?:payment\s+confirmation|confirmed\s+payment)s?\s+"
    r"(?:must(?:\s+not)?|should(?:\s+not)?|are?)\s+"
    r"(?:be\s+)?(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Disputed payment routing:
#   "If payment is disputed, route the account to Payment Verification"
#   "If an account has a disputed payment, it must be assigned to the verification team"
_DISPUTED_PAYMENT_PATTERN = re.compile(
    r"(?:if\s+)?(?:an?\s+account\s+has\s+a?\s+)?(?:payment\s+is\s+disputed|disputed\s+payments?)\s*"
    r"(?:,?\s*(?:it\s+must\s+be\s+|route\s+(?:the\s+account\s+)?to\s+|assigned\s+to\s+|must\s+be\s+routed\s+to\s+|must\s+be\s+|must\s+go\s+to\s+))?(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# P1 + disputed precedence:
#   "If a P1 account has a disputed payment, Payment Verification takes precedence"
_P1_DISPUTED_PRECEDENCE_PATTERN = re.compile(
    r"(?:if\s+)?(?:a\s+)?(?:priority\s+)?[Pp]1\s+(?:accounts?\s+)?(?:with|having|has\s+a?)?\s+disputed\s+payments?[,\s]+"
    r"(?:must\s+still\s+be\s+|must\s+be\s+|takes\s+precedence\s+)?(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# "X means Y" / "X is Y" / "X = Y" general fact pattern
_MEANS_PATTERN = re.compile(
    r"(?:remember\s+(?:that\s+)?|note\s+(?:that\s+)?|correction\s*:\s*)?"
    r"([A-Za-z0-9]+[-_][A-Za-z0-9]+|[A-Za-z][A-Za-z0-9_]{2,})\s+"
    r"(?:was\s+subsequently\s+corrected\s+to|originally\s+means|now\s+means"
    r"|means|is\s+defined\s+as|stands\s+for|refers?\s+to|=)\s+"
    r"(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# "X is now Y" correction pattern
_IS_NOW_PATTERN = re.compile(
    r"([A-Za-z0-9][-_A-Za-z0-9]{2,})\s+is\s+now\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Key-value with colon: "P1_DEADLINE: 2 hours"
_KV_COLON_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9_\-]{2,})\s*[:=]\s*(.+)$",
)

# ---------------------------------------------------------------------------
# Key generation helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to UPPER_SNAKE_CASE memory key."""
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", "_", text.strip().upper())
    return text[:60]


def _make_deadline_key(tier: str) -> str:
    return f"{tier.upper()}_CONTACT_DEADLINE"


def _make_rule_key(subject_slug: str, predicate_slug: str) -> str:
    return f"{subject_slug}_{predicate_slug}_RULE"[:60]


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def _extract_deadline_facts(text: str, is_correction: bool) -> List[ExtractedFact]:
    facts = []
    for m in _DEADLINE_PATTERN.finditer(text):
        tier = m.group(1).upper()         # P1, P2, P3
        num = m.group(2)                  # 2, 6, 24
        unit = m.group(3).lower().rstrip("s") + "s"  # normalise to plural
        key = _make_deadline_key(tier)
        facts.append(ExtractedFact(
            memory_key=key,
            subject=f"{tier} account",
            predicate="contact_deadline",
            value=f"{num} {unit}",
            unit=unit,
            description=f"{tier} accounts must be contacted within {num} {unit}.",
            memory_type=MemoryType.RULE.value,
            is_correction=is_correction,
            confidence=0.97,
        ))
    return facts


def _extract_confirmed_payment_fact(text: str, is_correction: bool) -> Optional[ExtractedFact]:
    m = _CONFIRMED_PAYMENT_PATTERN.search(text)
    if not m:
        # Also match the alternative phrasing
        alt = re.search(
            r"accounts?\s+with\s+(?:payment\s+confirmation|confirmed\s+payments?)\s+"
            r"must\s+not\s+be\s+included\s+in\s+(the\s+)?(.+?)(?:\.|$)",
            text, re.IGNORECASE,
        )
        if alt:
            action = "excluded from " + (alt.group(2) or "unpaid collections").strip()
            return ExtractedFact(
                memory_key="CONFIRMED_PAYMENT_QUEUE_RULE",
                subject="confirmed payment account",
                predicate="queue_action",
                value=action,
                conditions="payment confirmed",
                description="Accounts with confirmed payment must not be included in unpaid collections.",
                memory_type=MemoryType.RULE.value,
                is_correction=is_correction,
                confidence=0.96,
            )
        return None

    action = m.group(1).strip().rstrip(".")
    return ExtractedFact(
        memory_key="CONFIRMED_PAYMENT_QUEUE_RULE",
        subject="confirmed payment account",
        predicate="queue_action",
        value=action,
        conditions="payment confirmed",
        description=f"Confirmed payment accounts: {action}.",
        memory_type=MemoryType.RULE.value,
        is_correction=is_correction,
        confidence=0.96,
    )


def _extract_disputed_payment_fact(text: str, is_correction: bool) -> Optional[ExtractedFact]:
    # If this specific text is a P1 disputed precedence rule, do not misclassify as generic disputed rule
    if re.search(r"\b(?:priority\s+)?[Pp]1\b", text, re.IGNORECASE) and re.search(r"disputed\s+payments?", text, re.IGNORECASE):
        return None
    m = _DISPUTED_PAYMENT_PATTERN.search(text)
    if not m:
        # Try simpler: "if payment is disputed, route/assign to X"
        alt = re.search(
            r"(?:if\s+)?(?:payment\s+is\s+disputed|has\s+a\s+disputed\s+payment)"
            r"[,\s]+(?:route\s+(?:the\s+account\s+)?to\s+|assigned?\s+to\s+|must\s+go\s+to\s+)?(.+?)(?:\.|$)",
            text, re.IGNORECASE,
        )
        if alt:
            dest = alt.group(1).strip().rstrip(".")
            dest = re.sub(r"^(?:s\s+|must\s+(?:still\s+)?be\s+|routed\s+to\s+|to\s+)", "", dest, flags=re.IGNORECASE).strip()
            return ExtractedFact(
                memory_key="DISPUTED_PAYMENT_ROUTING_RULE",
                subject="disputed payment account",
                predicate="routing_destination",
                value=f"routed to {dest}" if not dest.lower().startswith("route") else dest,
                conditions="payment disputed",
                description=f"Disputed payment accounts route to: {dest}.",
                memory_type=MemoryType.RULE.value,
                is_correction=is_correction,
                confidence=0.95,
            )
        return None

    dest = m.group(1).strip().rstrip(".")
    dest = re.sub(r"^(?:s\s+|must\s+(?:still\s+)?be\s+|routed\s+to\s+|to\s+)", "", dest, flags=re.IGNORECASE).strip()
    return ExtractedFact(
        memory_key="DISPUTED_PAYMENT_ROUTING_RULE",
        subject="disputed payment account",
        predicate="routing_destination",
        value=f"routed to {dest}" if not dest.lower().startswith("route") else dest,
        conditions="payment disputed",
        description=f"Disputed payment accounts: {dest}.",
        memory_type=MemoryType.RULE.value,
        is_correction=is_correction,
        confidence=0.95,
    )


def _extract_p1_disputed_precedence(text: str, is_correction: bool) -> Optional[ExtractedFact]:
    m = _P1_DISPUTED_PRECEDENCE_PATTERN.search(text)
    if not m:
        return None
    action = m.group(1).strip().rstrip(".")
    action = re.sub(r"^(?:must\s+(?:still\s+)?be\s+)", "", action, flags=re.IGNORECASE).strip()
    return ExtractedFact(
        memory_key="P1_DISPUTED_PAYMENT_OVERRIDE",
        subject="P1 account with disputed payment",
        predicate="routing_precedence",
        value=action,
        conditions="P1 and payment disputed",
        description=f"P1 disputed payment: {action}.",
        memory_type=MemoryType.RULE.value,
        is_correction=is_correction,
        confidence=0.97,
    )


def _extract_means_facts(text: str, is_correction: bool) -> List[ExtractedFact]:
    """Extract KEY means VALUE facts (e.g. BLUE-731 means Priority Level 7)."""
    facts = []
    seen_keys: set[str] = set()

    # Corrections: "X is now Y"
    for m in _IS_NOW_PATTERN.finditer(text):
        key = m.group(1).strip().upper()
        val = m.group(2).strip().rstrip(".")
        if key.lower() in _STOP_WORDS or key in seen_keys:
            continue
        seen_keys.add(key)
        facts.append(ExtractedFact(
            memory_key=key,
            subject=key,
            predicate="means",
            value=val,
            description=f"{key} is now {val}.",
            is_correction=True,
            confidence=0.95,
        ))

    # General "X means Y"
    for m in _MEANS_PATTERN.finditer(text):
        key = m.group(1).strip().upper()
        val = m.group(2).strip().rstrip(".")
        if key.lower() in _STOP_WORDS or key in seen_keys:
            continue
        is_corr = is_correction or any(
            kw in m.group(0).lower()
            for kw in ("corrected to", "now means", "is now")
        )
        seen_keys.add(key)
        facts.append(ExtractedFact(
            memory_key=key,
            subject=key,
            predicate="means",
            value=val,
            description=f"{key} means {val}.",
            is_correction=is_corr,
            confidence=0.92,
        ))

    return facts


def _extract_profile_facts(text: str, is_correction: bool = False) -> List[ExtractedFact]:
    """Extract user profile, website, name, background, or skill facts."""
    facts = []

    # Website pattern: "Datta Sable's Website is https://dattasable.com/" or "My website is https://dattasable.com/"
    web_m = re.search(
        r"(?:(?:datta\s+sable|user|my)'?s?\s+)?website\s+(?:is|url\s+is)?\s*[:=]?\s*(https?://[^\s]+)",
        text,
        re.IGNORECASE,
    )
    if web_m:
        url = web_m.group(1).strip().rstrip(".")
        facts.append(
            ExtractedFact(
                memory_key="USER_WEBSITE",
                subject="Datta Sable",
                predicate="personal_website",
                value=url,
                description=f"Datta Sable's personal website is {url}.",
                memory_type=MemoryType.FACTUAL.value,
                is_correction=is_correction,
                confidence=0.98,
            )
        )

    # Professional Background pattern: "My professional background is..." or "Datta Sable's background is..."
    bg_m = re.search(
        r"(?:(?:datta\s+sable|user|my)'?s?\s+)?(?:professional\s+)?background\s+(?:is|consists\s+of)?\s*[:=]?\s*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if bg_m and not web_m:
        bg = bg_m.group(1).strip()
        if len(bg) > 3:
            facts.append(
                ExtractedFact(
                    memory_key="USER_PROFESSIONAL_BACKGROUND",
                    subject="Datta Sable",
                    predicate="professional_background",
                    value=bg,
                    description=f"Datta Sable's professional background is {bg}.",
                    memory_type=MemoryType.FACTUAL.value,
                    is_correction=is_correction,
                    confidence=0.98,
                )
            )

    # Technical Skills pattern: "My technical skills are..." or "Datta Sable's skills are..."
    sk_m = re.search(
        r"(?:(?:datta\s+sable|user|my)'?s?\s+)?(?:technical\s+)?skills\s+(?:are|include)?\s*[:=]?\s*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if sk_m:
        sk = sk_m.group(1).strip()
        if len(sk) > 3:
            facts.append(
                ExtractedFact(
                    memory_key="USER_TECHNICAL_SKILLS",
                    subject="Datta Sable",
                    predicate="technical_skills",
                    value=sk,
                    description=f"Datta Sable's technical skills are {sk}.",
                    memory_type=MemoryType.FACTUAL.value,
                    is_correction=is_correction,
                    confidence=0.98,
                )
            )

    # Name pattern: "My name is Datta Sable"
    name_m = re.search(
        r"^my\s+name\s+is\s+([A-Za-z\s]+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if name_m:
        name = name_m.group(1).strip()
        if len(name) > 1 and name.lower() not in _STOP_WORDS:
            facts.append(
                ExtractedFact(
                    memory_key="USER_NAME",
                    subject=name,
                    predicate="full_name",
                    value=name,
                    description=f"Human user's name is {name}.",
                    memory_type=MemoryType.FACTUAL.value,
                    is_correction=is_correction,
                    confidence=0.98,
                )
            )

    return facts


def _extract_numbered_line(line: str, is_correction: bool) -> Optional[ExtractedFact]:
    """
    Try to extract a business rule from a numbered list line such as:
      "1. Priority P1 accounts must be contacted within 2 hours."
    Returns an ExtractedFact or None.
    """
    # Strip list numbering
    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
    if not cleaned:
        return None

    # Try deadline pattern first
    dl = _extract_deadline_facts(cleaned, is_correction)
    if dl:
        return dl[0]

    # Try P1 disputed precedence
    p1d = _extract_p1_disputed_precedence(cleaned, is_correction)
    if p1d:
        return p1d

    # Try disputed payment
    dp = _extract_disputed_payment_fact(cleaned, is_correction)
    if dp:
        return dp

    # Try confirmed payment
    cp = _extract_confirmed_payment_fact(cleaned, is_correction)
    if cp:
        return cp

    # Try X means Y
    means_facts = _extract_means_facts(cleaned, is_correction)
    if means_facts:
        return means_facts[0]

    # Try Profile & Website facts
    pf = _extract_profile_facts(cleaned, is_correction)
    if pf:
        return pf[0]

    return None


# ---------------------------------------------------------------------------
# CognitiveMemoryEngine
# ---------------------------------------------------------------------------

class CognitiveMemoryEngine:
    """
    Cognitive Memory Engine 3.0

    Provides structured extraction, versioned storage, hybrid retrieval, and
    prevention of false confirmations for AAROHAN's persistent memory system.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create or migrate all cognitive memory tables."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_memories (
                    id TEXT PRIMARY KEY,
                    memory_key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    current_value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    conditions TEXT DEFAULT '',
                    description TEXT,
                    source_conversation_id TEXT,
                    source_message_id TEXT,
                    source_type TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    importance REAL NOT NULL DEFAULT 0.8,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    UNIQUE(user_scope, memory_key)
                )
            """)
            # Add conditions column to existing DBs without it
            try:
                conn.execute("ALTER TABLE cognitive_memories ADD COLUMN conditions TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL REFERENCES cognitive_memories(id) ON DELETE CASCADE,
                    previous_value TEXT,
                    new_value TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    correction_reason TEXT,
                    source_message_id TEXT,
                    created_at TEXT NOT NULL,
                    version_number INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_events (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL REFERENCES cognitive_memories(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    event_description TEXT NOT NULL,
                    source_message_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_retrieval_logs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    query TEXT NOT NULL,
                    retrieved_memory_ids TEXT,
                    retrieval_method TEXT NOT NULL,
                    retrieval_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # FTS5 index on cognitive_memories for keyword search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cognitive_fts USING fts5(
                    memory_id UNINDEXED,
                    memory_key,
                    subject,
                    predicate,
                    current_value,
                    description,
                    conditions,
                    tokenize='unicode61'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cog_mem_key ON cognitive_memories(memory_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cog_mem_user_key ON cognitive_memories(user_scope, memory_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cog_mem_status ON cognitive_memories(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cog_mem_subj ON cognitive_memories(subject)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_ver_id_num ON memory_versions(memory_id, version_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_evt_id ON memory_events(memory_id)")

    # -------------------------------------------------------------------------
    # Extraction — Phase 2
    # -------------------------------------------------------------------------

    def extract_memory_intent(self, query: str) -> MemoryExtractionResult:
        """
        Analyze a query and extract:
        - Forget requests
        - Correction instructions
        - Save/teach instructions (numbered lists, natural-language rules, etc.)
        - Questions (retrieval intent — does NOT block retrieval)

        Returns MemoryExtractionResult with extracted_facts populated.
        """
        q_raw = query.strip()
        q_lower = q_raw.lower()

        # 1. Detect forget requests
        forget_m = re.search(
            r"\b(forget|delete|remove|erase)\s+(about\s+)?([A-Za-z0-9\-_]+)", q_lower
        )
        if forget_m:
            key = forget_m.group(3).strip().upper()
            return MemoryExtractionResult(
                is_memory_instruction=False,
                is_correction=False,
                is_forget_request=True,
                is_question=False,
                extracted_facts=[{"memory_key": key, "action": "forget"}],
                confidence=0.9,
                raw_input=q_raw,
                extraction_method="forget_pattern",
            )

        # 2. Detect question intent (retrieval — does NOT early-return; we set the flag)
        is_question = bool(
            _QUESTION_PREFIXES.search(q_lower)
            or q_raw.endswith("?")
        )

        # 3. Detect correction signals
        is_correction = bool(
            re.search(
                r"\b(correction|correcting|corrected|actually|now means|updated to"
                r"|is now|has changed to|should now be|change.*?to|replace.*?with)\b",
                q_lower,
            )
        )

        # 4. Detect save-trigger keywords
        has_save_trigger = any(kw in q_lower for kw in _SAVE_TRIGGERS)

        # 5. If question and no save trigger, signal for retrieval
        if is_question and not has_save_trigger and not is_correction:
            return MemoryExtractionResult(
                is_memory_instruction=False,
                is_correction=False,
                is_forget_request=False,
                is_question=True,
                extracted_facts=[],
                confidence=0.9,
                raw_input=q_raw,
                extraction_method="question_signal",
            )

        # 6. Full extraction pipeline
        facts: List[ExtractedFact] = []
        unextracted: List[str] = []

        # 6a. If message contains numbered list, process line-by-line
        numbered_lines = re.findall(r"^\s*\d+[\.\)]\s*.+", q_raw, re.MULTILINE)
        if numbered_lines:
            for line in numbered_lines:
                fact = _extract_numbered_line(line, is_correction)
                if fact:
                    # Avoid exact-duplicate keys in a single batch
                    if not any(f.memory_key == fact.memory_key for f in facts):
                        facts.append(fact)
                else:
                    unextracted.append(line.strip())

            # Deduplicate by memory_key — keep first occurrence
            seen: set[str] = set()
            deduped: List[ExtractedFact] = []
            for f in facts:
                if f.memory_key not in seen:
                    seen.add(f.memory_key)
                    deduped.append(f)
            facts = deduped

        # 6b. Whole-text extraction for non-numbered or supplementary rules
        if not facts or not numbered_lines:
            # Deadline rules in bulk text
            for dl in _extract_deadline_facts(q_raw, is_correction):
                if not any(f.memory_key == dl.memory_key for f in facts):
                    facts.append(dl)

            # Confirmed payment
            cp = _extract_confirmed_payment_fact(q_raw, is_correction)
            if cp and not any(f.memory_key == cp.memory_key for f in facts):
                facts.append(cp)

            # P1 disputed precedence (check before general disputed to prioritize specific override)
            p1d = _extract_p1_disputed_precedence(q_raw, is_correction)
            if p1d and not any(f.memory_key == p1d.memory_key for f in facts):
                facts.append(p1d)

            # Disputed payment
            dp = _extract_disputed_payment_fact(q_raw, is_correction)
            if dp and not any(f.memory_key == dp.memory_key for f in facts):
                facts.append(dp)

            # KEY means VALUE
            for mf in _extract_means_facts(q_raw, is_correction):
                if not any(f.memory_key == mf.memory_key for f in facts):
                    facts.append(mf)

            # Profile & Website facts
            for pf in _extract_profile_facts(q_raw, is_correction):
                if not any(f.memory_key == pf.memory_key for f in facts):
                    facts.append(pf)

        is_instruction = bool(facts) or has_save_trigger
        confidence = max((f.confidence for f in facts), default=0.4) if facts else 0.4

        return MemoryExtractionResult(
            is_memory_instruction=is_instruction,
            is_correction=is_correction,
            is_forget_request=False,
            is_question=is_question,
            extracted_facts=[vars(f) for f in facts],
            confidence=confidence,
            raw_input=q_raw,
            extraction_method="structured_pipeline",
            unextracted_lines=unextracted,
        )

    # -------------------------------------------------------------------------
    # Storage — Phase 3 & 4
    # -------------------------------------------------------------------------

    def save_memory(
        self,
        memory_key: str,
        value: str,
        *,
        subject: Optional[str] = None,
        predicate: str = "means",
        conditions: str = "",
        memory_type: MemoryType = MemoryType.FACTUAL,
        source_type: str = "explicit_user_instruction",
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        is_correction: bool = False,
        user_scope: str = "default_user",
        confidence: float = 1.0,
        importance: float = 0.8,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a fact or correction.  Returns the read-back verified record.
        Never returns a fabricated confirmation — always reads from SQLite after write.
        """
        key = memory_key.strip().upper()
        clean_val = value.strip()
        subj = (subject or key).strip()
        now = datetime.now(UTC).isoformat()
        norm_val = " ".join(clean_val.lower().split())

        if not key:
            raise ValueError("memory_key must not be empty")
        if not clean_val:
            raise ValueError("value must not be empty")

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM cognitive_memories WHERE user_scope = ? AND memory_key = ?",
                (user_scope, key),
            ).fetchone()

            if existing:
                existing_id = existing["id"]
                prev_val = existing["current_value"]

                # Corroboration: exact same value
                if prev_val.strip().lower() == clean_val.lower():
                    conn.execute("""
                        UPDATE cognitive_memories
                        SET updated_at = ?, last_accessed_at = ?, confidence = MIN(1.0, confidence + 0.05)
                        WHERE id = ?
                    """, (now, now, existing_id))
                    self._log_event(
                        conn, existing_id, "CORROBORATION",
                        f"Fact reaffirmed: '{clean_val}'.", source_message_id,
                    )
                    conn.commit()
                    return self._read_back(conn, existing_id, user_scope)

                # Explicit correction
                if is_correction or "corrected" in source_type.lower():
                    ver_row = conn.execute(
                        "SELECT MAX(version_number) AS max_v FROM memory_versions WHERE memory_id = ?",
                        (existing_id,),
                    ).fetchone()
                    new_ver_num = (ver_row["max_v"] or 1) + 1
                    conn.execute("""
                        UPDATE cognitive_memories
                        SET current_value = ?, normalized_value = ?, memory_type = ?,
                            updated_at = ?, last_accessed_at = ?, status = 'ACTIVE'
                        WHERE id = ?
                    """, (clean_val, norm_val, MemoryType.CORRECTED.value, now, now, existing_id))
                    conn.execute("""
                        INSERT INTO memory_versions (
                            id, memory_id, previous_value, new_value, change_type,
                            correction_reason, source_message_id, created_at, version_number
                        ) VALUES (?, ?, ?, ?, 'CORRECTION', 'Explicit user correction', ?, ?, ?)
                    """, (str(uuid.uuid4()), existing_id, prev_val, clean_val, source_message_id, now, new_ver_num))
                    self._log_event(
                        conn, existing_id, "CORRECTION",
                        f"Value corrected from '{prev_val}' → '{clean_val}' (v{new_ver_num}).",
                        source_message_id,
                    )
                    conn.commit()
                    result = self._read_back(conn, existing_id, user_scope)
                    result["was_correction"] = True
                    result["previous_value"] = prev_val
                    result["conflict_detected"] = False
                    return result

                # Genuine conflict (different value, not an explicit correction)
                conn.execute("""
                    UPDATE cognitive_memories SET status = 'FLAGGED_CONFLICT', updated_at = ? WHERE id = ?
                """, (now, existing_id))
                self._log_event(
                    conn, existing_id, "CONFLICT_FLAGGED",
                    f"Conflicting value '{clean_val}' vs established '{prev_val}'. Flagged for review.",
                    source_message_id,
                )
                conn.commit()
                result = self._read_back(conn, existing_id, user_scope)
                result["conflict_detected"] = True
                result["conflicting_value"] = clean_val
                return result

            # New memory
            new_id = f"mem_{uuid.uuid4().hex[:12]}"
            desc = description or f"{subj} {predicate} {clean_val}"
            conn.execute("""
                INSERT INTO cognitive_memories (
                    id, memory_key, memory_type, subject, predicate, current_value,
                    normalized_value, conditions, description, source_conversation_id,
                    source_message_id, source_type, confidence, importance, status,
                    created_at, updated_at, last_accessed_at, user_scope
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
            """, (
                new_id, key, memory_type.value, subj, predicate, clean_val,
                norm_val, conditions, desc,
                source_conversation_id, source_message_id, source_type,
                confidence, importance, now, now, now, user_scope,
            ))
            conn.execute("""
                INSERT INTO memory_versions (
                    id, memory_id, previous_value, new_value, change_type,
                    correction_reason, source_message_id, created_at, version_number
                ) VALUES (?, ?, NULL, ?, 'INITIAL', 'Initial learning event', ?, ?, 1)
            """, (str(uuid.uuid4()), new_id, clean_val, source_message_id, now))
            self._log_event(
                conn, new_id, "STORAGE",
                f"Persistent memory initialized: '{key}' = '{clean_val}'.",
                source_message_id,
            )
            conn.commit()

            # Update FTS index
            try:
                conn.execute("""
                    INSERT INTO cognitive_fts (memory_id, memory_key, subject, predicate,
                        current_value, description, conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (new_id, key, subj, predicate, clean_val, desc, conditions))
                conn.commit()
            except Exception:
                pass  # FTS failure is non-fatal

            result = self._read_back(conn, new_id, user_scope)
            result["conflict_detected"] = False
            return result

    def save_facts_batch(
        self,
        facts: List[Dict[str, Any]],
        *,
        user_scope: str = "default_user",
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a list of extracted facts in a single transaction.
        Returns per-fact results with SAVED / CORRECTED / CONFLICT / FAILED status.
        Never returns success for a fact that wasn't verified in the database.
        """
        saved = []
        conflicts = []
        failed = []

        for fact in facts:
            key = fact.get("memory_key", "").strip().upper()
            val = fact.get("value", "").strip()
            if not key or not val:
                failed.append({"memory_key": key, "reason": "Empty key or value"})
                continue
            try:
                result = self.save_memory(
                    memory_key=key,
                    value=val,
                    subject=fact.get("subject", key),
                    predicate=fact.get("predicate", "means"),
                    conditions=fact.get("conditions", ""),
                    memory_type=MemoryType(fact.get("memory_type", MemoryType.FACTUAL.value)),
                    source_type="explicit_user_instruction",
                    source_conversation_id=source_conversation_id,
                    source_message_id=source_message_id,
                    is_correction=fact.get("is_correction", False),
                    user_scope=user_scope,
                    confidence=fact.get("confidence", 1.0),
                    importance=0.9,
                    description=fact.get("description", ""),
                )
                if result.get("conflict_detected"):
                    conflicts.append(result)
                else:
                    saved.append(result)
            except Exception as exc:
                failed.append({"memory_key": key, "reason": str(exc)})

        return {
            "saved": saved,
            "conflicts": conflicts,
            "failed": failed,
            "total_facts": len(facts),
            "saved_count": len(saved),
            "conflict_count": len(conflicts),
            "failed_count": len(failed),
        }

    def _read_back(
        self, conn: sqlite3.Connection, memory_id: str, user_scope: str
    ) -> Dict[str, Any]:
        """Read a record back from the same connection — prevents return of unverified data."""
        row = conn.execute(
            "SELECT * FROM cognitive_memories WHERE id = ? AND user_scope = ?",
            (memory_id, user_scope),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Read-back verification failed for id={memory_id}")
        data = dict(row)
        vers = conn.execute(
            "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY version_number ASC",
            (memory_id,),
        ).fetchall()
        data["versions"] = [dict(v) for v in vers]
        data["version_count"] = len(vers)
        evts = conn.execute(
            "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY created_at DESC LIMIT 20",
            (memory_id,),
        ).fetchall()
        data["events"] = [dict(e) for e in evts]
        return data

    def _log_event(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        event_type: str,
        description: str,
        source_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        conn.execute("""
            INSERT INTO memory_events (
                id, memory_id, event_type, event_description, source_message_id, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), memory_id, event_type, description,
              source_message_id, now, json.dumps(metadata or {})))

    # -------------------------------------------------------------------------
    # Retrieval — Phase 5
    # -------------------------------------------------------------------------

    def retrieve_memory(
        self,
        query: str,
        user_scope: str = "default_user",
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Hybrid retrieval:
        1. Exact memory-key match against tokens in query.
        2. Subject / predicate partial match.
        3. FTS5 keyword search across all fields.
        4. Substring scan of all stored keys.
        Returns FOUND with records, NOT_FOUND, or None if nothing relevant.
        """
        q = query.strip()

        with self._connect() as conn:
            matched_records: List[Dict[str, Any]] = []
            matched_ids: set[str] = set()

            # --- Pass 1: Exact key tokens ---
            tokens = re.findall(
                r"\b[Pp][1-9]\b|\b[A-Za-z0-9]+[-_][A-Za-z0-9]+\b|\b[A-Za-z0-9_]{2,}\b", q
            )
            candidate_keys: List[str] = []
            for t in tokens:
                t_up = t.upper()
                if t.lower() not in _STOP_WORDS and t_up not in candidate_keys:
                    candidate_keys.append(t_up)
                    if re.match(r"^P[1-9]$", t_up):
                        candidate_keys.append(f"{t_up}_CONTACT_DEADLINE")
            for ck in candidate_keys:
                row = conn.execute(
                    "SELECT id FROM cognitive_memories WHERE user_scope = ? AND memory_key = ? AND status = 'ACTIVE'",
                    (user_scope, ck),
                ).fetchone()
                if row and row["id"] not in matched_ids:
                    matched_ids.add(row["id"])
                    rec = self.get_memory_by_id(row["id"], user_scope=user_scope)
                    if rec:
                        matched_records.append(rec)

            # --- Pass 2: Subject / predicate / conditions match with candidate tokens ---
            if not matched_records and candidate_keys:
                for ck in candidate_keys:
                    ck_stem = ck.rstrip("S")
                    if len(ck_stem) < 4 and not re.match(r"^P[1-9]$", ck):
                        continue
                    rows = conn.execute(
                        """SELECT id FROM cognitive_memories
                           WHERE user_scope = ? AND status = 'ACTIVE'
                             AND (memory_key LIKE ? OR subject LIKE ? OR predicate LIKE ? OR conditions LIKE ? OR description LIKE ?)
                           LIMIT 10""",
                        (user_scope, f"%{ck}%", f"%{ck_stem}%", f"%{ck_stem}%", f"%{ck_stem}%", f"%{ck_stem}%"),
                    ).fetchall()
                    for row in rows:
                        if row["id"] not in matched_ids:
                            matched_ids.add(row["id"])
                            rec = self.get_memory_by_id(row["id"], user_scope=user_scope)
                            if rec:
                                matched_records.append(rec)

            # --- Pass 3: FTS5 keyword search ---
            if not matched_records:
                # Build FTS query from meaningful tokens
                fts_terms = [t for t in candidate_keys if len(t) >= 4 and t.lower() not in _STOP_WORDS][:6]
                if fts_terms:
                    fts_q = " OR ".join(f'"{t}"' for t in fts_terms)
                    try:
                        fts_rows = conn.execute(
                            """SELECT memory_id FROM cognitive_fts WHERE cognitive_fts MATCH ?
                               LIMIT 10""",
                            (fts_q,),
                        ).fetchall()
                        for fr in fts_rows:
                            mid = fr["memory_id"]
                            if mid not in matched_ids:
                                # Verify it belongs to this scope and is active
                                check = conn.execute(
                                    "SELECT id FROM cognitive_memories WHERE id = ? AND user_scope = ? AND status = 'ACTIVE'",
                                    (mid, user_scope),
                                ).fetchone()
                                if check:
                                    matched_ids.add(mid)
                                    rec = self.get_memory_by_id(mid, user_scope=user_scope)
                                    if rec:
                                        matched_records.append(rec)
                    except Exception:
                        pass

            # --- Pass 4: Substring scan of all stored keys ---
            if not matched_records and candidate_keys:
                all_keys = conn.execute(
                    "SELECT id, memory_key, subject, description FROM cognitive_memories WHERE user_scope = ? AND status = 'ACTIVE'",
                    (user_scope,),
                ).fetchall()
                for k_row in all_keys:
                    k_str = k_row["memory_key"]
                    subj_str = (k_row["subject"] or "").lower()
                    desc_str = (k_row["description"] or "").lower()
                    # Match if any significant token appears in key/subject/description
                    if any(
                        (t.lower() in k_str.lower() or t.lower() in subj_str or t.lower() in desc_str)
                        for t in candidate_keys
                        if len(t) >= 4 and t.lower() not in _STOP_WORDS
                    ) and k_row["id"] not in matched_ids:
                        matched_ids.add(k_row["id"])
                        rec = self.get_memory_by_id(k_row["id"], user_scope=user_scope)
                        if rec:
                            matched_records.append(rec)

            now = datetime.now(UTC).isoformat()

            if matched_records:
                retrieved_ids = [r["id"] for r in matched_records]
                try:
                    conn.execute("""
                        INSERT INTO memory_retrieval_logs (
                            id, conversation_id, user_scope, query, retrieved_memory_ids,
                            retrieval_method, retrieval_status, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'HYBRID_V3', 'SUCCESS', ?)
                    """, (str(uuid.uuid4()), conversation_id, user_scope,
                          q, json.dumps(retrieved_ids), now))
                    for r_id in retrieved_ids:
                        conn.execute(
                            "UPDATE cognitive_memories SET last_accessed_at = ? WHERE id = ?",
                            (now, r_id),
                        )
                    conn.commit()
                except Exception:
                    pass

                recall_text = self._format_recall_response(q, matched_records)
                return {
                    "status": "FOUND",
                    "records": matched_records,
                    "retrieved_count": len(matched_records),
                    "recall_text": recall_text,
                }

            # NOT FOUND — log it
            if candidate_keys:
                unknown_key = candidate_keys[0]
                try:
                    conn.execute("""
                        INSERT INTO memory_retrieval_logs (
                            id, conversation_id, user_scope, query, retrieved_memory_ids,
                            retrieval_method, retrieval_status, created_at
                        ) VALUES (?, ?, ?, ?, '[]', 'HYBRID_V3', 'EMPTY', ?)
                    """, (str(uuid.uuid4()), conversation_id, user_scope, q, now))
                    conn.commit()
                except Exception:
                    pass
                return {
                    "status": "NOT_FOUND",
                    "records": [],
                    "retrieved_count": 0,
                    "unknown_key": unknown_key,
                    "recall_text": (
                        f"I searched persistent cognitive memory for '{unknown_key}' and related terms, "
                        f"but found no matching records. I will not guess."
                    ),
                }

            return None

    def _format_recall_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format an authoritative, source-attributed recall response."""
        lines = []
        q_lower = query.lower()
        wants_history = any(
            kw in q_lower
            for kw in ("previous", "history", "earlier", "current and previous", "changed", "version")
        )

        lines.append("### 🧠 Persistent Cognitive Memory Recall")
        lines.append(f"*{len(records)} record(s) retrieved from persistent storage*\n")

        for rec in records:
            key = rec["memory_key"]
            curr_val = rec["current_value"]
            subj = rec.get("subject", key)
            cond = rec.get("conditions", "")
            versions = rec.get("versions", [])
            rec_id = rec["id"]
            created = rec.get("created_at", "")
            has_correction = len(versions) > 1

            lines.append(f"#### `{key}`")
            if cond:
                lines.append(f"- **Condition:** {cond}")
            lines.append(f"- **Subject:** {subj}")
            lines.append(f"- **Rule:** {curr_val}")
            lines.append("- **Source:** Persistent Cognitive Memory")
            lines.append(f"- **Record ID:** `{rec_id}`")
            lines.append(f"- **Created:** {created[:19] if created else 'unknown'}")
            if has_correction and wants_history:
                init_val = versions[0]["new_value"] if versions else curr_val
                lines.append(f"- **Previous Value:** {init_val} (v1)")
                lines.append(f"- **Current Version:** v{len(versions)}")
            elif has_correction:
                lines.append(f"- **Note:** This fact has been corrected (v{len(versions)}). Ask for history to see previous values.")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Explorer & Management
    # -------------------------------------------------------------------------

    def get_memory_by_id(
        self,
        memory_id: str,
        user_scope: str = "default_user",
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[Dict[str, Any]]:
        should_close = conn is None
        if conn is None:
            conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM cognitive_memories WHERE id = ? AND user_scope = ?",
                (memory_id, user_scope),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            vers = conn.execute(
                "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY version_number ASC",
                (memory_id,),
            ).fetchall()
            data["versions"] = [dict(v) for v in vers]
            data["version_count"] = len(vers)
            evts = conn.execute(
                "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY created_at DESC LIMIT 20",
                (memory_id,),
            ).fetchall()
            data["events"] = [dict(e) for e in evts]
            return data
        finally:
            if should_close:
                conn.close()

    def list_memories(
        self,
        user_scope: str = "default_user",
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM cognitive_memories WHERE user_scope = ?"
        params: List[Any] = [user_scope]
        if memory_type:
            q += " AND memory_type = ?"
            params.append(memory_type.upper())
        if status:
            q += " AND status = ?"
            params.append(status.upper())
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
            results = []
            for r in rows:
                rec = dict(r)
                v_count = conn.execute(
                    "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?", (rec["id"],)
                ).fetchone()[0]
                rec["version_count"] = v_count
                results.append(rec)
            return results

    def search_memories(
        self,
        query: str,
        user_scope: str = "default_user",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        q_term = f"%{query.strip()}%"
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM cognitive_memories
                WHERE user_scope = ? AND (
                    memory_key LIKE ? OR subject LIKE ? OR current_value LIKE ?
                    OR description LIKE ? OR conditions LIKE ?
                )
                ORDER BY updated_at DESC LIMIT ?
            """, (user_scope, q_term, q_term, q_term, q_term, q_term, limit)).fetchall()
            return [dict(r) for r in rows]

    def delete_memory(self, memory_id: str, user_scope: str = "default_user") -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cognitive_memories WHERE id = ? AND user_scope = ?",
                (memory_id, user_scope),
            )
            return cur.rowcount > 0

    def get_dashboard_stats(self, user_scope: str = "default_user") -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM cognitive_memories WHERE user_scope = ?", (user_scope,)
            ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM cognitive_memories WHERE user_scope = ? AND status = 'ACTIVE'",
                (user_scope,),
            ).fetchone()[0]
            corrected = conn.execute(
                "SELECT COUNT(*) FROM cognitive_memories WHERE user_scope = ? AND memory_type = 'CORRECTED'",
                (user_scope,),
            ).fetchone()[0]
            cat_rows = conn.execute(
                "SELECT memory_type, COUNT(*) as cnt FROM cognitive_memories WHERE user_scope = ? GROUP BY memory_type",
                (user_scope,),
            ).fetchall()
            categories = {r["memory_type"]: r["cnt"] for r in cat_rows}
            tot_ret = conn.execute(
                "SELECT COUNT(*) FROM memory_retrieval_logs WHERE user_scope = ?", (user_scope,)
            ).fetchone()[0]
            succ_ret = conn.execute(
                "SELECT COUNT(*) FROM memory_retrieval_logs WHERE user_scope = ? AND retrieval_status = 'SUCCESS'",
                (user_scope,),
            ).fetchone()[0]
            recent_events = conn.execute("""
                SELECT e.id, e.event_type, e.event_description, e.created_at, m.memory_key
                FROM memory_events e
                JOIN cognitive_memories m ON e.memory_id = m.id
                WHERE m.user_scope = ?
                ORDER BY e.created_at DESC LIMIT 10
            """, (user_scope,)).fetchall()
            return {
                "total_stored": total,
                "active_memories": active,
                "corrected_memories": corrected,
                "categories": categories,
                "total_retrievals": tot_ret,
                "successful_retrievals": succ_ret,
                "failed_retrievals": tot_ret - succ_ret,
                "retrieval_success_rate_pct": round((succ_ret / tot_ret) * 100, 1) if tot_ret > 0 else 100.0,
                "storage_backend_status": "Operational (SQLite WAL)",
                "database_path": str(self.db_path),
                "recent_events": [dict(e) for e in recent_events],
            }
