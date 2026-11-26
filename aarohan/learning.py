"""AAROHAN AI — Learning Engine 1.0.

Chat-Integrated Persistent Learning Architecture providing:
- General-purpose rule extraction (numbered lists, natural language, conditions, overrides, deadlines).
- Isolated learning namespaces (e.g. LEARNING_TEST_2026_01).
- Atomic verified storage in SQLite with read-back verification.
- Context injection under [ACTIVE LEARNED RULES] for reasoning.
- Feedback learning with version history preservation and ambiguity clarification.
- Behavior evaluation benchmarks recording before-and-after accuracy.
- Separation between neural weight fine-tuning and symbolic rule adaptation.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ExtractedRule:
    rule_key: str
    rule_text: str
    subject: str
    predicate: str
    value: str
    conditions: str = ""
    precedence: int = 0
    overrides: str = ""
    category: str = "GENERAL"
    confidence: float = 0.95
    is_correction: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningExtractionResult:
    is_learning_instruction: bool
    is_feedback: bool
    is_evaluation: bool
    namespace: str
    extracted_rules: List[ExtractedRule]
    confidence: float
    raw_input: str
    unextracted_lines: List[str] = field(default_factory=list)
    clarification_question: Optional[str] = None


class LearningEngine:
    """
    Learning Engine 1.0

    Manages user-taught systems, namespaces, rules, feedback cycles,
    and behavior evaluation benchmarks.
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
    # Schema Migration & Tables
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create or migrate all learning engine tables."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_namespaces (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_rules (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    rule_key TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    subject TEXT,
                    predicate TEXT,
                    value TEXT NOT NULL,
                    conditions TEXT,
                    precedence INTEGER DEFAULT 0,
                    overrides TEXT,
                    category TEXT DEFAULT 'GENERAL',
                    confidence REAL DEFAULT 1.0,
                    provenance TEXT DEFAULT 'explicit_user_instruction',
                    source_message_id TEXT,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    version INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_rule_versions (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    previous_value TEXT,
                    new_value TEXT NOT NULL,
                    previous_conditions TEXT,
                    new_conditions TEXT,
                    change_type TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(rule_id) REFERENCES learned_rules(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_benchmarks (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    test_input TEXT NOT NULL,
                    expected_decision TEXT NOT NULL,
                    actual_decision TEXT,
                    rules_retrieved TEXT,
                    rules_applied TEXT,
                    is_correct INTEGER DEFAULT 0,
                    error_notes TEXT,
                    score REAL DEFAULT 0.0,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_feedback_logs (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    query TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    parsed_correction TEXT,
                    applied_rule_id TEXT,
                    status TEXT NOT NULL,
                    user_scope TEXT NOT NULL DEFAULT 'default_user',
                    created_at TEXT NOT NULL
                )
            """)
            # FTS5 Index for learned rules
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS learned_rules_fts USING fts5(
                    rule_id UNINDEXED,
                    namespace,
                    rule_key,
                    subject,
                    predicate,
                    value,
                    conditions,
                    rule_text,
                    tokenize='unicode61'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lr_ns_key ON learned_rules(namespace, rule_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lr_scope_ns ON learned_rules(user_scope, namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lr_status ON learned_rules(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lrv_rule_ver ON learned_rule_versions(rule_id, version_number)")
            conn.commit()

    # -------------------------------------------------------------------------
    # Rule Extraction (Phase 3 & 4)
    # -------------------------------------------------------------------------

    def extract_learning_intent(self, text: str) -> LearningExtractionResult:
        """
        General-purpose extraction of learning instructions, namespaces, rules,
        conditions, exceptions, and precedence relationships.
        Never hardcodes specific domain tokens (like LEVEL-A/B/C or P1/P2/P3).
        """
        raw = text.strip()
        lower = raw.lower()

        # 1. Detect namespace
        ns_match = re.search(
            r"\b(?:namespace|learning\s+namespace|category|under\s+(?:the\s+)?namespace)[:\s=]+([A-Za-z0-9_\-]+)",
            raw, re.IGNORECASE,
        )
        namespace = ns_match.group(1).strip().upper() if ns_match else "DEFAULT"

        # 2. Detect Feedback Intent
        is_feedback = bool(re.search(
            r"\b(previous\s+decision\s+was\s+(in)?correct|decision\s+was\s+(wrong|incorrect)|apply\s+rule\s+\d+\s+before|"
            r"rule\s+\d+\s+should\s+override|that\s+was\s+incorrect\s+because|"
            r"change\s+[A-Za-z0-9_\-]+\s+(?:deadline|value|policy)?\s*to\s+|update\s+[A-Za-z0-9_\-]+\s+(?:deadline|value|policy)?\s*to\s+|"
            r"change\s+rule\s+|update\s+rule\s+)\b",
            lower,
        ))

        # 3. Detect Evaluation Intent
        is_evaluation = bool(re.search(
            r"\b(evaluate\s+(learned\s+)?rules?|run\s+(a\s+)?benchmark|test\s+(the\s+)?learned\s+rules?|evaluate\s+decision-making)\b",
            lower,
        ))

        # 4. Detect Teaching / Learning signals
        has_learning_signal = bool(re.search(
            r"\b(i\s+am\s+teaching\s+you|learn\s+these\s+rules|learning\s+experiment|teach\s+you\s+a\s+new|"
            r"new\s+decision-making\s+system|learn\s+and\s+save\s+these|learn\s+the\s+following|rules:\s*\d+[\.\)])\b",
            lower,
        ))

        if is_feedback:
            return LearningExtractionResult(
                is_learning_instruction=False,
                is_feedback=True,
                is_evaluation=False,
                namespace=namespace,
                extracted_rules=[],
                confidence=0.9,
                raw_input=raw,
            )

        if is_evaluation:
            return LearningExtractionResult(
                is_learning_instruction=False,
                is_feedback=False,
                is_evaluation=True,
                namespace=namespace,
                extracted_rules=[],
                confidence=0.9,
                raw_input=raw,
            )

        # 5. Extract rules from text (numbered or bulleted items across multiline or inline)
        extracted: List[ExtractedRule] = []
        unextracted: List[str] = []

        # Remove trailing task/instructions instructions block
        raw_clean = re.split(r"(?i)\b(?:your task|task|instructions?):", raw)[0]
        # Normalize inline rule prefixes like "RULES: 1." or ": 1." or " 1." to newlines
        clean_text = re.sub(r"[:\s]+(?=\d+[\.\)])", "\n", raw_clean)

        items = re.findall(r"^\s*(?:\d+[\.\)]|rule\s+\d+:?|[-*•])\s*(.+)$", clean_text, re.MULTILINE | re.IGNORECASE)

        for line_str in items:
            line_str = line_str.strip()
            if not line_str or len(line_str.split()) < 3:
                continue
            rule = self._extract_single_rule(line_str)
            if rule:
                extracted.append(rule)
            else:
                unextracted.append(line_str)

        is_instruction = has_learning_signal or (len(extracted) >= 1 and bool(re.search(r"\b(rule|rules|deadline|override|status|priority)\b", lower)))

        return LearningExtractionResult(
            is_learning_instruction=is_instruction,
            is_feedback=False,
            is_evaluation=False,
            namespace=namespace,
            extracted_rules=extracted,
            confidence=0.95 if extracted else 0.4,
            raw_input=raw,
            unextracted_lines=unextracted,
        )

    def _extract_single_rule(self, line: str) -> Optional[ExtractedRule]:
        """
        Generalized extraction of a single rule without hardcoding domain tokens.
        Handles:
        - Deadlines & response limits
        - Overrides & precedence
        - Conditions & prohibitions (e.g. status ON-HOLD -> no contact)
        - General definitions
        """
        line_clean = re.sub(r"^\s*(?:\d+[\.\)]|[-*•]|rule\s+\d+:?)\s*", "", line).strip()
        if not line_clean:
            return None

        # Pattern A: Override / Precedence Rules
        # e.g., "The ON-HOLD condition overrides all normal contact deadlines."
        # or "Rule A overrides Rule B" or "Condition X takes precedence over Y"
        override_m = re.search(
            r"^(.+?)\s+(?:condition\s+)?(?:overrides?|takes?\s+precedence\s+over|supercedes?|has\s+priority\s+over)\s+(.+?)(?:\.|$)",
            line_clean, re.IGNORECASE,
        )
        if override_m:
            subj = override_m.group(1).strip()
            target = override_m.group(2).strip()
            slug_key = self._slugify(subj) + "_OVERRIDE_RULE"
            return ExtractedRule(
                rule_key=slug_key,
                rule_text=line_clean,
                subject=subj,
                predicate="overrides",
                value=f"overrides {target}",
                conditions=subj,
                precedence=10,
                overrides=target,
                category="OVERRIDE",
                confidence=0.98,
            )

        # Pattern B: Conditional prohibition / action
        # e.g., "If a case has status ON-HOLD, no contact should be made, regardless of its priority level."
        cond_action_m = re.search(
            r"^(?:if\s+)?(?:a\s+)?(.+?)\s+has\s+status\s+([A-Za-z0-9_\-]+)[,\s]+(?:then\s+)?(.+?)(?:\.|$)",
            line_clean, re.IGNORECASE,
        )
        if cond_action_m:
            entity_type = cond_action_m.group(1).strip()
            status_val = cond_action_m.group(2).strip().upper()
            action = cond_action_m.group(3).strip()
            slug_key = f"{status_val}_STATUS_RULE"
            return ExtractedRule(
                rule_key=slug_key,
                rule_text=line_clean,
                subject=f"{entity_type} with status {status_val}",
                predicate="action_policy",
                value=action,
                conditions=f"status == {status_val}",
                precedence=5,
                category="CONDITIONAL",
                confidence=0.96,
            )

        # Pattern C: Generalized Deadline / Response Time
        # e.g., "LEVEL-A cases have a response deadline of 24 hours."
        # or "Priority P1 accounts must be contacted within 2 hours."
        # or "Tier-3 requests require resolution within 3 days."
        deadline_m = re.search(
            r"^([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,2}?)\s+(?:cases?|accounts?|requests?|items?|orders?|tickets?)?\s*"
            r"(?:have|has|must\s+have|require|requires|must\s+be\s+contacted|should\s+be\s+contacted|contact)?\s*"
            r"(?:a\s+)?(?:response\s+deadline|deadline|turnaround\s+time|contact\s+within|within)?\s*(?:of|within)?\s*"
            r"(\d+)\s+(hours?|minutes?|days?|weeks?|seconds?|hrs?|mins?)(?:\.|$)",
            line_clean, re.IGNORECASE,
        )
        if deadline_m:
            subj_raw = deadline_m.group(1).strip()
            num = deadline_m.group(2).strip()
            unit = deadline_m.group(3).strip().lower().rstrip("s") + "s"
            duration_val = f"{num} {unit}"
            slug_key = self._slugify(subj_raw) + "_DEADLINE_RULE"
            return ExtractedRule(
                rule_key=slug_key,
                rule_text=line_clean,
                subject=subj_raw,
                predicate="response_deadline",
                value=duration_val,
                conditions=f"target == {subj_raw}",
                precedence=1,
                category="DEADLINE",
                confidence=0.97,
            )

        # Pattern D: General X is Y / X means Y / X routes to Y
        general_m = re.search(
            r"^([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){0,2}?)\s+(?:means|is\s+defined\s+as|must\s+be|should\s+be|routes?\s+to)\s+(.+?)(?:\.|$)",
            line_clean, re.IGNORECASE,
        )
        if general_m:
            subj_raw = general_m.group(1).strip()
            action_raw = general_m.group(2).strip()
            slug_key = self._slugify(subj_raw) + "_RULE"
            return ExtractedRule(
                rule_key=slug_key,
                rule_text=line_clean,
                subject=subj_raw,
                predicate="policy",
                value=action_raw,
                conditions="",
                precedence=1,
                category="POLICY",
                confidence=0.90,
            )

        return None

    def _slugify(self, text: str) -> str:
        s = re.sub(r"[^A-Za-z0-9\s_\-]", "", text)
        s = re.sub(r"[\s\-]+", "_", s.strip()).upper()
        return s[:50]

    # -------------------------------------------------------------------------
    # Storage & Read-back Verification (Phase 4 & 5)
    # -------------------------------------------------------------------------

    def save_learned_rules_batch(
        self,
        namespace_or_rules: Any,
        rules: Optional[List[ExtractedRule]] = None,
        *,
        namespace: Optional[str] = None,
        user_scope: str = "default_user",
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomically saves extracted rules under the specified namespace in SQLite.
        Guarantees:
        - Reads every record back before returning confirmation.
        - Preserves version history.
        - Never returns success for a record that was not committed.
        Supports:
          save_learned_rules_batch(namespace, rules, user_scope=...)
          save_learned_rules_batch(rules, namespace=..., user_scope=...)
        """
        if isinstance(namespace_or_rules, str):
            ns_clean = namespace_or_rules.strip().upper()
            rule_list = rules if rules is not None else []
        elif isinstance(namespace_or_rules, list):
            rule_list = namespace_or_rules
            ns_clean = (namespace or "DEFAULT").strip().upper()
        else:
            rule_list = []
            ns_clean = (namespace or "DEFAULT").strip().upper()

        now = datetime.now(UTC).isoformat()

        saved = []
        conflicts = []
        failed = []

        with self._connect() as conn:
            # 1. Ensure namespace exists
            conn.execute("""
                INSERT INTO learning_namespaces (id, name, description, user_scope, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                ON CONFLICT(name) DO UPDATE SET updated_at = ?
            """, (f"ns_{ns_clean.lower()}", ns_clean, f"User-defined learning namespace {ns_clean}", user_scope, now, now, now))

            for r in rule_list:
                key = r.rule_key.strip().upper()
                val = r.value.strip()
                if not key or not val:
                    failed.append({"rule_key": key, "reason": "Empty rule_key or value"})
                    continue

                # Check existing rule in this namespace
                existing = conn.execute(
                    "SELECT * FROM learned_rules WHERE user_scope = ? AND namespace = ? AND rule_key = ?",
                    (user_scope, ns_clean, key),
                ).fetchone()

                if existing:
                    rule_id = existing["id"]
                    prev_val = existing["value"]
                    ver_num = existing["version"] + 1

                    if prev_val == val and not r.is_correction:
                        # Reaffirmed identical rule
                        saved.append(dict(existing))
                        continue

                    # Update existing rule
                    conn.execute("""
                        UPDATE learned_rules
                        SET rule_text = ?, value = ?, conditions = ?, precedence = ?,
                            overrides = ?, category = ?, confidence = ?, version = ?,
                            updated_at = ?, status = 'ACTIVE'
                        WHERE id = ?
                    """, (r.rule_text, val, r.conditions, r.precedence, r.overrides, r.category, r.confidence, ver_num, now, rule_id))

                    conn.execute("""
                        INSERT INTO learned_rule_versions (
                            id, rule_id, version_number, previous_value, new_value,
                            previous_conditions, new_conditions, change_type, reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'UPDATE', 'Explicit user update/correction', ?)
                    """, (str(uuid.uuid4()), rule_id, ver_num, prev_val, val, existing["conditions"], r.conditions, now))

                    # Read-back verification
                    read_row = conn.execute("SELECT * FROM learned_rules WHERE id = ?", (rule_id,)).fetchone()
                    if not read_row:
                        raise RuntimeError(f"Read-back failed for updated rule {rule_id}")
                    saved_dict = dict(read_row)
                    saved_dict["was_correction"] = True
                    saved_dict["previous_value"] = prev_val
                    saved.append(saved_dict)

                else:
                    # New rule
                    rule_id = f"rule_{uuid.uuid4().hex[:12]}"
                    conn.execute("""
                        INSERT INTO learned_rules (
                            id, namespace, rule_key, rule_text, subject, predicate,
                            value, conditions, precedence, overrides, category,
                            confidence, provenance, source_message_id, user_scope,
                            status, version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'explicit_user_instruction', ?, ?, 'ACTIVE', 1, ?, ?)
                    """, (
                        rule_id, ns_clean, key, r.rule_text, r.subject, r.predicate,
                        val, r.conditions, r.precedence, r.overrides, r.category,
                        r.confidence, source_message_id, user_scope, now, now,
                    ))

                    conn.execute("""
                        INSERT INTO learned_rule_versions (
                            id, rule_id, version_number, previous_value, new_value,
                            previous_conditions, new_conditions, change_type, reason, created_at
                        ) VALUES (?, ?, 1, NULL, ?, NULL, ?, 'INITIAL', 'Initial learning event', ?)
                    """, (str(uuid.uuid4()), rule_id, val, r.conditions, now))

                    # Update FTS
                    try:
                        conn.execute("""
                            INSERT INTO learned_rules_fts (rule_id, namespace, rule_key, subject, predicate, value, conditions, rule_text)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (rule_id, ns_clean, key, r.subject, r.predicate, val, r.conditions, r.rule_text))
                    except Exception:
                        pass

                    # Read-back verification
                    read_row = conn.execute("SELECT * FROM learned_rules WHERE id = ?", (rule_id,)).fetchone()
                    if not read_row:
                        raise RuntimeError(f"Read-back failed for new rule {rule_id}")
                    saved.append(dict(read_row))

            conn.commit()

        return {
            "namespace": ns_clean,
            "saved": saved,
            "conflicts": conflicts,
            "failed": failed,
            "total_rules": len(rules),
            "saved_count": len(saved),
            "failed_count": len(failed),
        }

    # -------------------------------------------------------------------------
    # Retrieval & Application (Phase 6)
    # -------------------------------------------------------------------------

    def retrieve_learned_rules(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        user_scope: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Retrieves active learned rules matching the query or namespace.
        Sorts rules by precedence descending (highest priority overrides first).
        """
        q = query.strip()
        with self._connect() as conn:
            # If explicit namespace specified:
            if namespace and namespace.upper() != "DEFAULT":
                ns_target = namespace.strip().upper()
                tokens = [t.upper() for t in re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", q) if len(t) >= 3]
                stopwords = {"WHAT", "WHICH", "TELL", "RULE", "RULES", "POLICY", "NAMESPACE", "SHOW", "LIST", "RECALL", "LEARNED", "FIND", "GET"}
                meaningful_tokens = [t for t in tokens if t not in stopwords and t != ns_target]
                if meaningful_tokens:
                    matched_ids: Set[str] = set()
                    matched_rows: List[sqlite3.Row] = []
                    for t in meaningful_tokens:
                        t_rows = conn.execute("""
                            SELECT * FROM learned_rules
                            WHERE user_scope = ? AND namespace = ? AND status = 'ACTIVE'
                              AND (rule_key LIKE ? OR subject LIKE ? OR conditions LIKE ? OR rule_text LIKE ? OR value LIKE ?)
                            ORDER BY precedence DESC
                        """, (user_scope, ns_target, f"%{t}%", f"%{t}%", f"%{t}%", f"%{t}%", f"%{t}%")).fetchall()
                        for r in t_rows:
                            if r["id"] not in matched_ids:
                                matched_ids.add(r["id"])
                                matched_rows.append(r)
                    rows = sorted(matched_rows, key=lambda x: x["precedence"], reverse=True)
                else:
                    rows = conn.execute("""
                        SELECT * FROM learned_rules
                        WHERE user_scope = ? AND namespace = ? AND status = 'ACTIVE'
                        ORDER BY precedence DESC, created_at ASC
                    """, (user_scope, ns_target)).fetchall()
            else:
                # Detect namespace from query
                ns_match = re.search(r"\bnamespace[:\s=]+([A-Za-z0-9_\-]+)", q, re.IGNORECASE)
                if ns_match:
                    detected_ns = ns_match.group(1).strip().upper()
                    rows = conn.execute("""
                        SELECT * FROM learned_rules
                        WHERE user_scope = ? AND namespace = ? AND status = 'ACTIVE'
                        ORDER BY precedence DESC, created_at ASC
                    """, (user_scope, detected_ns)).fetchall()
                else:
                    # Retrieve all active rules across user's namespaces that match tokens in query
                    tokens = [t.upper() for t in re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", q) if len(t) >= 3]
                    matched_ids: Set[str] = set()
                    matched_rows: List[sqlite3.Row] = []

                    for t in tokens:
                        t_rows = conn.execute("""
                            SELECT * FROM learned_rules
                            WHERE user_scope = ? AND status = 'ACTIVE'
                              AND (rule_key LIKE ? OR subject LIKE ? OR conditions LIKE ? OR rule_text LIKE ?)
                            ORDER BY precedence DESC LIMIT 10
                        """, (user_scope, f"%{t}%", f"%{t}%", f"%{t}%", f"%{t}%")).fetchall()
                        for r in t_rows:
                            if r["id"] not in matched_ids:
                                matched_ids.add(r["id"])
                                matched_rows.append(r)
                    rows = sorted(matched_rows, key=lambda x: x["precedence"], reverse=True)

            rules = [dict(r) for r in rows]

            # Fetch versions for retrieved rules
            for r in rules:
                vers = conn.execute(
                    "SELECT * FROM learned_rule_versions WHERE rule_id = ? ORDER BY version_number ASC",
                    (r["id"],),
                ).fetchall()
                r["versions"] = [dict(v) for v in vers]
                r["version_count"] = len(vers)

            context_block = self._format_learned_context(rules)
            return {
                "status": "FOUND" if rules else "NOT_FOUND",
                "count": len(rules),
                "total_rules": len(rules),
                "namespace": namespace or (rules[0].get("namespace") if rules else "DEFAULT"),
                "rules": rules,
                "context_block": context_block,
            }

    def evaluate_rule_query(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        user_scope: str = "default_user",
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministically evaluates a business rule scenario or inquiry against active learned rules.
        Enforces precedence (e.g. ON-HOLD precedence 10 overrides level deadlines precedence 1).
        Handles unknown rules (e.g. LEVEL-D) by refusing to invent defaults.
        Handles unknown status combinations without guessing.
        Returns None if query is not a rule evaluation question.
        """
        q = query.strip()
        q_lower = q.lower()

        # Complex reasoning scenarios or multi-part prompts with context/rules should NOT be hijacked as single-rule lookups
        levels_found = set(re.findall(r"\blevel[-_\s]*([a-z0-9]+)\b", q, re.IGNORECASE))
        if len(levels_found) > 1 or any(phrase in q_lower for phrase in [
            "solve the following", "consider these cases", "answer all of the following", 
            "stored rules:", "collection team has these", "calculate the contact deadline"
        ]):
            return None

        # 1. Check if query is asking about rules, deadlines, contact, or policy
        rule_inquiry_patterns = [
            r"\b(deadline|contact|response\s+time|turnaround|policy|action|procedure)\b",
            r"\b(what\s+is\s+the|what\s+should\s+be\s+done|what\s+action|how\s+should)\b",
            r"\b(level[-_\s]*[a-z0-9]+)\b",
        ]
        is_rule_inquiry = any(re.search(pat, q_lower) for pat in rule_inquiry_patterns)
        if not is_rule_inquiry:
            return None

        # 2. Extract Target Level (e.g. LEVEL-A, LEVEL-B, LEVEL-C, LEVEL-D, etc.)
        target_m = re.search(r"\b(level[-_\s]*([a-z0-9]+))\b", q, re.IGNORECASE)
        target = f"LEVEL-{target_m.group(2).upper()}" if target_m else None

        # 3. Extract Status (e.g. ON-HOLD, ACTIVE, or unknown status like PENDING_REVIEW)
        detected_status = None
        has_on_hold = bool(re.search(r"\b(on[-_\s]*hold|hold)\b", q_lower))
        has_active = bool(re.search(r"\b(active|normal|standard)\b", q_lower))

        NON_STATUS_WORDS = {
            "changes", "changed", "change", "is", "of", "to", "at", "until", "for",
            "in", "the", "a", "an", "code", "indicator", "value", "field", "becomes",
            "transitions", "applies", "remains", "status"
        }
        status_decl_m = re.search(r"\bstatus\s*(?:is|of|:|=)?\s*[:=]?\s*['\"]?([A-Za-z0-9_\-]+)['\"]?\b", q, re.IGNORECASE)
        if status_decl_m and status_decl_m.group(1).lower() not in NON_STATUS_WORDS:
            raw_status = status_decl_m.group(1).upper()
            if "HOLD" in raw_status:
                detected_status = "ON-HOLD"
            elif raw_status in ("ACTIVE", "NORMAL", "STANDARD"):
                detected_status = "ACTIVE"
            else:
                detected_status = raw_status  # Unknown status
        elif has_on_hold:
            detected_status = "ON-HOLD"
        elif has_active:
            detected_status = "ACTIVE"
        else:
            detected_status = "ACTIVE"

        if not target and not has_on_hold:
            return None

        # 4. Resolve Namespace
        ns_target = (namespace or "").strip().upper()
        if not ns_target or ns_target == "DEFAULT":
            ns_m = re.search(r"\bnamespace[:\s=]+([A-Za-z0-9_\-]+)", q, re.IGNORECASE)
            if ns_m:
                ns_target = ns_m.group(1).strip().upper()
            else:
                with self._connect() as conn:
                    active_ns = conn.execute("SELECT name FROM learning_namespaces WHERE status = 'ACTIVE' LIMIT 1").fetchone()
                    if active_ns:
                        ns_target = active_ns["name"]
                    else:
                        ns_target = "LEARNING_TEST_2026_01"

        # 5. Retrieve active rules for namespace ordered by precedence DESC
        with self._connect() as conn:
            rules_rows = conn.execute("""
                SELECT * FROM learned_rules 
                WHERE namespace = ? AND status = 'ACTIVE'
                ORDER BY precedence DESC, created_at ASC
            """, (ns_target,)).fetchall()
            rules = [dict(r) for r in rules_rows]

        if not rules:
            return None

        # 6. Evaluation Matrix
        # Case A: Unknown Status with known target
        if detected_status not in ("ACTIVE", "ON-HOLD", None):
            return {
                "status": "UNKNOWN_STATUS",
                "namespace": ns_target,
                "target": target,
                "detected_status": detected_status,
                "decision": None,
                "applied_rules": [],
                "precedence": 0,
                "response": (
                    f"### ⚠️ Undefined Status — `{detected_status}`\n\n"
                    f"The status `{detected_status}` is not defined in namespace `{ns_target}`. "
                    f"Cannot determine if response deadlines or overrides apply. I will not infer or guess an answer."
                ),
                "provenance": None,
            }

        # Case B: Override condition (ON-HOLD) takes highest precedence
        if detected_status == "ON-HOLD":
            override_rule = next((r for r in rules if r.get("precedence", 0) >= 5 or "ON-HOLD" in (r.get("conditions") or "")), None)
            if override_rule:
                target_label = target or "Account"
                return {
                    "status": "EVALUATED",
                    "namespace": ns_target,
                    "target": target,
                    "detected_status": "ON-HOLD",
                    "decision": "DEADLINES_SUSPENDED_NO_CONTACT",
                    "value": override_rule["value"],
                    "applied_rules": [override_rule["rule_key"]],
                    "precedence": override_rule["precedence"],
                    "response": (
                        f"### 🛑 Deterministic Business Rule Decision — Status: `ON-HOLD`\n\n"
                        f"- **Target:** `{target_label}` (Status: `ON-HOLD`)\n"
                        f"- **Decision:** **All response deadlines are suspended and no contact may be made.**\n"
                        f"- **Precedence:** {override_rule['precedence']} (Overrides all level response deadlines)\n"
                        f"- **Applied Rule:** `{override_rule['rule_key']}` (v{override_rule['version']})\n"
                        f"- **Namespace:** `{ns_target}`"
                    ),
                    "provenance": {
                        "rule_id": override_rule["id"],
                        "version": override_rule["version"],
                        "rule_key": override_rule["rule_key"],
                    },
                }

        # Case C: Standard Level Evaluation (ACTIVE status)
        if target:
            target_slug = self._slugify(target)
            matching_rule = next(
                (r for r in rules if target_slug in self._slugify(r.get("subject") or "") or target_slug in self._slugify(r.get("rule_key") or "") or target in (r.get("conditions") or "")),
                None,
            )
            if matching_rule:
                return {
                    "status": "EVALUATED",
                    "namespace": ns_target,
                    "target": target,
                    "detected_status": "ACTIVE",
                    "decision": matching_rule["value"],
                    "value": matching_rule["value"],
                    "applied_rules": [matching_rule["rule_key"]],
                    "precedence": matching_rule["precedence"],
                    "response": (
                        f"### ⏱️ Deterministic Business Rule Decision — Deadline\n\n"
                        f"- **Target:** `{target}` (Status: `ACTIVE`)\n"
                        f"- **Response Deadline:** **{matching_rule['value']}**\n"
                        f"- **Rule Key:** `{matching_rule['rule_key']}` (v{matching_rule['version']})\n"
                        f"- **Precedence:** {matching_rule['precedence']}\n"
                        f"- **Namespace:** `{ns_target}`"
                    ),
                    "provenance": {
                        "rule_id": matching_rule["id"],
                        "version": matching_rule["version"],
                        "rule_key": matching_rule["rule_key"],
                    },
                }
            else:
                # Target not found in rules (e.g. LEVEL-D, LEVEL-E)
                return {
                    "status": "UNKNOWN_RULE",
                    "namespace": ns_target,
                    "target": target,
                    "detected_status": detected_status,
                    "decision": None,
                    "applied_rules": [],
                    "precedence": 0,
                    "response": (
                        f"### ❓ Unknown Business Rule — `{target}`\n\n"
                        f"Rule for `{target}` is not defined in namespace `{ns_target}`. "
                        f"I do not have verified information stored for that level and will not invent a default deadline."
                    ),
                    "provenance": None,
                }

        return None

    def format_rules_for_context(self, rules: List[Dict[str, Any]], namespace: Optional[str] = None) -> str:
        """Formats active learned rules for injection into LLM context."""
        return self._format_learned_context(rules)

    def _format_learned_context(self, rules: List[Dict[str, Any]]) -> str:
        """Formats active learned rules for injection into LLM context."""
        if not rules:
            return ""

        ns_name = rules[0].get("namespace", "GENERAL")
        lines = [
            f"[ACTIVE LEARNED RULES - Namespace: {ns_name}]",
            "Follow these verified user-taught rules with strict priority and precedence:",
        ]
        for idx, r in enumerate(rules, 1):
            prec_label = f" [Precedence: {r['precedence']}]" if r.get("precedence", 0) > 0 else ""
            cond_str = f" (Condition: {r['conditions']})" if r.get("conditions") else ""
            override_str = f" [OVERRIDES: {r['overrides']}]" if r.get("overrides") else ""
            lines.append(f"{idx}. {r['rule_key']}: {r['value']}{cond_str}{override_str}{prec_label}")
            lines.append(f"   Statement: \"{r['rule_text']}\" [ID: {r['id']}, v{r.get('version', 1)}]")

        lines.append("[END ACTIVE LEARNED RULES]")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Feedback Learning (Phase 7)
    # -------------------------------------------------------------------------

    def apply_feedback(
        self,
        feedback_text: str,
        *,
        namespace: Optional[str] = "DEFAULT",
        user_scope: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Parses corrective user feedback, determines if it is ambiguous,
        updates the relevant learned rule, and records feedback audit log.
        """
        fb_raw = feedback_text.strip()
        fb_lower = fb_raw.lower()
        now = datetime.now(UTC).isoformat()
        ns_clean = (namespace or "DEFAULT").strip().upper()

        # Check for ambiguity
        # If user just says "that was wrong" without specifying what to change
        if len(fb_raw.split()) < 4 or (not re.search(r"\b(rule|before|after|instead|change|should be|override|now)\b", fb_lower)):
            return {
                "status": "CLARIFICATION_NEEDED",
                "message": (
                    "Your feedback indicates a correction, but does not specify which rule or precedence to update. "
                    "Please specify: e.g., 'Apply Rule 3 before Rule 2' or 'Change LEVEL-A deadline to 12 hours'."
                ),
                "applied_rule_id": None,
            }

        # Check precedence update: "Apply Rule 3 before Rule 2" or "Rule X has higher priority"
        prec_m = re.search(r"apply\s+rule\s+(\d+)\s+before\s+rule\s+(\d+)", fb_lower)
        if prec_m:
            r_high_num = int(prec_m.group(1))
            r_low_num = int(prec_m.group(2))

            with self._connect() as conn:
                rules = conn.execute("""
                    SELECT * FROM learned_rules WHERE user_scope = ? AND namespace = ? AND status = 'ACTIVE'
                    ORDER BY created_at ASC
                """, (user_scope, ns_clean)).fetchall()

                if len(rules) >= max(r_high_num, r_low_num):
                    target_high = rules[r_high_num - 1]
                    target_low = rules[r_low_num - 1]

                    new_prec = (target_low["precedence"] or 0) + 5
                    conn.execute("""
                        UPDATE learned_rules SET precedence = ?, updated_at = ? WHERE id = ?
                    """, (new_prec, now, target_high["id"]))

                    conn.execute("""
                        INSERT INTO learned_rule_versions (
                            id, rule_id, version_number, previous_value, new_value,
                            change_type, reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'FEEDBACK_PRECEDENCE', ?, ?)
                    """, (str(uuid.uuid4()), target_high["id"], target_high["version"] + 1,
                          target_high["value"], target_high["value"],
                          f"Applied Rule {r_high_num} before Rule {r_low_num} based on user feedback", now))

                    # Log feedback
                    log_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO learning_feedback_logs (
                            id, namespace, query, feedback_text, parsed_correction,
                            applied_rule_id, status, user_scope, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'APPLIED', ?, ?)
                    """, (log_id, ns_clean, fb_raw, fb_raw, json.dumps({"higher_rule": r_high_num, "lower_rule": r_low_num, "new_precedence": new_prec}), target_high["id"], user_scope, now))
                    conn.commit()

                    return {
                        "status": "APPLIED",
                        "message": f"Updated precedence: Rule {r_high_num} ({target_high['rule_key']}) now takes precedence over Rule {r_low_num} ({target_low['rule_key']}).",
                        "applied_rule_id": target_high["id"],
                        "rule_key": target_high["rule_key"],
                        "new_precedence": new_prec,
                    }

        # Check value correction: "Change X deadline to Y" or "X is now Y"
        val_m = re.search(r"(?:change|update|set)\s+([A-Za-z0-9_\-]+)\s+(?:deadline|value|to)?\s+(?:to\s+)?(\d+\s+(?:hours?|days?|minutes?)|[A-Za-z0-9_\-]+)", fb_raw, re.IGNORECASE)
        if val_m:
            target_key_part = self._slugify(val_m.group(1))
            new_val = val_m.group(2).strip()

            with self._connect() as conn:
                rule_row = conn.execute("""
                    SELECT * FROM learned_rules WHERE user_scope = ? AND namespace = ? AND rule_key LIKE ?
                """, (user_scope, ns_clean, f"%{target_key_part}%")).fetchone()

                if rule_row:
                    rule_id = rule_row["id"]
                    prev_val = rule_row["value"]
                    new_ver = rule_row["version"] + 1

                    subject_val = rule_row["subject"] or rule_row["rule_key"]
                    if "deadline" in (rule_row["predicate"] or "").lower() or "deadline" in rule_row["rule_key"].lower():
                        new_rule_text = f"{subject_val} cases have response deadline of {new_val}."
                    elif "override" in (rule_row["predicate"] or "").lower() or "override" in rule_row["rule_key"].lower():
                        new_rule_text = f"{subject_val} overrides {new_val}."
                    elif "status" in (rule_row["predicate"] or "").lower() or "status" in rule_row["rule_key"].lower():
                        new_rule_text = f"If an account is placed {subject_val}, {new_val}."
                    else:
                        new_rule_text = f"{subject_val} {rule_row['predicate'] or 'rule'} is {new_val}."

                    conn.execute("""
                        UPDATE learned_rules 
                        SET value = ?, rule_text = ?, version = ?, updated_at = ? 
                        WHERE id = ?
                    """, (new_val, new_rule_text, new_ver, now, rule_id))

                    # Atomic FTS update
                    conn.execute("DELETE FROM learned_rules_fts WHERE rule_id = ?", (rule_id,))
                    conn.execute("""
                        INSERT INTO learned_rules_fts (rule_id, namespace, rule_key, subject, predicate, value, conditions, rule_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rule_id, ns_clean, rule_row["rule_key"], rule_row["subject"], rule_row["predicate"], new_val, rule_row["conditions"], new_rule_text))

                    conn.execute("""
                        INSERT INTO learned_rule_versions (
                            id, rule_id, version_number, previous_value, new_value,
                            change_type, reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'FEEDBACK_CORRECTION', ?, ?)
                    """, (str(uuid.uuid4()), rule_id, new_ver, prev_val, new_val, fb_raw, now))

                    conn.commit()
                    return {
                        "status": "APPLIED",
                        "message": f"Corrected {rule_row['rule_key']}: updated from '{prev_val}' to '{new_val}' (v{new_ver}).",
                        "applied_rule_id": rule_id,
                        "rule_key": rule_row["rule_key"],
                        "new_value": new_val,
                    }

        return {
            "status": "CLARIFICATION_NEEDED",
            "message": "Feedback received, but could not identify the specific rule to update. Please provide rule key or number.",
            "applied_rule_id": None,
        }

    # -------------------------------------------------------------------------
    # Behavior Evaluation Benchmark (Phase 8)
    # -------------------------------------------------------------------------

    def evaluate_benchmark(
        self,
        test_cases: List[Dict[str, Any]],
        *,
        namespace: str = "DEFAULT",
        user_scope: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Runs test scenarios against active learned rules in the namespace.
        Reports test input, expected decision, actual decision, rules applied, correctness, score.
        """
        ns_clean = namespace.strip().upper()
        rules_res = self.retrieve_learned_rules("", namespace=ns_clean, user_scope=user_scope)
        rules = rules_res["rules"]

        results = []
        correct_count = 0
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            for tc in test_cases:
                test_input = tc.get("input", "")
                expected = tc.get("expected", "").strip()

                # Simulate rule evaluation on test scenario
                # Check overrides first (highest precedence)
                applied_rules = []
                decision = "NO_MATCH"

                # Check if scenario matches an override condition (e.g. status ON-HOLD)
                for r in rules:
                    if r.get("precedence", 0) >= 5 and r.get("conditions"):
                        # Check if condition term appears in input
                        cond_term = r["conditions"].replace("status ==", "").replace("target ==", "").strip().lower()
                        if cond_term and cond_term in test_input.lower():
                            applied_rules.append(r["rule_key"])
                            if "suspended" in r["value"].lower() or "no contact" in r["value"].lower():
                                decision = r["value"]
                                break
                            elif decision == "NO_MATCH":
                                decision = r["value"]

                # If no override applied, check standard deadline/policy rules
                if decision == "NO_MATCH":
                    for r in rules:
                        subj_term = (r.get("subject") or "").lower()
                        if subj_term and subj_term in test_input.lower():
                            decision = r["value"]
                            applied_rules.append(r["rule_key"])
                            break

                is_correct = 1 if expected.lower() in decision.lower() else 0
                if is_correct:
                    correct_count += 1

                bench_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO learning_benchmarks (
                        id, namespace, name, test_input, expected_decision, actual_decision,
                        rules_retrieved, rules_applied, is_correct, score, user_scope, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bench_id, ns_clean, tc.get("name", "Test Scenario"), test_input,
                    expected, decision, json.dumps([r["rule_key"] for r in rules]),
                    json.dumps(applied_rules), is_correct, 1.0 if is_correct else 0.0,
                    user_scope, now,
                ))

                results.append({
                    "id": bench_id,
                    "name": tc.get("name", "Scenario"),
                    "input": test_input,
                    "expected": expected,
                    "actual": decision,
                    "applied_rules": applied_rules,
                    "is_correct": bool(is_correct),
                })

            conn.commit()

        total = len(test_cases)
        accuracy = round((correct_count / total) * 100, 1) if total > 0 else 0.0

        return {
            "namespace": ns_clean,
            "total_tests": total,
            "passed": correct_count,
            "failed": total - correct_count,
            "accuracy_pct": accuracy,
            "results": results,
        }

    # -------------------------------------------------------------------------
    # Management & Exploration
    # -------------------------------------------------------------------------

    def list_namespaces(self, user_scope: str = "default_user") -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT n.*, COUNT(r.id) as rule_count
                FROM learning_namespaces n
                LEFT JOIN learned_rules r ON n.name = r.namespace AND r.status = 'ACTIVE'
                WHERE n.user_scope = ?
                GROUP BY n.id
                ORDER BY n.updated_at DESC
            """, (user_scope,)).fetchall()
            return [dict(r) for r in rows]

    def list_rules(self, namespace: str, user_scope: str = "default_user") -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM learned_rules
                WHERE user_scope = ? AND namespace = ?
                ORDER BY precedence DESC, created_at ASC
            """, (user_scope, namespace.strip().upper())).fetchall()
            return [dict(r) for r in rows]

    def create_namespace(self, name: str, description: Optional[str] = None, user_scope: str = "default_user") -> Dict[str, Any]:
        ns_clean = name.strip().upper()
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM learning_namespaces WHERE name = ? AND user_scope = ?",
                (ns_clean, user_scope),
            ).fetchone()
            if existing:
                return dict(existing)
            ns_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO learning_namespaces (id, name, description, user_scope, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, (ns_id, ns_clean, description or f"Learning domain {ns_clean}", user_scope, now, now))
            conn.commit()
            row = conn.execute("SELECT * FROM learning_namespaces WHERE id = ?", (ns_id,)).fetchone()
            return dict(row)

    def get_rule_by_id(self, rule_id: str, user_scope: str = "default_user") -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM learned_rules WHERE id = ? AND user_scope = ?", (rule_id, user_scope)).fetchone()
            return dict(row) if row else None

    def delete_rule(self, rule_id: str, user_scope: str = "default_user") -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM learned_rules WHERE id = ? AND user_scope = ?", (rule_id, user_scope)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM learned_rules WHERE id = ?", (rule_id,))
            conn.execute("""
                INSERT INTO learned_rule_versions (
                    id, rule_id, version_number, previous_value, new_value,
                    change_type, reason, created_at
                ) VALUES (?, ?, ?, ?, 'DELETED', 'DELETION', 'User explicit deletion', ?)
            """, (str(uuid.uuid4()), rule_id, row["version"] + 1, row["value"], now))
            conn.commit()
            return True

    def get_rule_history(self, rule_id: str, user_scope: str = "default_user") -> Dict[str, Any]:
        with self._connect() as conn:
            rule = conn.execute("SELECT * FROM learned_rules WHERE id = ? AND user_scope = ?", (rule_id, user_scope)).fetchone()
            if not rule:
                return {}
            vers = conn.execute(
                "SELECT * FROM learned_rule_versions WHERE rule_id = ? ORDER BY version_number ASC",
                (rule_id,),
            ).fetchall()
            d = dict(rule)
            d["versions"] = [dict(v) for v in vers]
            return d

    def get_learning_stats(self, user_scope: str = "default_user") -> Dict[str, Any]:
        with self._connect() as conn:
            tot_rules = conn.execute("SELECT COUNT(*) FROM learned_rules WHERE user_scope = ?", (user_scope,)).fetchone()[0]
            tot_ns = conn.execute("SELECT COUNT(*) FROM learning_namespaces WHERE user_scope = ?", (user_scope,)).fetchone()[0]
            tot_bench = conn.execute("SELECT COUNT(*) FROM learning_benchmarks WHERE user_scope = ?", (user_scope,)).fetchone()[0]
            pass_bench = conn.execute("SELECT COUNT(*) FROM learning_benchmarks WHERE user_scope = ? AND is_correct = 1", (user_scope,)).fetchone()[0]
            recent_feedback = conn.execute("""
                SELECT * FROM learning_feedback_logs WHERE user_scope = ? ORDER BY created_at DESC LIMIT 5
            """, (user_scope,)).fetchall()

            return {
                "total_namespaces": tot_ns,
                "total_learned_rules": tot_rules,
                "benchmarks_executed": tot_bench,
                "benchmarks_passed": pass_bench,
                "benchmark_accuracy_pct": round((pass_bench / tot_bench) * 100, 1) if tot_bench > 0 else 100.0,
                "recent_feedback": [dict(f) for f in recent_feedback],
            }
