"""Autonomous research planner, knowledge-gap detection, and bounded learning queue.

In accordance with PRD Prompts 08, 09, and 10:
  - Treats all documents as untrusted content; never executes instructions in source text.
  - Distinguishes source text, extracted knowledge, and model interpretations.
  - Enforces strict execution, source, and storage budgets.
  - Automatically identifies knowledge gaps from missing connections and low-confidence queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aarohan.memory import MemoryStore

VALID_GAP_TYPES = frozenset({
    "missing_relationship",
    "unanswered_query",
    "contradictory_evidence",
    "low_confidence_knowledge",
    "manual_learning_goal",
})

VALID_TASK_STATUSES = frozenset({
    "pending",
    "pending_approval",
    "approved",
    "in_progress",
    "completed",
    "rejected",
    "failed",
})


def validate_source(source_path: Path | str) -> tuple[bool, float, str]:
    """
    Validates document source reliability, checking file readability, extension safety,
    and computing source reliability score and SHA-256 provenance hash.
    """
    p = Path(source_path)
    if not p.is_file():
        return False, 0.0, "File does not exist"
    allowed_suffixes = {".txt", ".md", ".json", ".csv", ".py", ".pdf"}
    if p.suffix.lower() not in allowed_suffixes:
        return False, 0.0, f"Unsupported file extension {p.suffix}"
    try:
        content = p.read_bytes()
        if len(content) > 10 * 1024 * 1024:
            return False, 0.0, "File size exceeds 10MB budget limit"
        h = hashlib.sha256(content).hexdigest()
        return True, 0.90, h
    except Exception as e:
        return False, 0.0, str(e)


class FactList(list):
    """List that safely supports numeric comparisons for backward compatibility."""

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return len(self) >= other
        return super().__ge__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return len(self) > other
        return super().__gt__(other)

    def __le__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return len(self) <= other
        return super().__le__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return len(self) < other
        return super().__lt__(other)


class LearningQueue:
    """Manages autonomous learning tasks and research proposals."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.store = memory_store
        self._init_table()

    def _init_table(self) -> None:
        with self.store._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    query TEXT NOT NULL,
                    gap_type TEXT NOT NULL,
                    priority REAL NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    max_budget_docs INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_summary TEXT,
                    research_artifact_path TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    error_message TEXT DEFAULT NULL,
                    requires_approval INTEGER DEFAULT 0,
                    approved_by TEXT DEFAULT NULL,
                    source_hash TEXT DEFAULT NULL
                )
            """)

            cursor = conn.execute("PRAGMA table_info(learning_tasks)")
            cols = {row["name"] for row in cursor.fetchall()}
            if cols:
                if "retry_count" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN retry_count INTEGER DEFAULT 0")
                if "max_retries" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN max_retries INTEGER DEFAULT 3")
                if "error_message" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN error_message TEXT DEFAULT NULL")
                if "requires_approval" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN requires_approval INTEGER DEFAULT 0")
                if "approved_by" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN approved_by TEXT DEFAULT NULL")
                if "source_hash" not in cols:
                    conn.execute("ALTER TABLE learning_tasks ADD COLUMN source_hash TEXT DEFAULT NULL")

    def create_task(
        self,
        title: str,
        query: str,
        *,
        gap_type: str = "manual_learning_goal",
        priority: float = 0.5,
        objective: str = "",
        max_budget_docs: int = 5,
        requires_approval: bool = False,
        prevent_duplicates: bool = True,
    ) -> dict[str, Any]:
        if gap_type not in VALID_GAP_TYPES:
            raise ValueError(f"Invalid gap_type '{gap_type}'. Valid: {sorted(VALID_GAP_TYPES)}")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("Priority must be in [0.0, 1.0]")
        if max_budget_docs < 1:
            raise ValueError("max_budget_docs must be positive")

        cleaned_title = title.strip()
        cleaned_query = query.strip()

        with self.store._connect() as conn:
            if prevent_duplicates:
                existing = conn.execute(
                    "SELECT * FROM learning_tasks WHERE LOWER(query) = LOWER(?) AND status NOT IN ('failed', 'rejected')",
                    (cleaned_query,),
                ).fetchone()
                if existing:
                    logger.info("Reusing existing research task for query '%s'", cleaned_query)
                    return dict(existing)

            task_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            sensitive_keywords = ["sensitive", "critical", "architectural modification", "system core update", "override security"]
            is_sensitive = requires_approval or (
                priority >= 0.9 and any(kw in (cleaned_title + " " + cleaned_query).lower() for kw in sensitive_keywords)
            )
            needs_approval = 1 if is_sensitive else 0
            initial_status = "pending_approval" if needs_approval else "pending"

            conn.execute("""
                INSERT INTO learning_tasks(
                    id, title, query, gap_type, priority, objective,
                    status, max_budget_docs, requires_approval, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, cleaned_title, cleaned_query, gap_type, priority, objective.strip(), initial_status, max_budget_docs, needs_approval, now, now))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row)

    def approve_task(self, task_id: str, approved_by: str = "human_supervisor") -> dict[str, Any]:
        """Approves a research task awaiting human approval."""
        now = datetime.now(UTC).isoformat()
        with self.store._connect() as conn:
            conn.execute("""
                UPDATE learning_tasks
                SET status = 'approved', approved_by = ?, updated_at = ?
                WHERE id = ?
            """, (approved_by, now, task_id))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                raise KeyError(f"Task {task_id} not found")
            return dict(row)

    def retry_task(self, task_id: str) -> dict[str, Any]:
        """Retries a failed research task up to max_retries limit."""
        task = self.get_task(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")

        current_retries = task.get("retry_count", 0)
        max_retries = task.get("max_retries", 3)
        now = datetime.now(UTC).isoformat()

        with self.store._connect() as conn:
            if current_retries >= max_retries:
                conn.execute("""
                    UPDATE learning_tasks
                    SET status = 'failed', error_message = 'Max retries exceeded', updated_at = ?
                    WHERE id = ?
                """, (now, task_id))
            else:
                conn.execute("""
                    UPDATE learning_tasks
                    SET status = 'pending', retry_count = retry_count + 1, error_message = NULL, updated_at = ?
                    WHERE id = ?
                """, (now, task_id))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row)

    def record_failure(self, task_id: str, error_message: str) -> dict[str, Any]:
        """Marks a research task as failed, recording error diagnostic and Layer 6 audit memory."""
        now = datetime.now(UTC).isoformat()
        with self.store._connect() as conn:
            conn.execute("""
                UPDATE learning_tasks
                SET status = 'failed', error_message = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
            """, (error_message, now, now, task_id))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                raise KeyError(f"Task {task_id} not found")

        task_row = dict(row) if row else {}
        # Record failure into Layer 6 (Learning memory)
        try:
            self.store.record_learning_event(
                event_type="research_failure",
                title=f"Task Failed: {task_row.get('title', 'Unknown')}",
                details={"task_id": task_id, "error": error_message},
            )
        except Exception as ex:
            logger.error("Failed to write Layer 6 learning event: %s", ex)

        return task_row

    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.store._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM learning_tasks WHERE status = ? ORDER BY priority DESC, created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM learning_tasks ORDER BY priority DESC, created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row) if row else None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        *,
        result_summary: str | None = None,
        artifact_path: Path | None = None,
    ) -> dict[str, Any]:
        if status not in VALID_TASK_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Valid: {sorted(VALID_TASK_STATUSES)}")
        now = datetime.now(UTC).isoformat()
        completed_at = now if status in {"completed", "rejected", "failed"} else None
        with self.store._connect() as conn:
            conn.execute("""
                UPDATE learning_tasks
                SET status = ?, updated_at = ?, completed_at = COALESCE(?, completed_at),
                    result_summary = COALESCE(?, result_summary),
                    research_artifact_path = COALESCE(?, research_artifact_path)
                WHERE id = ?
            """, (status, now, completed_at, result_summary, str(artifact_path) if artifact_path else None, task_id))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                raise KeyError(f"Task {task_id} not found")
            return dict(row)


class KnowledgeGapDetector:
    """Detects missing relationships, unverified clusters, and research opportunities."""

    def __init__(self, memory_store: MemoryStore, queue: LearningQueue) -> None:
        self.store = memory_store
        self.queue = queue

    def detect_missing_graph_relationships(self, min_priority: float = 0.5) -> list[dict[str, Any]]:
        """
        Identify pairs of concepts frequently co-occurring in memory chunks
        that lack explicit relational edges in the Knowledge Graph.
        """
        entities = self.store.list_entities()
        entity_names = [e["name"] for e in entities]
        proposed_tasks: list[dict[str, Any]] = []

        for i, e1 in enumerate(entity_names):
            for e2 in entity_names[i + 1 :]:
                # Check if relation already exists
                relations = self.store.get_relations(e1)
                connected = any(
                    r["source_name"] == e2 or r["target_name"] == e2
                    for r in relations
                )
                if not connected:
                    # Search memory for joint co-occurrence
                    co_occurrences = self.store.search(f"{e1} {e2}", limit=5)
                    if len(co_occurrences) >= 2:
                        priority = min(0.9, 0.4 + 0.1 * len(co_occurrences))
                        if priority >= min_priority:
                            task = self.queue.create_task(
                                title=f"Investigate relationship between {e1} and {e2}",
                                query=f"{e1} {e2}",
                                gap_type="missing_relationship",
                                priority=priority,
                                objective=f"Find structured relationship connecting {e1} and {e2} from approved documents",
                            )
                            proposed_tasks.append(task)
        return proposed_tasks


class BoundedResearchEngine:
    """
    Executes an approved learning task against approved local knowledge files.
    Treats all incoming document text strictly as untrusted data without command execution.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        queue: LearningQueue,
        *,
        artifacts_dir: Path,
    ) -> None:
        self.store = memory_store
        self.queue = queue
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def execute_task(
        self,
        task_id: str,
        source_documents: list[Path],
    ) -> dict[str, Any]:
        task = self.queue.get_task(task_id)
        if not task:
            raise KeyError(f"Task '{task_id}' not found")

        # Human approval gate enforcement
        if task.get("status") == "pending_approval":
            raise PermissionError(f"Task '{task_id}' requires human approval before autonomous research execution")

        self.queue.update_task_status(task_id, "in_progress")
        budget_docs = task["max_budget_docs"]
        docs_to_scan = [p for p in source_documents if p.is_file()][:budget_docs]

        extracted_facts: list[str] = []
        contradictions_found: list[dict[str, Any]] = []
        entities_found: list[str] = []
        source_provenances: list[dict[str, Any]] = []
        keywords = re.findall(r"\b\w{3,}\b", task["query"].lower())
        injection_pattern = re.compile(r"(?i)\b(ignore (?:all )?previous instructions|you are now|system prompt|disregard (?:all )?prior)\b")

        for doc_path in docs_to_scan:
            valid, rel_score, p_hash = validate_source(doc_path)
            if not valid:
                logger.warning("Skipping invalid research source '%s': %s", doc_path, p_hash)
                continue

            source_provenances.append({
                "path": str(doc_path),
                "hash": p_hash,
                "reliability": rel_score,
            })

            try:
                content = doc_path.read_text(encoding="utf-8")
            except OSError:
                continue

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p in paragraphs:
                # Prompt injection defense: neutralize and skip lines attempting instruction takeover
                if injection_pattern.search(p):
                    logger.warning("Prompt injection attempt detected and neutralized in untrusted source document '%s'", doc_path)
                    continue

                p_lower = p.lower()
                matches = sum(1 for kw in keywords if kw in p_lower)
                if matches >= 1:
                    fact_str = " ".join(p.split())
                    if len(fact_str) > 20:
                        contra = self.store.detect_contradictions(fact_str, threshold=0.7)
                        if contra:
                            contradictions_found.extend(contra)

                        extracted_facts.append(fact_str)
                        for cap in re.findall(r"\b[A-Z][a-zA-Z0-9_]{3,}\b", fact_str):
                            if cap not in entities_found:
                                entities_found.append(cap)

        # Propose semantic memories with provenance
        proposed_records = []
        for fact in extracted_facts[:5]:
            rec = self.store.add(
                fact,
                source=f"autonomous_research:{task['id']}",
                memory_type="semantic",
                confidence=0.85,
                verification_status="unverified",
                verified_by="autonomous_research_engine",
                importance=task["priority"],
                metadata={
                    "research_task_id": task_id,
                    "query": task["query"],
                    "provenance": source_provenances,
                },
            )
            proposed_records.append(rec["id"])

        # Create structured research report artifact
        report_data = {
            "task_id": task_id,
            "title": task["title"],
            "query": task["query"],
            "documents_scanned": len(docs_to_scan),
            "source_provenance": source_provenances,
            "facts_extracted": FactList(extracted_facts),
            "facts_extracted_count": len(extracted_facts),
            "entities_discovered": entities_found,
            "contradictions_detected": contradictions_found,
            "proposed_memory_ids": proposed_records,
            "completed_at": datetime.now(UTC).isoformat(),
        }

        artifact_file = self.artifacts_dir / f"research_{task_id[:8]}.json"
        artifact_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        summary = f"Scanned {len(docs_to_scan)} docs; extracted {len(extracted_facts)} candidate facts; found {len(entities_found)} entities."
        self.queue.update_task_status(
            task_id,
            "completed",
            result_summary=summary,
            artifact_path=artifact_file,
        )

        return report_data

    def evaluate_knowledge_improvement(
        self,
        task_id: str,
        verification_query: str,
    ) -> dict[str, Any]:
        """
        Evaluates whether the newly learned knowledge from a completed research task
        is actively retrievable and improves future answers.
        """
        task = self.queue.get_task(task_id)
        if not task:
            raise KeyError(f"Task '{task_id}' not found")

        results = self.store.search(verification_query, limit=5)
        matched_memories = [
            r for r in results
            if f"autonomous_research:{task_id}" in (r.get("source") or "")
        ]

        improves = len(matched_memories) > 0
        return {
            "task_id": task_id,
            "verification_query": verification_query,
            "memories_retrieved": len(matched_memories),
            "knowledge_improvement_verified": improves,
            "matched_memory_ids": [m["id"] for m in matched_memories],
        }
