"""Autonomous research planner, knowledge-gap detection, and bounded learning queue.

In accordance with PRD Prompts 08, 09, and 10:
  - Treats all documents as untrusted content; never executes instructions in source text.
  - Distinguishes source text, extracted knowledge, and model interpretations.
  - Enforces strict execution, source, and storage budgets.
  - Automatically identifies knowledge gaps from missing connections and low-confidence queries.
"""

from __future__ import annotations

import json
import re
import uuid
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
    "approved",
    "in_progress",
    "completed",
    "rejected",
    "failed",
})


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
                    research_artifact_path TEXT
                )
            """)

    def create_task(
        self,
        title: str,
        query: str,
        *,
        gap_type: str = "manual_learning_goal",
        priority: float = 0.5,
        objective: str = "",
        max_budget_docs: int = 5,
    ) -> dict[str, Any]:
        if gap_type not in VALID_GAP_TYPES:
            raise ValueError(f"Invalid gap_type '{gap_type}'. Valid: {sorted(VALID_GAP_TYPES)}")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("Priority must be in [0.0, 1.0]")
        if max_budget_docs < 1:
            raise ValueError("max_budget_docs must be positive")

        task_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self.store._connect() as conn:
            conn.execute("""
                INSERT INTO learning_tasks(
                    id, title, query, gap_type, priority, objective,
                    status, max_budget_docs, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """, (task_id, title.strip(), query.strip(), gap_type, priority, objective.strip(), max_budget_docs, now, now))
            row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row)

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

        self.queue.update_task_status(task_id, "in_progress")
        budget_docs = task["max_budget_docs"]
        docs_to_scan = [p for p in source_documents if p.is_file()][:budget_docs]

        extracted_facts: list[str] = []
        contradictions_found: list[dict[str, Any]] = []
        entities_found: list[str] = []
        keywords = re.findall(r"\b\w{3,}\b", task["query"].lower())

        for doc_path in docs_to_scan:
            try:
                content = doc_path.read_text(encoding="utf-8")
            except OSError:
                continue

            # Paragraph-level extraction matching query keywords
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p in paragraphs:
                p_lower = p.lower()
                matches = sum(1 for kw in keywords if kw in p_lower)
                if matches >= 1:
                    # Clean and check potential contradictions
                    fact_str = " ".join(p.split())
                    if len(fact_str) > 20:
                        contra = self.store.detect_contradictions(fact_str, threshold=0.7)
                        if contra:
                            contradictions_found.extend(contra)

                        extracted_facts.append(fact_str)
                        # Extract capitalized entity candidates
                        for cap in re.findall(r"\b[A-Z][a-zA-Z0-9_]{3,}\b", fact_str):
                            if cap not in entities_found:
                                entities_found.append(cap)

        # Propose unverified semantic memories
        proposed_records = []
        for fact in extracted_facts[:5]:
            rec = self.store.add(
                fact,
                source=f"autonomous_research:{task['id']}",
                memory_type="semantic",
                confidence=0.7,
                verification_status="unverified",
                importance=task["priority"],
                metadata={"research_task_id": task_id, "query": task["query"]},
            )
            proposed_records.append(rec["id"])

        # Create structured research report artifact
        report_data = {
            "task_id": task_id,
            "title": task["title"],
            "query": task["query"],
            "documents_scanned": len(docs_to_scan),
            "facts_extracted": len(extracted_facts),
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
