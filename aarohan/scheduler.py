"""Autonomous learning scheduler executing bounded research and controlled improvement cycles.

In accordance with PRD Section 11 and Prompt 13:
  - Inspects learning queue for prioritized tasks.
  - Checks resource availability before training.
  - Executes bounded research on approved topics.
  - Trains candidate models with experience replay.
  - Enforces regression gate testing.
  - Leaves final production deployment requiring explicit approval.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch

from aarohan.continual import RegressionGate, evaluate_benchmark_loss, train_continual_candidate
from aarohan.memory import MemoryStore
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.research import BoundedResearchEngine, LearningQueue
from aarohan.safety import AuditLogger
from aarohan.tokenizer import load_tokenizer


class LearningScheduler:
    """Orchestrates autonomous learning cycles with resource checks and safety gates."""

    def __init__(
        self,
        memory_store: MemoryStore,
        queue: LearningQueue,
        research_engine: BoundedResearchEngine,
        audit_logger: AuditLogger,
        *,
        artifacts_dir: Path,
    ) -> None:
        self.store = memory_store
        self.queue = queue
        self.engine = research_engine
        self.audit = audit_logger
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_system_resources(max_ram_percent: float = 98.0) -> dict[str, Any]:
        """Verify memory and compute headroom prior to initiating a training cycle."""
        vm = psutil.virtual_memory()
        cuda_available = torch.cuda.is_available()
        cuda_mem_free = None
        if cuda_available:
            cuda_mem_free = torch.cuda.mem_get_info()[0] / (1024 * 1024)

        has_headroom = vm.percent < max_ram_percent
        return {
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "ram_percent_used": vm.percent,
            "cuda_available": cuda_available,
            "cuda_free_mb": cuda_mem_free,
            "can_train": has_headroom,
        }

    def execute_cycle(
        self,
        *,
        base_checkpoint: Path,
        tokenizer_path: Path,
        historical_replay_file: Path,
        source_documents: list[Path],
        output_candidate_path: Path,
        device_name: str = "cpu",
        max_ram_percent: float = 98.0,
    ) -> dict[str, Any]:
        """Execute one complete bounded learning cycle."""
        cycle_start = datetime.now(UTC).isoformat()
        res_status = self.check_system_resources(max_ram_percent=max_ram_percent)
        if not res_status["can_train"]:
            self.audit.log("learning_cycle_aborted", actor="scheduler", status="failed", details=res_status)
            raise RuntimeError(f"Insufficient resources to execute learning cycle: {res_status}")

        # 1. Fetch highest priority task
        pending_tasks = self.queue.list_tasks(status="pending")
        if not pending_tasks:
            return {"status": "no_pending_tasks", "message": "Queue has no tasks awaiting learning."}

        task = pending_tasks[0]
        task_id = task["id"]
        self.audit.log("learning_cycle_started", actor="scheduler", target=task_id, details={"task": task["title"]})

        # 2. Execute bounded research
        research_report = self.engine.execute_task(task_id, source_documents)

        # 3. Formulate training candidate instructions from extracted facts
        facts = self.store.search(task["query"], limit=3)
        candidate_records = []
        for i, f in enumerate(facts):
            candidate_records.append({
                "id": f"autolearn_{task_id[:6]}_{i}",
                "category": "qa",
                "messages": [
                    {"role": "system", "content": "You are AAROHAN, an independent AI assistant."},
                    {"role": "user", "content": f"Tell me about {task['query']}."},
                    {"role": "assistant", "content": f["content"]},
                ],
            })

        # 4. Train continual learning candidate with replay
        train_res = train_continual_candidate(
            base_checkpoint_path=base_checkpoint,
            tokenizer_path=tokenizer_path,
            new_records=candidate_records,
            replay_records_file=historical_replay_file,
            output_checkpoint_path=output_candidate_path,
            epochs=5,
            replay_ratio=0.5,
            device_name=device_name,
        )

        # 5. Evaluate regression gate on historical benchmark
        benchmark_records = []
        with historical_replay_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    benchmark_records.append(json.loads(line))

        tok = load_tokenizer(tokenizer_path)
        dev = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")

        b_data = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        m_base = AAROHANLM(AAROHANConfig(**b_data["config"])).to(dev)
        m_base.load_state_dict(b_data["model_state"])
        base_loss, _ = evaluate_benchmark_loss(m_base, tok, benchmark_records, device=dev)

        c_data = torch.load(output_candidate_path, map_location="cpu", weights_only=False)
        m_cand = AAROHANLM(AAROHANConfig(**c_data["config"])).to(dev)
        m_cand.load_state_dict(c_data["model_state"])
        cand_loss, _ = evaluate_benchmark_loss(m_cand, tok, benchmark_records, device=dev)

        gate = RegressionGate(max_allowed_regression_percent=8.0)
        gate_res = gate.evaluate(base_loss, cand_loss)

        # 6. Save Cycle Report
        report_data = {
            "cycle_start": cycle_start,
            "cycle_completed": datetime.now(UTC).isoformat(),
            "task_id": task_id,
            "task_title": task["title"],
            "research_report": research_report,
            "training_result": train_res,
            "regression_gate": gate_res,
            "deployment_status": "STAGED_AWAITING_APPROVAL" if gate_res["passed"] else "REJECTED_REGRESSION",
        }

        report_file = self.artifacts_dir / f"learning_cycle_{task_id[:8]}.json"
        report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        self.audit.log(
            "learning_cycle_completed",
            actor="scheduler",
            target=task_id,
            status="success" if gate_res["passed"] else "regression_warning",
            details={"gate_status": gate_res["status"], "report_file": str(report_file)},
        )

        return report_data
