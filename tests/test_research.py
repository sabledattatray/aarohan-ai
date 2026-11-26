from aarohan.memory import MemoryStore
from aarohan.research import BoundedResearchEngine, KnowledgeGapDetector, LearningQueue
from scripts.research_cli import run


def test_learning_queue_lifecycle(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    queue = LearningQueue(store)

    # 1. Create task
    task = queue.create_task(
        title="Learn RoPE details",
        query="rotary positional embeddings",
        gap_type="manual_learning_goal",
        priority=0.85,
        objective="Extract 2D rotation formulas",
        max_budget_docs=3,
    )
    assert task["title"] == "Learn RoPE details"
    assert task["status"] == "pending"
    assert task["priority"] == 0.85

    # 2. List tasks
    pending = queue.list_tasks(status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == task["id"]

    # 3. Update task
    updated = queue.update_task_status(task["id"], "in_progress")
    assert updated["status"] == "in_progress"

    completed = queue.update_task_status(task["id"], "completed", result_summary="Extracted RoPE mechanics.")
    assert completed["status"] == "completed"
    assert completed["result_summary"] == "Extracted RoPE mechanics."


def test_knowledge_gap_detector_identifies_missing_relations(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    queue = LearningQueue(store)
    detector = KnowledgeGapDetector(store, queue)

    # 1. Add two entities without a connecting relationship
    store.add_entity("RMSNorm", entity_type="normalization")
    store.add_entity("SwiGLU", entity_type="activation")

    # 2. Add multiple memory chunks where both RMSNorm and SwiGLU co-occur
    store.add("AAROHAN combines RMSNorm and SwiGLU in each transformer layer block.", source="doc1")
    store.add("Both RMSNorm and SwiGLU optimize forward throughput on the RTX 2060.", source="doc2")

    # 3. Detect gaps
    proposed_tasks = detector.detect_missing_graph_relationships(min_priority=0.5)
    assert len(proposed_tasks) >= 1
    assert "RMSNorm" in proposed_tasks[0]["title"]
    assert "SwiGLU" in proposed_tasks[0]["title"]
    assert proposed_tasks[0]["gap_type"] == "missing_relationship"


def test_bounded_research_engine_execution(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    queue = LearningQueue(store)
    artifacts_dir = tmp_path / "artifacts"
    engine = BoundedResearchEngine(store, queue, artifacts_dir=artifacts_dir)

    # 1. Create approved task
    task = queue.create_task(
        title="Research Causal Attention",
        query="causal self attention masking",
        priority=0.9,
        max_budget_docs=2,
    )

    # 2. Prepare sample source documents
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(
        "# Causal Attention\n\nCausal Self Attention prevents looking into the future by applying an upper triangular mask.\n\n"
        "It preserves autoregressive order across all sequence tokens.",
        encoding="utf-8",
    )
    doc2 = tmp_path / "doc2.md"
    doc2.write_text(
        "# Irrelevant Notes\n\nThis note does not discuss attention mechanisms.",
        encoding="utf-8",
    )

    # 3. Execute research task
    report = engine.execute_task(task["id"], [doc1, doc2])
    assert report["task_id"] == task["id"]
    assert report["documents_scanned"] == 2
    assert report["facts_extracted"] >= 1
    assert "Causal" in report["entities_discovered"] or "Attention" in report["entities_discovered"]

    # 4. Check that proposed semantic memories were created as unverified
    assert len(report["proposed_memory_ids"]) >= 1
    mem = store.get(report["proposed_memory_ids"][0])
    assert mem is not None
    assert mem["verification_status"] == "unverified"
    assert mem["memory_type"] == "semantic"

    # 5. Check task is completed in queue
    task_after = queue.get_task(task["id"])
    assert task_after is not None
    assert task_after["status"] == "completed"
    assert task_after["research_artifact_path"] is not None


def test_research_cli_end_to_end(tmp_path) -> None:
    db = tmp_path / "memory.sqlite"
    artifacts = tmp_path / "artifacts"

    # 1. CLI Task Create
    created = run(["task-create", "--title", "CLI Research Task", "--query", "transformer heads", "--priority", "0.75"], database_path=db)
    task_id = created["id"]
    assert task_id

    # 2. CLI Task List
    listing = run(["task-list", "--status", "pending"], database_path=db)
    assert len(listing["tasks"]) >= 1

    # 3. CLI Research Run
    doc = tmp_path / "heads.txt"
    doc.write_text("Multi-head transformer heads divide width into parallel projections for diverse attention focus.\n\nKey query value tensors.", encoding="utf-8")
    run_res = run(["research-run", "--task-id", task_id, "--docs", str(doc), "--artifacts-dir", str(artifacts)], database_path=db)
    assert run_res["task_id"] == task_id
    assert run_res["facts_extracted"] >= 1
