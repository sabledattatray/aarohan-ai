import pytest

from aarohan.memory import MemoryStore


def test_memory_persists_and_searches_with_source_metadata(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    first_session = MemoryStore(database)
    saved = first_session.add("AAROHAN uses a local Transformer model.", source="project notes")
    assert saved["verification_status"] == "unverified"

    second_session = MemoryStore(database)
    matches = second_session.search("local Transformer")
    assert len(matches) == 1
    assert matches[0]["id"] == saved["id"]
    assert matches[0]["source"] == "project notes"


def test_verified_memory_requires_source_and_reviewer(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="require a source"):
        store.add("A sourced fact.", verification_status="verified", verified_by="reviewer")
    verified = store.add("A sourced fact.", source="manual page", verification_status="verified", verified_by="reviewer")
    assert verified["verification_status"] == "verified"
    assert len(store.search("sourced fact", verified_only=True)) == 1


def test_sensitive_memory_is_blocked_at_storage_layer(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="Sensitive memories are blocked"):
        store.add("private secret", privacy_classification="sensitive")


def test_duplicate_can_be_promoted_only_with_verification_evidence(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    unverified = store.add("The value is twelve.")
    duplicate = store.add(" The value is twelve. ", source="checked record", verification_status="verified", verified_by="adult reviewer")
    assert duplicate["id"] == unverified["id"]
    assert duplicate["verification_status"] == "verified"
    assert duplicate["source"] == "checked record"


def test_correction_marks_old_record_and_delete_removes_search_result(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    old = store.add("The count is eleven.", source="old note", verification_status="verified", verified_by="reviewer")
    corrected = store.correct(old["id"], "The count is twelve.", source="checked source", corrected_by="reviewer")
    old_record = store.get(old["id"])
    assert old_record is not None
    assert old_record["verification_status"] == "superseded"
    assert old_record["superseded_by"] == corrected["id"]
    assert [row["id"] for row in store.search("count", verified_only=True)] == [corrected["id"]]
    assert store.delete(corrected["id"]) is True
    assert store.search("twelve") == []


def test_multi_layer_memory_storage_and_vector_search(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    # 1. Add memories across different layers
    mem1 = store.add("Rotary positional embeddings encode token positions.", memory_type="semantic", source="doc1", verification_status="verified", verified_by="reviewer")
    mem2 = store.add("How to clean training text data with deduplication.", memory_type="procedural", source="doc2")
    mem3 = store.add("Failed to predict next token on step 45.", memory_type="learning", source="run_log")

    assert mem1["memory_type"] == "semantic"
    assert mem2["memory_type"] == "procedural"
    assert mem3["memory_type"] == "learning"

    # 2. Vector search retrieves semantically close match
    results = store.vector_search("rotary positions and tokens", limit=3, min_similarity=0.2)
    assert len(results) >= 1
    assert results[0]["id"] == mem1["id"]
    assert "similarity" in results[0]
    assert results[0]["similarity"] > 0.2

    # 3. Hybrid search combines keyword and vector
    hybrid_res = store.hybrid_search("token positions", limit=3)
    assert len(hybrid_res) >= 1
    assert any(r["id"] == mem1["id"] for r in hybrid_res)


def test_knowledge_graph_entities_relations_and_paths(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")

    # 1. Entities
    ent_trans = store.add_entity("Transformer", entity_type="architecture", description="Neural network architecture")
    ent_attn = store.add_entity("SelfAttention", entity_type="mechanism", description="Attention mechanism")
    ent_aarohan = store.add_entity("AAROHAN", entity_type="model", description="Independent AI model")

    assert ent_trans["name"] == "Transformer"
    assert ent_attn["name"] == "SelfAttention"
    assert ent_aarohan["name"] == "AAROHAN"
    assert len(store.list_entities()) == 3

    # 2. Relations
    rel1 = store.add_relation("SelfAttention", "Transformer", relation_type="part_of", confidence=1.0)
    rel2 = store.add_relation("AAROHAN", "Transformer", relation_type="implements", confidence=1.0)

    assert rel1["relation_type"] == "part_of"
    assert rel2["relation_type"] == "implements"

    # 3. Graph query
    relations = store.get_relations("Transformer")
    assert len(relations) == 2

    # 4. Path finding: SelfAttention -> Transformer
    paths = store.find_path("SelfAttention", "Transformer")
    assert len(paths) >= 1
    assert paths[0] == ["SelfAttention", "part_of", "Transformer"]


def test_contradiction_detection_and_consolidation(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")

    # 1. Add episodic memories with recurring concepts
    store.add("Rotary embeddings are applied in the transformer attention blocks.", memory_type="episodic")
    store.add("Rotary embeddings encode token distances accurately.", memory_type="episodic")

    # 2. Run consolidation
    summary = store.consolidate(recent_limit=10)
    assert summary["episodic_records_inspected"] >= 2
    assert summary["new_entities_created"] >= 1  # "Rotary" or "Embeddings" recurring

    # 3. Contradiction detection: opposing polarity
    contra = store.detect_contradictions("Rotary embeddings are not applied in transformer blocks", threshold=0.5)
    assert len(contra) >= 1
    assert "Polarity mismatch" in contra[0]["reason"]
