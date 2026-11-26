"""
Unit and regression test suite for Phase 5: Knowledge Graph Engine.
Validates:
1. Entity extraction, normalization, and duplicate entity prevention.
2. Relationship extraction with explicit semantic markers (no spurious co-occurrence edges).
3. Verified facts vs. inferred relationships distinction.
4. Multi-hop graph traversal (path finding).
5. Temporal relationships and source attribution.
6. Graph contradiction and conflict detection.
"""

import pytest
from aarohan.memory import MemoryStore, VALID_RELATION_TYPES, VALID_RELATION_CLASSES


@pytest.fixture
def temp_memory_store(tmp_path):
    """Create an isolated SQLite memory store for knowledge graph tests."""
    db_path = tmp_path / "test_graph.db"
    store = MemoryStore(database_path=db_path)
    yield store


def test_relation_types_and_classes_supported():
    required_types = {"developed", "uses", "contains", "is_a", "depends_on", "part_of", "contradicts"}
    assert required_types.issubset(VALID_RELATION_TYPES)
    assert {"verified", "inferred"}.issubset(VALID_RELATION_CLASSES)


def test_entity_normalization_and_duplicate_prevention(temp_memory_store):
    # 1. Add canonical entity
    e1 = temp_memory_store.add_entity("Datta Sable", entity_type="person", description="Creator of AAROHAN AI")
    assert e1["id"] is not None
    assert e1["name"] == "Datta Sable"

    # 2. Add with differing whitespace and lowercase
    e2 = temp_memory_store.add_entity("  datta sable  ", entity_type="person")
    assert e2["id"] == e1["id"], "Case and whitespace variance must reuse existing entity"

    # 3. Add alias
    e3 = temp_memory_store.add_entity("Datta Sable", aliases=["D. Sable", "Datta"])
    assert e3["id"] == e1["id"]

    # 4. Lookup by alias
    by_alias = temp_memory_store.get_entity("D. Sable")
    assert by_alias is not None
    assert by_alias["id"] == e1["id"]


def test_canonical_graph_relationships_and_classes(temp_memory_store):
    # Phase 5 Canonical Architecture Triples:
    # Datta Sable -> developed -> AAROHAN AI
    # AAROHAN AI -> uses -> Qwen foundation model
    # AAROHAN AI -> contains -> Cognitive Memory Engine

    r1 = temp_memory_store.add_relation(
        "Datta Sable",
        "AAROHAN AI",
        "developed",
        confidence=1.0,
        source="official_credits",
        relation_class="verified",
    )
    assert r1["relation_class"] == "verified"
    assert r1["confidence"] == 1.0

    r2 = temp_memory_store.add_relation(
        "AAROHAN AI",
        "Qwen foundation model",
        "uses",
        confidence=0.95,
        source="system_architecture",
        relation_class="verified",
    )
    assert r2["relation_class"] == "verified"

    r3 = temp_memory_store.add_relation(
        "AAROHAN AI",
        "Cognitive Memory Engine",
        "contains",
        confidence=1.0,
        source="system_architecture",
        relation_class="verified",
    )
    assert r3["relation_class"] == "verified"

    # Inferred relationship test
    r_inf = temp_memory_store.add_relation(
        "AAROHAN AI",
        "FastAPI",
        "uses",
        confidence=0.6,
        source="heuristics",
        relation_class="inferred",
    )
    assert r_inf["relation_class"] == "inferred"
    assert r_inf["confidence"] == 0.6


def test_multi_hop_graph_traversal(temp_memory_store):
    # Setup chain:
    # Datta Sable -> developed -> AAROHAN AI -> contains -> Cognitive Memory Engine
    temp_memory_store.add_relation("Datta Sable", "AAROHAN AI", "developed", confidence=1.0)
    temp_memory_store.add_relation("AAROHAN AI", "Cognitive Memory Engine", "contains", confidence=1.0)

    # 2-hop traversal
    paths = temp_memory_store.find_path("Datta Sable", "Cognitive Memory Engine", max_depth=3)
    assert len(paths) >= 1
    expected_path = ["Datta Sable", "developed", "AAROHAN AI", "contains", "Cognitive Memory Engine"]
    assert expected_path in paths


def test_relation_extraction_without_spurious_cooccurrence(temp_memory_store):
    # Valid explicit relation sentence
    text_with_verbs = "Datta Sable developed AAROHAN AI. AAROHAN AI contains Cognitive Memory Engine."
    entities, relations = temp_memory_store.extract_entities_and_relations(text_with_verbs, relation_class="verified")

    assert len(relations) >= 2
    rel_types = [r["relation_type"] for r in relations]
    assert "developed" in rel_types
    assert "contains" in rel_types

    # Co-occurrence without relational verb
    cooccur_text = "Datta Sable and AAROHAN AI and Cognitive Memory Engine were mentioned together in the meeting."
    _, cooccur_rels = temp_memory_store.extract_entities_and_relations(cooccur_text)
    # MUST NOT invent an edge
    assert len(cooccur_rels) == 0, "Graph must never invent relations between co-occurring entities without explicit verbs"


def test_temporal_relationships(temp_memory_store):
    r = temp_memory_store.add_relation(
        "Qwen-0.5B",
        "AAROHAN AI",
        "used_in",
        confidence=0.9,
        source="benchmarks",
        relation_class="verified",
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2026-12-31T23:59:59Z",
    )
    assert r["valid_from"] == "2026-01-01T00:00:00Z"
    assert r["valid_until"] == "2026-12-31T23:59:59Z"


def test_graph_contradiction_detection(temp_memory_store):
    # Add an explicit contradiction edge
    temp_memory_store.add_relation(
        "Flat Earth Theory",
        "Planetary Science",
        "contradicts",
        confidence=1.0,
        source="astronomy_standards",
    )

    contradictions = temp_memory_store.detect_graph_contradictions("Flat Earth Theory")
    assert len(contradictions) >= 1
    assert contradictions[0]["type"] == "explicit_contradiction"
    assert contradictions[0]["target"] == "Planetary Science"
