"""
Unit and regression test suite for Phase 4: Cognitive Memory Engine.
Validates:
1. Multi-user session isolation (no cross-tenant memory leakage).
2. Content digest collision avoidance across user scopes.
3. Fact vs. assumption classification.
4. Automatic preference & instruction extraction (extract_and_remember).
5. Memory conflict resolution (superseding stale records).
6. TTL expiration and purge_expired logic.
7. Six-layer memory support (working, episodic, semantic, procedural, reflective, learning).
"""

import time
import pytest
from pathlib import Path
import tempfile
from aarohan.memory import MemoryStore, VALID_MEMORY_TYPES


@pytest.fixture
def temp_memory_store(tmp_path):
    """Create an isolated, temporary SQLite memory store."""
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(database_path=db_path)
    yield store


def test_six_memory_layers_supported(temp_memory_store):
    expected_layers = {"working", "episodic", "semantic", "procedural", "reflective", "learning"}
    assert expected_layers.issubset(VALID_MEMORY_TYPES)

    for layer in expected_layers:
        item = temp_memory_store.add(
            content=f"Test layer item for {layer}",
            source="test_runner",
            memory_type=layer,
            confidence=0.9,
            user_scope="alice",
        )
        assert item["id"] is not None
        assert item["memory_type"] == layer


def test_multi_user_isolation(temp_memory_store):
    # Alice stores a sensitive personal secret
    alice_item = temp_memory_store.add(
        content="Alice's confidential pin is 4321",
        source="alice_input",
        user_scope="alice",
        memory_type="semantic",
        confidence=1.0,
    )

    # Bob stores his favorite hobby
    bob_item = temp_memory_store.add(
        content="Bob loves playing chess in the park",
        source="bob_input",
        user_scope="bob",
        memory_type="semantic",
        confidence=1.0,
    )

    # 1. Direct get isolation
    assert temp_memory_store.get(alice_item["id"], user_scope="alice") is not None
    assert temp_memory_store.get(alice_item["id"], user_scope="bob") is None
    assert temp_memory_store.get(bob_item["id"], user_scope="bob") is not None
    assert temp_memory_store.get(bob_item["id"], user_scope="alice") is None

    # 2. Keyword search isolation
    alice_search_results = temp_memory_store.search("pin", user_scope="alice", include_global=False)
    assert any("4321" in r["content"] for r in alice_search_results)

    bob_search_results = temp_memory_store.search("pin", user_scope="bob", include_global=False)
    assert not any("4321" in r["content"] for r in bob_search_results)

    # 3. Vector & hybrid search isolation
    bob_hybrid_results = temp_memory_store.hybrid_search("confidential pin", user_scope="bob", include_global=False)
    assert not any("4321" in r["content"] for r in bob_hybrid_results)

    alice_hybrid_results = temp_memory_store.hybrid_search("confidential pin", user_scope="alice", include_global=False)
    assert any("4321" in r["content"] for r in alice_hybrid_results)


def test_digest_user_scope_collision_avoidance(temp_memory_store):
    identical_content = "I reside in Mumbai"

    # Alice and Bob store identical statement
    item1 = temp_memory_store.add(
        content=identical_content,
        source="user_input",
        user_scope="alice",
        memory_type="semantic",
    )

    item2 = temp_memory_store.add(
        content=identical_content,
        source="user_input",
        user_scope="bob",
        memory_type="semantic",
    )

    assert item1["id"] != item2["id"]
    assert item1["user_scope"] == "alice"
    assert item2["user_scope"] == "bob"


def test_fact_vs_assumption_classification(temp_memory_store):
    fact_text = "The speed of light in a vacuum is 299,792,458 meters per second."
    assumption_text = "I guess it might rain tomorrow, or maybe next week."
    preference_text = "I believe the server might be overloaded, possibly."

    fact_type1, conf1 = temp_memory_store.classify_fact_or_assumption(fact_text)
    assert fact_type1 == "fact"
    assert conf1 >= 0.85

    fact_type2, conf2 = temp_memory_store.classify_fact_or_assumption(assumption_text)
    assert fact_type2 == "assumption"
    assert conf2 <= 0.55

    fact_type3, conf3 = temp_memory_store.classify_fact_or_assumption(preference_text)
    assert fact_type3 == "assumption"


def test_auto_extraction_and_remember(temp_memory_store):
    msg = "Please remember that I prefer dark mode and my favorite language is Python."
    remembered = temp_memory_store.extract_and_remember(msg, user_scope="charlie")

    assert len(remembered) >= 1
    found_dark_mode = any("dark mode" in r["content"].lower() or "python" in r["content"].lower() for r in remembered)
    assert found_dark_mode

    # Verify charlie can search it
    results = temp_memory_store.search("dark mode", user_scope="charlie")
    assert len(results) > 0


def test_conflict_resolution(temp_memory_store):
    # Old memory
    old_item = temp_memory_store.add(
        content="User's primary phone is 111-222-3333",
        source="user_profile",
        user_scope="david",
        memory_type="semantic",
    )

    # Resolve conflict with new memory
    new_item = temp_memory_store.resolve_conflict(
        memory_id_to_supersede=old_item["id"],
        new_content="User's primary phone updated to 999-888-7777",
        user_scope="david",
        source="user_override",
    )

    # Check old item is superseded
    stale = temp_memory_store.get(old_item["id"], user_scope="david")
    assert stale["verification_status"] == "superseded"
    assert stale["superseded_by"] == new_item["id"]

    # Check new item is active
    active = temp_memory_store.get(new_item["id"], user_scope="david")
    assert active["verification_status"] == "verified"


def test_ttl_expiration_and_purge(temp_memory_store):
    # Add a short-lived working memory with 1 second TTL
    ephemeral = temp_memory_store.add(
        content="Temporary OTP verification token #987654",
        source="auth_flow",
        user_scope="eve",
        memory_type="working",
        ttl_seconds=1,
    )

    # Immediately searchable
    immediate = temp_memory_store.search("OTP verification token", user_scope="eve")
    assert any("987654" in r["content"] for r in immediate)

    # Wait for expiration
    time.sleep(1.2)

    # Should no longer be visible in search
    after_expiry = temp_memory_store.search("OTP verification token", user_scope="eve")
    assert not any("987654" in r["content"] for r in after_expiry)

    # Purge expired items
    purged_count = temp_memory_store.purge_expired(user_scope="eve")
    assert purged_count >= 1

    # Raw fetch should confirm it is removed from DB
    assert temp_memory_store.get(ephemeral["id"], user_scope="eve") is None
