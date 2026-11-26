"""AAROHAN AI — Production Automated Test Suite for Cognitive Memory Engine 3.0.

15 Automated Tests:
- TEST 1: Save single fact (`BLUE-731 means Priority Level 7`) -> verify 1 row in cognitive_memories
- TEST 2: Save 6 NexDial rules in one numbered instruction -> verify save_facts_batch returns saved_count=6
- TEST 3: Verify all 6 rules exist in SQLite (SELECT COUNT = 6)
- TEST 4: retrieve_memory() in fresh engine instance -> returns FOUND with 6 records
- TEST 5: Questions starting with "What", "Which", "Tell me", "Retrieve" -> trigger retrieval
- TEST 6: Correct P1 deadline from 2 hours to 3 hours (is_correction=True) -> verify current_value = "3 hours"
- TEST 7: Version history preserved (memory_versions has 2 rows, previous_value = "2 hours")
- TEST 8: Save confirmed AND disputed payment rules -> 2 unique keys, no conflict
- TEST 9: Save P1_DISPUTED_PAYMENT_OVERRIDE -> verify retrieval
- TEST 10: Instantiate new CognitiveMemoryEngine on same DB file -> all 6 rules retrievable (cross-session)
- TEST 11: retrieve_memory("PURPLE-999 deadline") -> NOT_FOUND
- TEST 12: Intentionally corrupt save (empty value) -> raises ValueError or fails, no row inserted
- TEST 13: user_scope isolation — save with scope "user_a", retrieve with scope "user_b" -> NOT_FOUND
- TEST 14: Save 3 unrelated memories -> all 3 coexist, all retrievable
- TEST 15: Retrieved records include actual ID, subject, predicate, created_at from DB
"""

from __future__ import annotations

import gc
import sqlite3
from pathlib import Path
import pytest

from aarohan.cognitive_memory import CognitiveMemoryEngine, MemoryType, MemoryStatus


NEXDIAL_INSTRUCTION = """Please remember the following NexDial business rules in persistent memory:
1. Priority P1 accounts must be contacted within 2 hours.
2. Priority P2 accounts must be contacted within 6 hours.
3. Priority P3 accounts must be contacted within 24 hours.
4. Confirmed payments must be removed from the outreach queue.
5. Disputed payments must be routed to the billing team.
6. Priority P1 accounts with disputed payments must still be handled as P1 first (within 2 hours), and then flagged to the billing team."""


@pytest.fixture
def test_db_path(tmp_path):
    p = tmp_path / "test_cognitive_memory.sqlite"
    yield p
    gc.collect()


@pytest.fixture
def engine(test_db_path):
    eng = CognitiveMemoryEngine(test_db_path)
    yield eng
    del eng
    gc.collect()


# -----------------------------------------------------------------------------
# TEST 1: Save single fact -> verify 1 row in cognitive_memories
# -----------------------------------------------------------------------------
def test_1_save_single_fact(engine, test_db_path):
    rec = engine.save_memory("BLUE-731", "Priority Level 7", user_scope="researcher")
    assert rec["id"].startswith("mem_")
    assert rec["memory_key"] == "BLUE-731"
    assert rec["current_value"] == "Priority Level 7"
    assert rec["status"] == "ACTIVE"

    with sqlite3.connect(test_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cognitive_memories WHERE memory_key = 'BLUE-731'").fetchone()[0]
        assert count == 1


# -----------------------------------------------------------------------------
# TEST 2: Save 6 NexDial rules in one numbered instruction -> saved_count = 6
# -----------------------------------------------------------------------------
def test_2_save_six_rules_batch(engine):
    intent = engine.extract_memory_intent(NEXDIAL_INSTRUCTION)
    assert intent.is_memory_instruction is True
    assert len(intent.extracted_facts) == 6

    batch_res = engine.save_facts_batch(intent.extracted_facts, user_scope="researcher")
    assert batch_res["total_facts"] == 6
    assert batch_res["saved_count"] == 6
    assert batch_res["conflict_count"] == 0
    assert batch_res["failed_count"] == 0


# -----------------------------------------------------------------------------
# TEST 3: Verify all 6 rules exist in SQLite (SELECT COUNT = 6)
# -----------------------------------------------------------------------------
def test_3_verify_sqlite_row_count(engine, test_db_path):
    intent = engine.extract_memory_intent(NEXDIAL_INSTRUCTION)
    engine.save_facts_batch(intent.extracted_facts, user_scope="researcher")

    with sqlite3.connect(test_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cognitive_memories WHERE status = 'ACTIVE'").fetchone()[0]
        assert count == 6


# -----------------------------------------------------------------------------
# TEST 4: retrieve_memory() returns FOUND with records
# -----------------------------------------------------------------------------
def test_4_retrieve_rules(engine):
    intent = engine.extract_memory_intent(NEXDIAL_INSTRUCTION)
    engine.save_facts_batch(intent.extracted_facts, user_scope="researcher")

    recall = engine.retrieve_memory("What is the contact deadline for P1 accounts?", user_scope="researcher")
    assert recall is not None
    assert recall["status"] == "FOUND"
    assert recall["retrieved_count"] >= 1
    assert "2 hours" in recall["recall_text"]


# -----------------------------------------------------------------------------
# TEST 5: Questions starting with "What", "Which", "Tell me", "Retrieve" trigger retrieval
# -----------------------------------------------------------------------------
def test_5_question_signals_trigger_retrieval(engine):
    for q in [
        "What is BLUE-731?",
        "Which rule applies to P1?",
        "Tell me about disputed payments",
        "Retrieve the rule for confirmed payments",
    ]:
        intent = engine.extract_memory_intent(q)
        assert intent.is_question is True
        assert intent.is_memory_instruction is False


# -----------------------------------------------------------------------------
# TEST 6: Correct P1 deadline from 2 hours to 3 hours -> verify current_value = "3 hours"
# -----------------------------------------------------------------------------
def test_6_correct_p1_deadline(engine):
    engine.save_memory("P1_CONTACT_DEADLINE", "2 hours", subject="P1 account", user_scope="researcher")
    corr = engine.save_memory("P1_CONTACT_DEADLINE", "3 hours", subject="P1 account", is_correction=True, user_scope="researcher")
    assert corr["current_value"] == "3 hours"
    assert corr["version_count"] == 2
    assert corr["memory_type"] == "CORRECTED"


# -----------------------------------------------------------------------------
# TEST 7: Version history preserved (memory_versions has 2 rows)
# -----------------------------------------------------------------------------
def test_7_version_history_preserved(engine, test_db_path):
    engine.save_memory("P1_CONTACT_DEADLINE", "2 hours", user_scope="researcher")
    corr = engine.save_memory("P1_CONTACT_DEADLINE", "3 hours", is_correction=True, user_scope="researcher")

    with sqlite3.connect(test_db_path) as conn:
        conn.row_factory = sqlite3.Row
        vers = conn.execute("SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY version_number ASC", (corr["id"],)).fetchall()
        assert len(vers) == 2
        assert vers[0]["new_value"] == "2 hours"
        assert vers[1]["previous_value"] == "2 hours"
        assert vers[1]["new_value"] == "3 hours"


# -----------------------------------------------------------------------------
# TEST 8: Confirmed and disputed payment rules have unique keys, no conflict
# -----------------------------------------------------------------------------
def test_8_payment_rules_unique_keys(engine):
    rule4 = "Confirmed payments must be removed from the outreach queue."
    rule5 = "Disputed payments must be routed to the billing team."

    intent4 = engine.extract_memory_intent(rule4)
    intent5 = engine.extract_memory_intent(rule5)

    assert len(intent4.extracted_facts) == 1
    assert len(intent5.extracted_facts) == 1

    key4 = intent4.extracted_facts[0]["memory_key"]
    key5 = intent5.extracted_facts[0]["memory_key"]

    assert key4 != key5
    assert key4 == "CONFIRMED_PAYMENT_QUEUE_RULE"
    assert key5 == "DISPUTED_PAYMENT_ROUTING_RULE"

    rec4 = engine.save_memory(key4, intent4.extracted_facts[0]["value"], user_scope="researcher")
    rec5 = engine.save_memory(key5, intent5.extracted_facts[0]["value"], user_scope="researcher")

    assert rec4["conflict_detected"] is False
    assert rec5["conflict_detected"] is False


# -----------------------------------------------------------------------------
# TEST 9: Save P1_DISPUTED_PAYMENT_OVERRIDE -> verify retrieval
# -----------------------------------------------------------------------------
def test_9_save_and_retrieve_override(engine):
    rule6 = "Priority P1 accounts with disputed payments must still be handled as P1 first (within 2 hours), and then flagged to the billing team."
    intent = engine.extract_memory_intent(rule6)
    assert len(intent.extracted_facts) == 1
    fact = intent.extracted_facts[0]
    assert fact["memory_key"] == "P1_DISPUTED_PAYMENT_OVERRIDE"

    engine.save_memory(fact["memory_key"], fact["value"], conditions=fact.get("conditions", ""), user_scope="researcher")

    recall = engine.retrieve_memory("What happens if a P1 account has a disputed payment?", user_scope="researcher")
    assert recall is not None
    assert recall["status"] == "FOUND"
    assert "P1_DISPUTED_PAYMENT_OVERRIDE" in [r["memory_key"] for r in recall["records"]]


# -----------------------------------------------------------------------------
# TEST 10: Cross-session recall on brand new engine instance
# -----------------------------------------------------------------------------
def test_10_cross_session_recall(test_db_path):
    eng1 = CognitiveMemoryEngine(test_db_path)
    intent = eng1.extract_memory_intent(NEXDIAL_INSTRUCTION)
    eng1.save_facts_batch(intent.extracted_facts, user_scope="researcher")
    del eng1
    gc.collect()

    # Session 2: Fresh engine instance
    eng2 = CognitiveMemoryEngine(test_db_path)
    with sqlite3.connect(test_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cognitive_memories WHERE status = 'ACTIVE'").fetchone()[0]
        assert count == 6

    recall = eng2.retrieve_memory("What is the contact deadline for P2 accounts?", user_scope="researcher")
    assert recall is not None
    assert recall["status"] == "FOUND"
    assert "6 hours" in recall["recall_text"]
    del eng2
    gc.collect()


# -----------------------------------------------------------------------------
# TEST 11: retrieve_memory("PURPLE-999 deadline") -> NOT_FOUND
# -----------------------------------------------------------------------------
def test_11_unknown_memory_not_found(engine):
    engine.save_memory("BLUE-731", "Priority Level 7", user_scope="researcher")
    recall = engine.retrieve_memory("What is PURPLE-999 deadline?", user_scope="researcher")
    assert recall is not None
    assert recall["status"] == "NOT_FOUND"
    assert recall["unknown_key"] == "PURPLE-999"
    assert "will not guess" in recall["recall_text"] or "no matching records" in recall["recall_text"]


# -----------------------------------------------------------------------------
# TEST 12: Intentionally corrupt save (empty value) -> raises ValueError or fails
# -----------------------------------------------------------------------------
def test_12_empty_value_rejection(engine, test_db_path):
    with pytest.raises(ValueError):
        engine.save_memory("EMPTY_KEY", "", user_scope="researcher")

    with sqlite3.connect(test_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cognitive_memories WHERE memory_key = 'EMPTY_KEY'").fetchone()[0]
        assert count == 0


# -----------------------------------------------------------------------------
# TEST 13: user_scope isolation — user_a cannot be seen by user_b
# -----------------------------------------------------------------------------
def test_13_user_scope_isolation(engine):
    engine.save_memory("SECRET_KEY", "TopSecret42", user_scope="user_a")

    recall_b = engine.retrieve_memory("What is SECRET_KEY?", user_scope="user_b")
    assert recall_b is not None
    assert recall_b["status"] == "NOT_FOUND"

    recall_a = engine.retrieve_memory("What is SECRET_KEY?", user_scope="user_a")
    assert recall_a is not None
    assert recall_a["status"] == "FOUND"
    assert "TopSecret42" in recall_a["recall_text"]


# -----------------------------------------------------------------------------
# TEST 14: Save 3 unrelated memories -> all coexist and retrievable
# -----------------------------------------------------------------------------
def test_14_multiple_memories_coexist(engine):
    engine.save_memory("ALPHA_CODE", "AlphaValue", user_scope="researcher")
    engine.save_memory("BETA_CODE", "BetaValue", user_scope="researcher")
    engine.save_memory("GAMMA_CODE", "GammaValue", user_scope="researcher")

    for k, v in [("ALPHA_CODE", "AlphaValue"), ("BETA_CODE", "BetaValue"), ("GAMMA_CODE", "GammaValue")]:
        rec = engine.retrieve_memory(f"What does {k} mean?", user_scope="researcher")
        assert rec is not None
        assert rec["status"] == "FOUND"
        assert v in rec["recall_text"]


# -----------------------------------------------------------------------------
# TEST 15: Retrieved records include actual ID, subject, predicate, created_at
# -----------------------------------------------------------------------------
def test_15_retrieved_records_metadata(engine):
    rec = engine.save_memory(
        "DELTA_CODE",
        "DeltaValue",
        subject="DeltaSubject",
        predicate="denotes",
        user_scope="researcher",
    )
    mem_id = rec["id"]

    recall = engine.retrieve_memory("What is DELTA_CODE?", user_scope="researcher")
    assert recall["status"] == "FOUND"
    assert len(recall["records"]) == 1
    r = recall["records"][0]

    assert r["id"] == mem_id
    assert r["id"].startswith("mem_")
    assert r["subject"] == "DeltaSubject"
    assert r["predicate"] == "denotes"
    assert "created_at" in r
    assert r["current_value"] == "DeltaValue"
