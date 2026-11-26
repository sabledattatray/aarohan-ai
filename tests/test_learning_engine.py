"""AAROHAN AI — Production Automated Test Suite for Learning Engine 1.0.

15 Automated Regression Tests:
- TEST 1: Learn 5 fictional rules under LEARNING_TEST_2026_01
- TEST 2: Verify all 5 records exist in SQLite (SELECT COUNT = 5)
- TEST 3: Fresh session retrieve all 5 rules (cross-session recall)
- TEST 4: Apply rules to a scenario (LEVEL-A -> 24 hours deadline)
- TEST 5: Respect conditional overrides (ON-HOLD overrides LEVEL-A -> suspended)
- TEST 6: Reject unknown rule/key without guessing or fabricating
- TEST 7: Correct one rule and preserve previous version in learned_rule_versions
- TEST 8: Verify unrelated namespaces remain isolated (NAMESPACE_ALPHA vs NAMESPACE_BETA)
- TEST 9: Retrieval query never triggers a database write
- TEST 10: Ambiguous feedback returns CLARIFICATION_NEEDED without altering rules
- TEST 11: Feedback reordering ("Apply Rule 3 before Rule 2") updates precedence
- TEST 12: Both /api/chat and /api/chat/stream process learning instructions identically
- TEST 13: Failed storage never returns a success confirmation
- TEST 14: User scope isolation (user_alice vs user_bob)
- TEST 15: Backend restart persistence (records survive new LearningEngine instance on disk)
"""

from __future__ import annotations

import gc
import sqlite3

import pytest
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.learning import ExtractedRule, LearningEngine

LEARNING_PROMPT = """I am teaching you 5 new business rules for handling accounts.
Please learn and store these rules under namespace LEARNING_TEST_2026_01:

RULES:
1. LEVEL-A cases have response deadline of 24 hours.
2. LEVEL-B cases have response deadline of 48 hours.
3. LEVEL-C cases have response deadline of 72 hours.
4. If account has status ON-HOLD, all response deadlines are suspended and no contact may be made.
5. ON-HOLD condition overrides LEVEL-A, LEVEL-B, and LEVEL-C deadlines."""


@pytest.fixture
def test_db_path(tmp_path):
    p = tmp_path / "test_learning_engine.sqlite"
    yield p
    gc.collect()


@pytest.fixture
def engine(test_db_path):
    eng = LearningEngine(test_db_path)
    yield eng
    del eng
    gc.collect()


# -----------------------------------------------------------------------------
# TEST 1: Learn 5 fictional rules under LEARNING_TEST_2026_01
# -----------------------------------------------------------------------------
def test_1_learn_fictional_rules(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    assert res.is_learning_instruction is True
    assert res.namespace == "LEARNING_TEST_2026_01"
    assert len(res.extracted_rules) == 5

    batch = engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")
    assert batch["saved_count"] == 5
    assert batch["failed_count"] == 0
    assert batch["namespace"] == "LEARNING_TEST_2026_01"


# -----------------------------------------------------------------------------
# TEST 2: Verify all 5 records exist in SQLite (SELECT COUNT = 5)
# -----------------------------------------------------------------------------
def test_2_verify_records_in_sqlite(engine, test_db_path):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    with sqlite3.connect(str(test_db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM learned_rules WHERE namespace = 'LEARNING_TEST_2026_01' AND user_scope = 'tester'"
        ).fetchall()
        assert len(rows) == 5
        for r in rows:
            assert r["id"].startswith("rule_")
            assert r["status"] == "ACTIVE"
            assert r["version"] == 1
            assert len(r["rule_key"]) > 0
            assert len(r["value"]) > 0


# -----------------------------------------------------------------------------
# TEST 3: Fresh session retrieve all 5 rules (cross-session recall)
# -----------------------------------------------------------------------------
def test_3_fresh_session_retrieve_rules(test_db_path):
    # Save rules in first session
    eng1 = LearningEngine(test_db_path)
    res = eng1.extract_learning_intent(LEARNING_PROMPT)
    eng1.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")
    del eng1
    gc.collect()

    # Retrieve rules in fresh session
    eng2 = LearningEngine(test_db_path)
    recall = eng2.retrieve_learned_rules(
        "What are the rules in namespace LEARNING_TEST_2026_01?",
        user_scope="tester",
    )
    assert recall["status"] == "FOUND"
    assert recall["total_rules"] == 5
    assert len(recall["rules"]) == 5
    rule_keys = [r["rule_key"] for r in recall["rules"]]
    assert any("LEVEL_A" in k for k in rule_keys)
    assert any("ON_HOLD" in k for k in rule_keys)


# -----------------------------------------------------------------------------
# TEST 4: Apply rules to a scenario (LEVEL-A -> 24 hours deadline)
# -----------------------------------------------------------------------------
def test_4_apply_rules_to_scenario(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    benchmark = engine.evaluate_benchmark(
        [
            {"name": "Standard Level A Case", "input": "Case is priority LEVEL-A with normal active status.", "expected": "24 hours"}
        ],
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert benchmark["total_tests"] == 1
    assert benchmark["passed"] == 1
    assert benchmark["accuracy_pct"] == 100.0
    assert "24 hours" in benchmark["results"][0]["actual"]


# -----------------------------------------------------------------------------
# TEST 5: Respect conditional overrides (ON-HOLD overrides LEVEL-A -> suspended)
# -----------------------------------------------------------------------------
def test_5_respect_conditional_override(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    benchmark = engine.evaluate_benchmark(
        [
            {"name": "Overridden Level A Case", "input": "Account has priority LEVEL-A but status is ON-HOLD.", "expected": "suspended"}
        ],
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert benchmark["total_tests"] == 1
    assert benchmark["passed"] == 1
    assert benchmark["accuracy_pct"] == 100.0
    assert "suspended" in benchmark["results"][0]["actual"].lower()


# -----------------------------------------------------------------------------
# TEST 6: Reject unknown rule/key without guessing or fabricating
# -----------------------------------------------------------------------------
def test_6_reject_unknown_rule_without_guessing(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    recall = engine.retrieve_learned_rules(
        "What is the policy for UNKNOWN_TIER_9999?",
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert recall["status"] == "NOT_FOUND"
    assert len(recall["rules"]) == 0


# -----------------------------------------------------------------------------
# TEST 7: Correct one rule and preserve previous version in learned_rule_versions
# -----------------------------------------------------------------------------
def test_7_correct_rule_preserve_version(engine, test_db_path):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    fb = engine.apply_feedback(
        "In namespace LEARNING_TEST_2026_01, change LEVEL-A deadline to 12 hours.",
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert fb["status"] == "APPLIED"

    history = engine.get_rule_history(fb["applied_rule_id"], user_scope="tester")
    assert "LEVEL_A" in history["rule_key"]
    assert history["value"] == "12 hours"
    assert history["version"] == 2
    assert len(history["versions"]) >= 2
    corr_ver = next(v for v in history["versions"] if v["version_number"] == 2)
    assert corr_ver["previous_value"] == "24 hours"
    assert corr_ver["new_value"] == "12 hours"


# -----------------------------------------------------------------------------
# TEST 8: Verify unrelated namespaces remain isolated (ALPHA vs BETA)
# -----------------------------------------------------------------------------
def test_8_unrelated_namespaces_isolated(engine):
    rule_alpha = [
        ExtractedRule(rule_key="ALPHA_CODE", rule_text="Alpha code means Red", subject="ALPHA", predicate="means", value="Red")
    ]
    rule_beta = [
        ExtractedRule(rule_key="BETA_CODE", rule_text="Beta code means Blue", subject="BETA", predicate="means", value="Blue")
    ]
    engine.save_learned_rules_batch("NAMESPACE_ALPHA", rule_alpha, user_scope="tester")
    engine.save_learned_rules_batch("NAMESPACE_BETA", rule_beta, user_scope="tester")

    res_alpha = engine.retrieve_learned_rules("", namespace="NAMESPACE_ALPHA", user_scope="tester")
    res_beta = engine.retrieve_learned_rules("", namespace="NAMESPACE_BETA", user_scope="tester")

    assert res_alpha["total_rules"] == 1
    assert res_alpha["rules"][0]["rule_key"] == "ALPHA_CODE"

    assert res_beta["total_rules"] == 1
    assert res_beta["rules"][0]["rule_key"] == "BETA_CODE"


# -----------------------------------------------------------------------------
# TEST 9: Retrieval query never triggers a database write
# -----------------------------------------------------------------------------
def test_9_retrieval_query_does_not_write(engine, test_db_path):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    with sqlite3.connect(str(test_db_path)) as conn:
        before_count = conn.execute("SELECT COUNT(*) FROM learned_rules").fetchone()[0]
        before_vers = conn.execute("SELECT COUNT(*) FROM learned_rule_versions").fetchone()[0]

    # Run multiple retrieval queries
    engine.retrieve_learned_rules("What is LEVEL-A deadline?", user_scope="tester")
    engine.retrieve_learned_rules("Recall rules for namespace LEARNING_TEST_2026_01", user_scope="tester")
    engine.retrieve_learned_rules("Show all rules", user_scope="tester")

    with sqlite3.connect(str(test_db_path)) as conn:
        after_count = conn.execute("SELECT COUNT(*) FROM learned_rules").fetchone()[0]
        after_vers = conn.execute("SELECT COUNT(*) FROM learned_rule_versions").fetchone()[0]

    assert before_count == after_count
    assert before_vers == after_vers


# -----------------------------------------------------------------------------
# TEST 10: Ambiguous feedback returns CLARIFICATION_NEEDED without altering rules
# -----------------------------------------------------------------------------
def test_10_ambiguous_feedback_requires_clarification(engine, test_db_path):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    fb = engine.apply_feedback(
        "Make that rule faster please.",
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert fb["status"] == "CLARIFICATION_NEEDED"
    assert fb["applied_rule_id"] is None

    # Verify no rules were modified
    rules = engine.list_rules("LEARNING_TEST_2026_01", user_scope="tester")
    for r in rules:
        assert r["version"] == 1


# -----------------------------------------------------------------------------
# TEST 11: Feedback reordering ("Apply Rule 3 before Rule 2") updates precedence
# -----------------------------------------------------------------------------
def test_11_feedback_reordering_updates_precedence(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    # Reorder: Rule 3 (LEVEL-C) before Rule 2 (LEVEL-B)
    fb = engine.apply_feedback(
        "Apply Rule 3 before Rule 2.",
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester",
    )
    assert fb["status"] == "APPLIED"

    rules = engine.list_rules("LEARNING_TEST_2026_01", user_scope="tester")
    r_c = next(r for r in rules if "LEVEL_C" in r["rule_key"])
    r_b = next(r for r in rules if "LEVEL_B" in r["rule_key"])
    assert r_c["precedence"] > r_b["precedence"]


# -----------------------------------------------------------------------------
# TEST 12: Both /api/chat and /api/chat/stream process learning instructions identically
# -----------------------------------------------------------------------------
def test_12_chat_and_stream_endpoints_identical(test_db_path):
    app = create_app(database_path=test_db_path)
    client = TestClient(app)

    # Test /api/chat
    chat_resp = client.post("/api/chat", json={"query": LEARNING_PROMPT})
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "Learning Engine — Rules Stored & Verified" in chat_data["response"]
    assert "LEARNING_TEST_2026_01" in chat_data["response"]
    assert "LEVEL_A" in chat_data["response"]

    # Test /api/chat/stream
    stream_prompt = "I am teaching you 1 rule under namespace STREAM_TEST: 1. STREAM-A cases have response deadline of 10 hours."
    stream_resp = client.post("/api/chat/stream", json={"query": stream_prompt})
    assert stream_resp.status_code == 200
    stream_text = stream_resp.text
    assert "STREAM_TEST" in stream_text
    assert "STREAM_A" in stream_text


# -----------------------------------------------------------------------------
# TEST 13: Failed storage never returns a success confirmation
# -----------------------------------------------------------------------------
def test_13_failed_storage_never_confirms_success(engine):
    bad_rule = ExtractedRule(
        rule_key="",  # Empty key should fail validation
        rule_text="",
        subject="",
        predicate="",
        value="",
    )
    batch = engine.save_learned_rules_batch("FAIL_NS", [bad_rule], user_scope="tester")
    assert batch["saved_count"] == 0
    assert batch["failed_count"] == 1
    assert len(batch["failed"]) == 1
    assert "empty" in batch["failed"][0]["reason"].lower()


# -----------------------------------------------------------------------------
# TEST 14: User scope isolation (user_alice vs user_bob)
# -----------------------------------------------------------------------------
def test_14_user_scope_isolation(engine):
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="user_alice")

    recall_alice = engine.retrieve_learned_rules("", namespace="LEARNING_TEST_2026_01", user_scope="user_alice")
    recall_bob = engine.retrieve_learned_rules("", namespace="LEARNING_TEST_2026_01", user_scope="user_bob")

    assert recall_alice["total_rules"] == 5
    assert recall_bob["total_rules"] == 0
    assert recall_bob["status"] == "NOT_FOUND"


# -----------------------------------------------------------------------------
# TEST 15: Backend restart persistence (records survive new LearningEngine instance on disk)
# -----------------------------------------------------------------------------
def test_15_persistence_across_engine_restarts(test_db_path):
    eng1 = LearningEngine(test_db_path)
    res = eng1.extract_learning_intent(LEARNING_PROMPT)
    eng1.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="persister")
    del eng1
    gc.collect()

    eng2 = LearningEngine(test_db_path)
    rules = eng2.list_rules("LEARNING_TEST_2026_01", user_scope="persister")
    assert len(rules) == 5
    stats = eng2.get_learning_stats(user_scope="persister")
    assert stats["total_learned_rules"] == 5
    assert stats["total_namespaces"] >= 1


# -----------------------------------------------------------------------------
# TEST 16: Deterministic Precedence Matrix (Phase 2 Acceptance Verification)
# -----------------------------------------------------------------------------
def test_16_deterministic_precedence_matrix(engine):
    # Learn standard rules
    res = engine.extract_learning_intent(LEARNING_PROMPT)
    engine.save_learned_rules_batch(res.namespace, res.extracted_rules, user_scope="tester")

    # Correct LEVEL-A to 12 hours
    engine.apply_feedback(
        "Change LEVEL-A deadline to 12 hours",
        namespace="LEARNING_TEST_2026_01",
        user_scope="tester"
    )

    # 1. LEVEL-A + ACTIVE -> 12 hours
    r1 = engine.evaluate_rule_query("What is the contact deadline for LEVEL-A account?", namespace="LEARNING_TEST_2026_01", user_scope="tester")
    assert r1 is not None
    assert r1["status"] == "EVALUATED"
    assert "12 hours" in r1["value"]
    assert r1["precedence"] == 1

    # 2. LEVEL-A + ON-HOLD -> Suspended / No contact (Precedence 10 override)
    r2 = engine.evaluate_rule_query("What action should be taken for a LEVEL-A account that is ON-HOLD?", namespace="LEARNING_TEST_2026_01", user_scope="tester")
    assert r2 is not None
    assert r2["status"] == "EVALUATED"
    assert r2["decision"] == "DEADLINES_SUSPENDED_NO_CONTACT"
    assert r2["precedence"] == 10

    # 3. LEVEL-B + ON-HOLD -> Suspended / No contact (Precedence 10 override)
    r3 = engine.evaluate_rule_query("Account is LEVEL-B and status is ON-HOLD. What is the response deadline?", namespace="LEARNING_TEST_2026_01", user_scope="tester")
    assert r3 is not None
    assert r3["status"] == "EVALUATED"
    assert r3["decision"] == "DEADLINES_SUSPENDED_NO_CONTACT"
    assert r3["precedence"] == 10

    # 4. LEVEL-D + ACTIVE -> Unknown rule / refusal to guess
    r4 = engine.evaluate_rule_query("What is the deadline for LEVEL-D account?", namespace="LEARNING_TEST_2026_01", user_scope="tester")
    assert r4 is not None
    assert r4["status"] == "UNKNOWN_RULE"
    assert r4["decision"] is None
    assert "not defined" in r4["response"]

    # 5. Unknown status + known level -> Unknown status / refusal to infer
    r5 = engine.evaluate_rule_query("Account is LEVEL-A and status is PENDING_REVIEW. What is the deadline?", namespace="LEARNING_TEST_2026_01", user_scope="tester")
    assert r5 is not None
    assert r5["status"] == "UNKNOWN_STATUS"
    assert r5["decision"] is None
    assert "PENDING_REVIEW" in r5["response"]
