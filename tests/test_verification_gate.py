"""Unit tests for AAROHAN AI Output Verification & Self-Correction Gate (Phase 6)."""


import pytest

from aarohan.learning import LearningEngine
from aarohan.tools.math_verifier import MathVerifier
from aarohan.verification_gate import OutputVerificationGate


@pytest.fixture
def gate(tmp_path):
    db_path = tmp_path / "test_gate.sqlite"
    eng = LearningEngine(db_path)
    # Seed LEVEL-A 12 hours active rule
    with eng._connect() as conn:
        conn.execute("""
            INSERT INTO learned_rules (
                id, namespace, rule_key, rule_text, subject, predicate, value,
                conditions, precedence, status, version, user_scope, created_at, updated_at
            ) VALUES (
                'rule_e78d030b7beb', 'LEARNING_TEST_2026_01', 'RULE_LEVEL_A',
                'LEVEL-A accounts have a contact deadline of 12 hours.',
                'LEVEL-A', 'has_contact_deadline', '12 hours',
                'status == ACTIVE', 1, 'ACTIVE', 2, 'tester', '2026-01-01', '2026-01-01'
            )
        """)
        conn.commit()

    return OutputVerificationGate(
        learning_engine=eng,
        math_verifier=MathVerifier(),
    )


def test_01_canonical_identity_preserved(gate):
    text = "I am AAROHAN, developed by Datta Sable."
    v = gate.guard_response(text, response_source="aarohan_identity_resolver", model_invoked=False)
    assert v.status == "VERIFIED"
    assert v.is_truthful is True
    assert v.contradictions_detected == 0
    assert "Datta Sable" in v.guarded_text


def test_02_intercept_prohibited_creator(gate):
    text = "I am a helpful assistant developed by Anthropic."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True)
    assert v.status == "CORRECTED"
    assert v.contradictions_detected >= 1
    assert "Anthropic" not in v.guarded_text
    assert "developed by Datta Sable" in v.guarded_text


def test_03_intercept_fake_website(gate):
    text = "You can view my creator's projects at https://aarohan.ai"
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True)
    assert v.status == "CORRECTED"
    assert "https://dattasable.com/" in v.guarded_text
    assert "aarohan.ai" not in v.guarded_text


def test_04_intercept_role_inversion(gate):
    text = "Hello, my name is Datta Sable and I am an AI assistant."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True)
    assert v.status == "CORRECTED"
    assert "I am AAROHAN, and you are Datta Sable" in v.guarded_text


def test_05_arithmetic_claim_verification_success(gate):
    text = "The answer is straightforward: 12 + 8 = 20."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True)
    assert v.contradictions_detected == 0
    assert "12 + 8 = 20" in v.guarded_text


def test_06_arithmetic_hallucination_corrected(gate):
    text = "Based on our computation, 25 * 4 = 125."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True)
    assert v.status == "CORRECTED"
    assert v.contradictions_detected >= 1
    assert "25 * 4 = 100 *(verified)*" in v.guarded_text


def test_07_stale_rule_level_a_corrected(gate):
    text = "According to policy, LEVEL-A has a contact deadline of 24 hours."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True, user_scope="tester")
    assert v.status == "CORRECTED"
    assert "12 hours" in v.guarded_text
    assert "24 hours" not in v.guarded_text


def test_08_invented_rule_level_d_refused(gate):
    text = "For LEVEL-D accounts, the contact deadline is 48 hours."
    v = gate.guard_response(text, response_source="foundation_engine", model_invoked=True, user_scope="tester")
    assert v.status == "CORRECTED"
    assert "Unknown Business Rule — `LEVEL-D`" in v.guarded_text
    assert "will not infer or guess" in v.guarded_text


def test_09_grounding_score_deterministic(gate):
    v = gate.guard_response("Confirmed rule", response_source="aarohan_learning_engine", model_invoked=False)
    assert v.grounding_score == 1.0
    assert v.status == "VERIFIED"


def test_10_grounding_score_with_retrieved_memory(gate):
    memory = [{"content": "Datta Sable is an MIS and reporting automation professional."}]
    v = gate.guard_response(
        "Datta Sable specializes in MIS reporting automation.",
        response_source="foundation_engine",
        model_invoked=True,
        retrieved_memory=memory,
    )
    assert v.grounding_score >= 0.75
    assert v.status in ("GROUNDED", "VERIFIED")
