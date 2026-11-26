"""Tests for Aarohan 2.0 Scientific Intelligence and Mathematical Formulation."""

import pytest
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.nlp.intent import Intent, IntentClassifier
from aarohan.nlp.normalizer import TextNormalizer
from aarohan.nlp.science import resolve_scientific_law


@pytest.fixture
def normalizer():
    return TextNormalizer()


@pytest.fixture
def intent_classifier():
    return IntentClassifier()


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_science.sqlite"
    app = create_app(database_path=db_path)
    return TestClient(app)


def test_normalizer_thermodynamics_typo(normalizer):
    q = "tell me the formula of third low of thermodynamics"
    norm = normalizer.normalize(q)
    assert "third law of thermodynamics" in norm.normalized


def test_intent_scientific_law(intent_classifier):
    q = "tell me the formula of third law of thermodynamics"
    res = intent_classifier.classify(q)
    assert res.primary_intent in (Intent.SCIENTIFIC_LAW, Intent.MATHEMATICAL_CALCULATION)


def test_resolve_scientific_law_third_law():
    queries = [
        "tell me the formula of third low of thermodynamics",
        "third law of thermodynamics",
        "formula of third law of thermodynamics",
        "3rd law of thermodynamics",
    ]
    for q in queries:
        law = resolve_scientific_law(q)
        assert law is not None
        assert law.law_id == "third_law_thermodynamics"
        assert r"\lim_{T \to 0} S = 0" in law.formula_latex or r"\lim_{T \to 0} S = 0" in law.detailed_markdown
        assert "Boltzmann" in law.detailed_markdown
        # Guarantee no hallucination
        assert "T_3 = " not in law.detailed_markdown


def test_resolve_other_laws():
    assert resolve_scientific_law("first law of thermodynamics") is not None
    assert resolve_scientific_law("second law of thermodynamics") is not None
    assert resolve_scientific_law("zeroth law of thermodynamics") is not None
    assert resolve_scientific_law("newton's second law") is not None
    assert resolve_scientific_law("ohm's law formula") is not None


def test_api_chat_third_law(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}
    payload = {
        "query": "tell me the formula of third low of thermodynamics",
        "engine": "foundation",
    }
    response = client.post("/api/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    text = data["model_response"]["text"]
    assert "Third Law of Thermodynamics" in text
    assert r"\lim_{T \to 0} S = 0" in text or "k_{\\text{B}} \\ln \\Omega" in text or "Boltzmann" in text
    assert "T_3 = " not in text
    assert data["intent"] in ("SCIENTIFIC_LAW", "MATHEMATICAL_CALCULATION")


def test_api_chat_stream_third_law(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}
    payload = {
        "query": "tell me the formula of third low of thermodynamics",
        "engine": "foundation",
    }
    response = client.post("/api/chat/stream", json=payload, headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "Third Law of Thermodynamics" in content
    assert "T_3 = " not in content
