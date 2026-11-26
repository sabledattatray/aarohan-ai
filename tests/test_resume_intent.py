"""Test that resume pastes and document analysis requests do not trigger dataset generation."""

from starlette.testclient import TestClient

from aarohan.api import create_app
from aarohan.capabilities import resolve_capability_query
from aarohan.nlp.intent import Intent, IntentClassifier


def test_resume_paste_does_not_trigger_dataset_generation():
    classifier = IntentClassifier()
    resume_sample = (
        "DATTATRAY SABLE SENIOR MIS & BUSINESS INTELLIGENCE ANALYST "
        "Power BI | SQL Server | Advanced Excel | BFSI Analytics | Reporting Automation "
        "Experience: Managed daily, weekly and monthly MIS reporting using Excel, SQL Server, Power BI. "
        "Built a 231,130-row synthetic retail dataset for dashboard reference. "
        "Converts Excel or CSV data into multi-sheet xlsx dashboards."
    )
    intent_res = classifier.classify(resume_sample)
    # The intent should not be DATASET_GENERATION or EXCEL_GENERATION because there is no action verb
    assert intent_res.primary_intent not in (Intent.DATASET_GENERATION, Intent.EXCEL_GENERATION)


def test_pdf_resume_analysis_capability_query():
    ans = resolve_capability_query("can you analyze my pdf resume")
    assert ans is not None
    assert "Yes, AAROHAN can analyze PDF, DOCX, and text resumes." in ans
    assert "Document Intelligence Engine" in ans


def test_resume_in_chat_stream_does_not_generate_dataset():
    app = create_app()
    client = TestClient(app)

    # Login to get auth token
    login_resp = client.post("/api/auth/login", json={"username": "admin", "password": "aarohan_secret_2026"})
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    resume_text = (
        "DATTATRAY SABLE SENIOR MIS & BUSINESS INTELLIGENCE ANALYST\n"
        "Power BI | SQL Server | Advanced Excel | BFSI Analytics\n"
        "Experience: Managed daily MIS reporting using Excel and SQL Server."
    )
    resp = client.post(
        "/api/chat/stream",
        json={"query": resume_text, "engine": "scratch"},
        headers=headers,
    )
    assert resp.status_code == 200
    stream_content = resp.text
    # Should NOT have auto-generated dataset
    assert "Aarohan Dataset Generated & Validated" not in stream_content
