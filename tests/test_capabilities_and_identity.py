"""AAROHAN AI - Automated Tests for Identity, Capability Registry, and Honesty Engine.

Verifies:
1. Who are you?
2. Who created you?
3. Which model powers you?
4. What can you do?
5. Can you generate an image?
6. Can you analyze an Excel file?
7. Can you execute SQL?
8. Can you create a downloadable PDF?
9. Real PDF Generation and Downloadable File verification.
10. Safe SQL Execution and remote DB configuration reporting.
11. REST API endpoints (/api/capabilities, /api/documents/generate-pdf, /api/sql/execute).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.capabilities import (
    CapabilityStatus,
    get_capability_registry,
    resolve_capability_query,
)
from aarohan.documents.generator import PDFGenerator
from aarohan.nlp.identity import resolve_identity_query
from aarohan.tools.sql_executor import SQLExecutor


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_capabilities.sqlite"
    app = create_app(database_path=db_path)
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Identity Verification Tests
# -----------------------------------------------------------------------------
def test_identity_who_are_you():
    for q in [
        "Who are you?", "who are you", "what is your name", "what's your name",
        "whats your name", "whats your nam", "whats ur name", "your name",
    ]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "My name is AAROHAN" in reply
        assert "Datta Sable" in reply


def test_identity_what_is_my_name():
    for q in ["What is my name?", "what is my name", "who am i", "tell me my name", "whats my name"]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "Your name is Datta Sable" in reply
        assert "AAROHAN AI" in reply


def test_identity_user_background():
    for q in [
        "What is my professional background?",
        "tell me about my professional background",
        "who is datta sable",
        "what is datta sable background",
    ]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "independent AI engineer" in reply and "Datta Sable" in reply
        assert "Anthropic" not in reply or "not affiliated with Anthropic" in reply


def test_identity_website():
    for q in [
        "What is my website?",
        "whats my website",
        "what is your website",
        "whats your website",
        "my website",
        "what is the official website",
        "what is the github repo",
    ]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "https://dattasable.com/" in reply
        assert "https://github.com/sabledattatray/aarohan-ai" in reply


def test_identity_who_created_you():
    for q in [
        "Who created you?",
        "who created you",
        "who is your creator",
        "who made you",
        "who developed you",
        "who built you",
        "tell me who created you",
    ]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "I was created and developed by Datta Sable." == reply


def test_identity_which_model_powers_you():
    # Foundation engine
    for q in [
        "Which model powers you?",
        "1. Which AI model powers you right now?",
        "which ai model powers you right now",
        "what model powers you",
    ]:
        reply_foundation = resolve_identity_query(q, engine="foundation")
        assert reply_foundation is not None
        assert "Qwen2.5-0.5B-Instruct" in reply_foundation
        assert "Alibaba Cloud" in reply_foundation
        assert "running locally" in reply_foundation

    # Scratch checkpoint engine
    reply_scratch = resolve_identity_query("what model powers you", engine="scratch")
    assert reply_scratch is not None
    assert "custom pre-trained" in reply_scratch
    assert "Datta Sable" in reply_scratch


def test_identity_third_party_separation():
    for q in [
        "Are you created by Anthropic or OpenAI?",
        "Are you made by OpenAI?",
        "Are you ChatGPT?",
        "Are you Claude?",
    ]:
        reply = resolve_identity_query(q)
        assert reply is not None
        assert "No. AAROHAN is a separate AI application developed by Datta Sable." in reply


# -----------------------------------------------------------------------------
# 2. Capability Awareness & Truthful Registry Tests
# -----------------------------------------------------------------------------
def test_capability_registry_completeness():
    reg = get_capability_registry()
    caps = reg.list_all()
    assert len(caps) >= 13

    expected_ids = {
        "ai_text_generation",
        "reasoning_qa",
        "sql_generation_and_explanation",
        "python_code_generation",
        "excel_csv_analysis",
        "dataset_processing",
        "chart_generation",
        "pdf_document_generation",
        "image_generation",
        "image_editing",
        "web_research",
        "knowledge_graph",
        "cognitive_memory",
    }
    registered_ids = {c.id for c in caps}
    for expected in expected_ids:
        assert expected in registered_ids

    # Web research must be honest about requiring configuration
    web_cap = reg.get("web_research")
    assert web_cap.status == CapabilityStatus.CONFIGURATION_REQUIRED
    assert web_cap.requirements is not None


def test_capability_what_can_you_do():
    for q in [
        "What can you do?",
        "what can you do",
        "what are your capabilities",
        "tell me your capabilities",
        "list your features",
    ]:
        reply = resolve_capability_query(q)
        assert reply is not None
        assert "AAROHAN AI — Capabilities & Operational Status" in reply
        assert "Available & Fully Operational Capabilities" in reply
        assert "Capabilities Requiring Configuration" in reply
        assert "Web research" in reply
        assert "Honesty Policy" in reply


def test_capability_can_you_generate_an_image():
    for q in [
        "Can you generate an image?",
        "can you generate an image",
        "can you make images",
        "can you create images",
    ]:
        reply = resolve_capability_query(q)
        assert reply is not None
        assert "Yes, AAROHAN can generate images." in reply
        assert "Image Generation Engine 1.0" in reply
        assert "/app/images" in reply


def test_capability_can_you_analyze_an_excel_file():
    for q in [
        "Can you analyze an Excel file?",
        "can you analyze an excel file",
        "can you analyze csv files",
        "can you read excel",
    ]:
        reply = resolve_capability_query(q)
        assert reply is not None
        assert "Yes, AAROHAN can analyze Excel and CSV files." in reply
        assert "Analytics Engine" in reply
        assert "pandas" in reply


def test_capability_can_you_execute_sql():
    for q in [
        "Can you execute SQL?",
        "can you execute sql",
        "can you run sql queries",
        "can you execute database queries",
    ]:
        reply = resolve_capability_query(q)
        assert reply is not None
        assert "Yes, AAROHAN can execute SQL queries with specific operational boundaries" in reply
        assert "What works now" in reply
        assert "Local SQLite Execution" in reply
        assert "Configuration required (Remote Databases)" in reply


def test_capability_can_you_create_a_downloadable_pdf():
    for q in [
        "Can you create a downloadable PDF?",
        "can you create a downloadable pdf",
        "can you generate a pdf",
        "can you make pdf files",
    ]:
        reply = resolve_capability_query(q)
        assert reply is not None
        assert "Yes, AAROHAN can create downloadable PDF documents." in reply
        assert "PDF Generation Engine" in reply
        assert "reportlab" in reply
        assert "/api/documents/download/" in reply


# -----------------------------------------------------------------------------
# 3. Real Tool & Execution Tests (Zero Simulation)
# -----------------------------------------------------------------------------
def test_real_pdf_generation(tmp_path):
    generator = PDFGenerator(output_dir=tmp_path)
    res = generator.generate_report(
        title="AAROHAN Engineering Audit",
        sections=[
            {"heading": "Executive Summary", "content": "AAROHAN is an independent AI created by Datta Sable."},
            {
                "heading": "Verified Modules",
                "content": "All core modules are verified.",
                "table": [
                    ["Module", "Status"],
                    ["Identity", "Verified"],
                    ["Capabilities", "Operational"],
                    ["Image Studio", "Functional"],
                ],
            },
        ],
        author="Datta Sable",
        filename="test_audit.pdf",
    )
    assert res["success"] is True
    pdf_file = Path(res["file_path"])
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 500  # Real PDF with binary bytes
    assert res["download_url"].startswith("/api/documents/download/")


def test_safe_sql_execution(tmp_path):
    executor = SQLExecutor(default_db=tmp_path / "test.db")
    # Test valid query
    res = executor.execute_query("SELECT 1 AS test_val, 'Aarohan' AS app_name;")
    assert res["success"] is True
    assert res["columns"] == ["test_val", "app_name"]
    assert res["rows"] == [(1, "Aarohan")]
    assert res["row_count"] == 1

    # Test dangerous mutation query blocked in read-only mode
    res_unsafe = executor.execute_query("DROP TABLE users;")
    assert res_unsafe["success"] is False
    assert res_unsafe["status"] == "FORBIDDEN"

    # Test remote database connection reporting configuration required
    res_remote = executor.execute_query("CONNECT TO SERVER='prod.db.corp.net' USER='admin'")
    assert res_remote["success"] is False
    assert res_remote["status"] == "CONFIGURATION_REQUIRED"


# -----------------------------------------------------------------------------
# 4. HTTP API Endpoints Integration Tests
# -----------------------------------------------------------------------------
def test_api_capabilities_endpoint(client):
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert len(data["capabilities"]) >= 13
    assert "summary_markdown" in data


def test_api_chat_identity_queries(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # 1. Who are you?
    r1 = client.post("/api/chat", json={"query": "Who are you?", "engine": "foundation"}, headers=headers)
    assert r1.status_code == 200
    assert "My name is AAROHAN" in r1.json()["model_response"]["text"]
    assert "Datta Sable" in r1.json()["model_response"]["text"]

    # 2. Who created you?
    r2 = client.post("/api/chat", json={"query": "Who created you?", "engine": "foundation"}, headers=headers)
    assert r2.status_code == 200
    assert "I was created and developed by Datta Sable." == r2.json()["model_response"]["text"]

    # 3. Which model powers you?
    r3 = client.post("/api/chat", json={"query": "Which model powers you?", "engine": "foundation"}, headers=headers)
    assert r3.status_code == 200
    assert "Qwen2.5-0.5B-Instruct" in r3.json()["model_response"]["text"]


def test_api_chat_capability_queries(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # 4. What can you do?
    r4 = client.post("/api/chat", json={"query": "What can you do?", "engine": "foundation"}, headers=headers)
    assert r4.status_code == 200
    assert "AAROHAN AI — Capabilities & Operational Status" in r4.json()["model_response"]["text"]

    # 5. Can you generate an image?
    r5 = client.post("/api/chat", json={"query": "Can you generate an image?", "engine": "foundation"}, headers=headers)
    assert r5.status_code == 200
    assert "Yes, AAROHAN can generate images." in r5.json()["model_response"]["text"]

    # 6. Can you analyze an Excel file?
    r6 = client.post("/api/chat", json={"query": "Can you analyze an Excel file?", "engine": "foundation"}, headers=headers)
    assert r6.status_code == 200
    assert "Yes, AAROHAN can analyze Excel and CSV files." in r6.json()["model_response"]["text"]

    # 7. Can you execute SQL?
    r7 = client.post("/api/chat", json={"query": "Can you execute SQL?", "engine": "foundation"}, headers=headers)
    assert r7.status_code == 200
    assert "Yes, AAROHAN can execute SQL queries with specific operational boundaries" in r7.json()["model_response"]["text"]

    # 8. Can you create a downloadable PDF?
    r8 = client.post("/api/chat", json={"query": "Can you create a downloadable PDF?", "engine": "foundation"}, headers=headers)
    assert r8.status_code == 200
    assert "Yes, AAROHAN can create downloadable PDF documents." in r8.json()["model_response"]["text"]


def test_api_pdf_and_sql_endpoints(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # Generate PDF endpoint
    pdf_payload = {
        "title": "Quarterly Performance",
        "sections": [
            {"heading": "Highlights", "content": "All systems operational."},
        ],
        "author": "Datta Sable",
    }
    r_pdf = client.post("/api/documents/generate-pdf", json=pdf_payload, headers=headers)
    assert r_pdf.status_code == 200
    pdf_data = r_pdf.json()
    assert pdf_data["success"] is True
    filename = pdf_data["filename"]

    # Download PDF endpoint
    r_dl = client.get(f"/api/documents/download/{filename}")
    assert r_dl.status_code == 200
    assert r_dl.headers["content-type"] == "application/pdf"
    assert len(r_dl.content) > 500

    # Execute SQL endpoint
    sql_payload = {"query": "SELECT 42 AS answer, 'Truthful' AS status"}
    r_sql = client.post("/api/sql/execute", json=sql_payload, headers=headers)
    assert r_sql.status_code == 200
    sql_data = r_sql.json()
    assert sql_data["success"] is True
    assert sql_data["rows"] == [[42, "Truthful"]]
