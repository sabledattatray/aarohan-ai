"""AAROHAN AI - Automated Integration Tests for Execution & Verification Engine 1.0.

In accordance with Section 7 of AAROHAN AI — EXECUTION & VERIFICATION ENGINE 1.0:
1. Generate exactly 5 CSV records with 4 columns.
2. Verify the CSV can be reopened.
3. Verify numeric data types.
4. Verify date formats.
5. Execute a Python calculation.
6. Validate a SQL Server query.
7. Detect duplicate account numbers.
8. Distinguish JOIN from UNION.
9. Verify mathematical results.
10. Return a real downloadable file.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.generation.datasets.generator import DatasetGenerator
from aarohan.tools.math_verifier import MathVerifier
from aarohan.tools.sandbox import CodeSandbox
from aarohan.tools.sql_validator import SQLDialect, SQLValidator
from aarohan.tools.verification import ExecutionStatus


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_exec.sqlite"
    app = create_app(database_path=db_path)
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Generate exactly 5 CSV records with 4 columns
# -----------------------------------------------------------------------------
def test_1_generate_exact_5_records_4_columns(tmp_path):
    gen = DatasetGenerator(output_dir=tmp_path)
    ds = gen.generate(domain="banking", rows=5, columns=4, fmt="csv", seed=101)

    assert ds.rows == 5
    assert len(ds.columns) == 4
    assert len(ds.preview_rows) == 5
    assert ds.format == "csv"
    assert ds.validation_status == "VALIDATED"


# -----------------------------------------------------------------------------
# 2. Verify the CSV can be reopened
# -----------------------------------------------------------------------------
def test_2_verify_csv_can_be_reopened(tmp_path):
    gen = DatasetGenerator(output_dir=tmp_path)
    ds = gen.generate(domain="banking", rows=5, columns=4, fmt="csv", seed=202)

    csv_path = Path(ds.file_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    # Reopen and inspect from disk
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    assert len(header) == 4
    assert len(data_rows) == 5
    assert ds.validation_details["reopened"] is True
    assert ds.validation_details["row_count_match"] is True
    assert ds.validation_details["column_count_match"] is True


# -----------------------------------------------------------------------------
# 3. Verify numeric data types
# -----------------------------------------------------------------------------
def test_3_verify_numeric_data_types(tmp_path):
    gen = DatasetGenerator(output_dir=tmp_path)
    ds = gen.generate(domain="banking", rows=5, columns=4, fmt="csv", seed=303)

    csv_path = Path(ds.file_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    # Amount_INR is the numeric column in 4-column banking dataset
    for r in rows:
        raw_val = r["Amount_INR"]
        amount = float(raw_val)
        assert amount > 0.0
        assert isinstance(amount, float)

    assert "Amount_INR" in ds.validation_details["detected_numeric_columns"]


# -----------------------------------------------------------------------------
# 4. Verify date formats
# -----------------------------------------------------------------------------
def test_4_verify_date_formats(tmp_path):
    gen = DatasetGenerator(output_dir=tmp_path)
    ds = gen.generate(domain="banking", rows=5, columns=4, fmt="csv", seed=404)

    csv_path = Path(ds.file_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    for r in rows:
        raw_date = r["Date"]
        # Format is standard '%Y-%m-%d %H:%M:%S'
        parsed_dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        assert parsed_dt.year >= 2020

    assert "Date" in ds.validation_details["detected_date_columns"]


# -----------------------------------------------------------------------------
# 5. Execute a Python calculation
# -----------------------------------------------------------------------------
def test_5_execute_python_calculation():
    sandbox = CodeSandbox(timeout_seconds=5)
    code = (
        "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
        "sum_of_squares = sum(x**2 for x in numbers)\n"
        "print(f'RESULT={sum_of_squares}')\n"
    )
    res = sandbox.execute_and_verify(code, expected_output="RESULT=385")

    assert res.execution_status == ExecutionStatus.VALIDATED
    assert res.validation_status == "VALIDATED"
    assert "RESULT=385" in res.output_preview["stdout"]


# -----------------------------------------------------------------------------
# 6. Validate a SQL Server query
# -----------------------------------------------------------------------------
def test_6_validate_sql_server_query():
    validator = SQLValidator()
    sql_server_query = "SELECT TOP 5 [Account_Number], [Balance] FROM [Accounts] ORDER BY [Balance] DESC;"
    report = validator.validate_query(sql_server_query)

    assert report.is_valid is True
    assert report.dialect == SQLDialect.SQL_SERVER
    assert report.is_read_only is True
    assert report.destructive is False
    assert any("TOP" in w for w in report.warnings)


# -----------------------------------------------------------------------------
# 7. Detect duplicate account numbers
# -----------------------------------------------------------------------------
def test_7_detect_duplicate_account_numbers():
    validator = SQLValidator()
    records = [
        {"Account_Number": "AC1001", "Customer_Name": "Alice", "Amount": 5000.0},
        {"Account_Number": "AC1002", "Customer_Name": "Bob", "Amount": 7500.0},
        {"Account_Number": "AC1001", "Customer_Name": "Alice Duplicate", "Amount": 1200.0},
        {"Account_Number": "AC1003", "Customer_Name": "Charlie", "Amount": 3400.0},
        {"Account_Number": "AC1002", "Customer_Name": "Bob Duplicate", "Amount": 900.0},
    ]
    res = validator.detect_duplicate_accounts(records, key_column="Account_Number")

    assert res["has_duplicates"] is True
    assert res["total_records"] == 5
    assert res["unique_count"] == 3
    assert res["duplicate_count"] == 2
    assert "AC1001" in res["duplicates"]
    assert res["duplicates"]["AC1001"] == 2
    assert "AC1002" in res["duplicates"]
    assert res["duplicates"]["AC1002"] == 2


# -----------------------------------------------------------------------------
# 8. Distinguish JOIN from UNION
# -----------------------------------------------------------------------------
def test_8_distinguish_join_from_union():
    validator = SQLValidator()
    distinction = validator.explain_join_vs_union()

    assert distinction["status"] == "VALIDATED"
    # Horizontal vs Vertical
    assert "Horizontal" in distinction["join"]["orientation"]
    assert "Vertical" in distinction["union"]["orientation"]
    # Schema requirements
    assert "Does NOT require identical columns" in distinction["join"]["column_requirement"]
    assert "SAME number of columns" in distinction["union"]["column_requirement"]
    # Duplicate handling in UNION
    assert any("UNION ALL" in t for t in distinction["union"]["types"])


# -----------------------------------------------------------------------------
# 9. Verify mathematical results
# -----------------------------------------------------------------------------
def test_9_verify_mathematical_results():
    verifier = MathVerifier()

    # Calculation 1: Exact compound arithmetic with functions
    expr = "sqrt(144) + 15 * 4 - (8 / 2)"  # 12 + 60 - 4 = 68.0
    res = verifier.evaluate_and_verify(expr)
    assert res.is_verified is True
    assert res.primary_result == 68.0
    assert res.contradictions_detected is False
    assert len(res.assumptions) > 0

    # Calculation 2: Edge case - division by zero
    res_div0 = verifier.evaluate_and_verify("250 / 0")
    assert res_div0.primary_result == "Undefined"
    assert any("Division by zero" in ec for ec in res_div0.edge_cases_checked)

    # Calculation 3: Statistical analysis with method declaration & independent cross-check
    stats = verifier.calculate_statistics([10.0, 20.0, 30.0, 40.0, 50.0], method="sample")
    assert stats["status"] == "VALIDATED"
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert "Bessel's correction" in stats["method_declared"]
    assert stats["independent_cross_check_passed"] is True


# -----------------------------------------------------------------------------
# 10. Return a real downloadable file
# -----------------------------------------------------------------------------
def test_10_return_real_downloadable_file(client, tmp_path):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # Request generation of real 5-row CSV dataset
    chat_payload = {
        "query": "Generate a CSV dataset of banking transactions with exactly 5 records and 4 columns",
        "engine": "foundation",
        "max_new_tokens": 15,
    }
    chat_resp = client.post("/api/chat", json=chat_payload, headers=headers)
    assert chat_resp.status_code == 200
    chat_text = chat_resp.json()["model_response"]["text"]

    assert "/api/datasets/download/" in chat_text
    import re
    match = re.search(r"/api/datasets/download/([^\)\s]+)", chat_text)
    assert match is not None
    filename = match.group(1)

    # Test downloading the file directly through the download endpoint
    dl_resp = client.get(f"/api/datasets/download/{filename}")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"].startswith("text/csv")
    assert f'filename="{filename}"' in dl_resp.headers["content-disposition"]

    # Verify content has exactly 5 data rows + 1 header row = 6 non-empty lines
    lines = [line.strip() for line in dl_resp.text.strip().splitlines() if line.strip()]
    assert len(lines) == 6  # 1 header + 5 records
