"""Tests for Aarohan 2.0 Dataset Generation Studio."""

import json
from pathlib import Path

import openpyxl

from aarohan.generation.datasets.generator import DatasetGenerator, DatasetRequest


def test_banking_dataset_csv_generation(tmp_path: Path):
    generator = DatasetGenerator(output_dir=tmp_path)
    req = DatasetRequest(topic="banking", row_count=50, format="csv", seed=101)
    dataset = generator.generate(req)

    assert dataset.row_count == 50
    assert dataset.format == "csv"
    assert Path(dataset.file_path).is_file()
    assert "Transaction_ID" in dataset.columns
    assert "Customer_Name" in dataset.columns
    assert len(dataset.preview_rows) == 5

    # Verify CSV file contents
    content = Path(dataset.file_path).read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert len(lines) == 51  # Header + 50 rows


def test_excel_xlsx_generation(tmp_path: Path):
    generator = DatasetGenerator(output_dir=tmp_path)
    req = DatasetRequest(topic="sales for power bi", row_count=25, format="xlsx", seed=42)
    dataset = generator.generate(req)

    assert dataset.format == "xlsx"
    assert Path(dataset.file_path).is_file()
    assert "Total_Revenue" in dataset.columns

    # Verify workbook is readable by openpyxl
    wb = openpyxl.load_workbook(dataset.file_path)
    assert len(wb.sheetnames) >= 1
    ws = wb.active
    assert ws.max_row == 26  # Header + 25 rows


def test_json_and_sql_generation(tmp_path: Path):
    generator = DatasetGenerator(output_dir=tmp_path)

    # JSON test
    json_req = DatasetRequest(topic="employee", row_count=15, format="json", seed=99)
    json_ds = generator.generate(json_req)
    assert Path(json_ds.file_path).is_file()
    data = json.loads(Path(json_ds.file_path).read_text(encoding="utf-8"))
    assert len(data) == 15
    assert "Employee_ID" in data[0]

    # SQL test
    sql_req = DatasetRequest(topic="ecommerce", row_count=10, format="sql", seed=12)
    sql_ds = generator.generate(sql_req)
    assert Path(sql_ds.file_path).is_file()
    sql_text = Path(sql_ds.file_path).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS" in sql_text
    assert "INSERT INTO" in sql_text
