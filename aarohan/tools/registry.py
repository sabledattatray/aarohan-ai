"""AAROHAN AI 2.0 - Unified Tool Registry & Orchestrator.

Enables the AI agent to discover, validate permissions, and dispatch tools
for dataset generation, data analytics, research, document extraction, and code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from aarohan.documents.generator import PDFGenerator
from aarohan.generation.datasets.generator import DatasetGenerator
from aarohan.tools.analytics import AnalyticsEngine
from aarohan.tools.math_verifier import MathVerifier
from aarohan.tools.sandbox import CodeSandbox
from aarohan.tools.sql_executor import SQLExecutor
from aarohan.tools.sql_validator import SQLValidator


class ToolRegistry:
    """Central registry of executable agent tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.analytics = AnalyticsEngine()
        self.sandbox = CodeSandbox()
        self.dataset_generator = DatasetGenerator()
        self.pdf_generator = PDFGenerator()
        self.sql_executor = SQLExecutor()
        self.sql_validator = SQLValidator()
        self.math_verifier = MathVerifier()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register_tool(
            name="generate_dataset",
            description="Generate synthetic business datasets in CSV, XLSX, JSON, or SQL format.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of dataset desired."},
                    "row_count": {"type": "integer", "default": 100},
                    "format": {"type": "string", "enum": ["csv", "xlsx", "json", "sql"], "default": "csv"},
                },
                "required": ["prompt"],
            },
            handler=self._tool_generate_dataset,
        )

        self.register_tool(
            name="execute_python",
            description="Execute Python code in an isolated subprocess sandbox.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code."},
                },
                "required": ["code"],
            },
            handler=self._tool_execute_python,
        )

        self.register_tool(
            name="profile_data",
            description="Compute statistical profile, missing values and distributions for a dataset file.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV or Excel file."},
                },
                "required": ["file_path"],
            },
            handler=self._tool_profile_data,
        )

        self.register_tool(
            name="generate_pdf",
            description="Generate a formatted downloadable PDF document with headings, paragraphs, and tables.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the document."},
                    "sections": {"type": "array", "description": "List of section objects with heading and content."},
                    "author": {"type": "string", "default": "Datta Sable / AAROHAN AI"},
                },
                "required": ["title", "sections"],
            },
            handler=self._tool_generate_pdf,
        )

        self.register_tool(
            name="execute_sql",
            description="Execute read-only SQL query against local SQLite databases.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute."},
                    "max_rows": {"type": "integer", "default": 100},
                },
                "required": ["query"],
            },
            handler=self._tool_execute_sql,
        )

        self.register_tool(
            name="validate_sql",
            description="Validate SQL query for SQL Server, PostgreSQL, MySQL, and SQLite, check for destructive operations, or explain JOIN vs UNION.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to validate."},
                },
                "required": ["query"],
            },
            handler=self._tool_validate_sql,
        )

        self.register_tool(
            name="verify_math",
            description="Perform mathematical calculation with independent multi-method verification and edge case analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate and verify."},
                },
                "required": ["expression"],
            },
            handler=self._tool_verify_math,
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable) -> None:
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in self.tools.values()
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"success": False, "error": f"Tool '{name}' is not registered."}
        try:
            return self.tools[name]["handler"](**arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_generate_dataset(self, prompt: str, row_count: int = 100, format: str = "csv") -> Dict[str, Any]:
        lower = prompt.lower()
        domain = "banking"
        if "sale" in lower or "revenue" in lower:
            domain = "sales"
        elif "ecom" in lower or "product" in lower or "order" in lower:
            domain = "ecommerce"
        elif "employ" in lower or "attend" in lower or "staff" in lower:
            domain = "employee"
        res = self.dataset_generator.generate(domain=domain, rows=row_count, fmt=format)
        return {"success": True, "result": res.to_dict()}

    def _tool_execute_python(self, code: str) -> Dict[str, Any]:
        return self.sandbox.execute_python(code)

    def _tool_profile_data(self, file_path: str) -> Dict[str, Any]:
        return self.analytics.profile_dataframe(file_path)

    def _tool_generate_pdf(self, title: str, sections: List[Dict[str, Any]], author: str = "Datta Sable / AAROHAN AI") -> Dict[str, Any]:
        return self.pdf_generator.generate_report(title=title, sections=sections, author=author)

    def _tool_execute_sql(self, query: str, max_rows: int = 100) -> Dict[str, Any]:
        return self.sql_executor.execute_query(query=query, max_rows=max_rows)

    def _tool_validate_sql(self, query: str) -> Dict[str, Any]:
        report = self.sql_validator.validate_query(query)
        return {
            "is_valid": report.is_valid,
            "dialect": report.dialect,
            "destructive": report.destructive,
            "syntax_errors": report.syntax_errors,
            "warnings": report.warnings,
            "ast_summary": report.ast_summary,
            "status": "VALIDATED" if report.is_valid else "FAILED",
        }

    def _tool_verify_math(self, expression: str) -> Dict[str, Any]:
        res = self.math_verifier.evaluate_and_verify(expression)
        return res.to_execution_result().to_dict()
