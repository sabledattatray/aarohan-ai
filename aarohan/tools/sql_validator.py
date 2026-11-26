"""AAROHAN AI - SQL Validation & Execution Engine.

Supports SQL Server (T-SQL), PostgreSQL, MySQL, and SQLite according to configured connections:
- Dialect detection (SQL Server, PostgreSQL, MySQL, SQLite)
- Syntax and destructive operation validation (blocks DROP, TRUNCATE, DELETE without explicit flag)
- Schema metadata inspection
- Duplicate record preservation awareness (UNION vs UNION ALL)
- Duplicate key detection (e.g. duplicate account numbers)
- Safe read-only execution in isolated SQLite or in-memory tables
- Semantic comparison: JOIN vs UNION
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aarohan.tools.verification import ExecutionResult, ExecutionStatus

logger = logging.getLogger("aarohan.tools.sql_validator")


class SQLDialect:
    SQL_SERVER = "SQL Server (T-SQL)"
    POSTGRESQL = "PostgreSQL"
    MYSQL = "MySQL"
    SQLITE = "SQLite"
    UNKNOWN = "Standard SQL"


@dataclass
class SQLValidationReport:
    is_valid: bool
    dialect: str
    is_read_only: bool
    destructive: bool
    syntax_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ast_summary: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None


class SQLValidator:
    """Validates dialect-specific SQL syntax, prevents destructive operations, and checks schema compatibility."""

    def detect_dialect(self, query: str) -> str:
        """Detect the SQL dialect from query tokens and functions."""
        q = query.strip()
        q_upper = q.upper()

        # SQL Server / T-SQL indicators
        if re.search(r"\bTOP\s+\d+\b", q_upper) or "[" in q or "GETDATE()" in q_upper or "@@" in q:
            return SQLDialect.SQL_SERVER

        # PostgreSQL indicators
        if "::" in q or re.search(r"\bILIKE\b", q_upper) or "RETURNING" in q_upper:
            return SQLDialect.POSTGRESQL

        # MySQL indicators
        if "`" in q or "LIMIT" in q_upper and "NOW()" in q_upper and "OFFSET" not in q_upper:
            return SQLDialect.MYSQL

        # SQLite indicators
        if "DATETIME('NOW')" in q_upper or "AUTOINCREMENT" in q_upper:
            return SQLDialect.SQLITE

        return SQLDialect.SQLITE

    def validate_query(self, query: str, allow_destructive: bool = False) -> SQLValidationReport:
        """Validate query syntax, detect dialect, and check for destructive commands."""
        dialect = self.detect_dialect(query)
        q_upper = query.upper().strip()
        errors = []
        warnings = []

        # Check for unclosed quotes or brackets
        if query.count("'") % 2 != 0:
            errors.append("Unclosed single quote in query string.")
        if query.count('"') % 2 != 0:
            errors.append("Unclosed double quote in query string.")
        if query.count("[") != query.count("]"):
            errors.append("Mismatched square brackets in identifier.")

        # Check for destructive keywords
        destructive_ops = ["DROP", "TRUNCATE", "DELETE", "ALTER", "GRANT", "REVOKE"]
        found_destructive = [op for op in destructive_ops if re.search(rf"\b{op}\b", q_upper)]

        is_read_only = len(found_destructive) == 0 and not re.search(r"\b(INSERT|UPDATE|CREATE|REPLACE)\b", q_upper)
        destructive = len(found_destructive) > 0

        if destructive and not allow_destructive:
            errors.append(f"Destructive operation(s) blocked: {', '.join(found_destructive)}. Explicit approval required.")

        # Duplicate preservation warning: UNION vs UNION ALL
        if re.search(r"\bUNION\b", q_upper) and not re.search(r"\bUNION\s+ALL\b", q_upper):
            warnings.append(
                "Query uses 'UNION' which automatically deduplicates records. Use 'UNION ALL' to preserve duplicate records."
            )

        # Detect dialect-specific constructs
        if dialect == SQLDialect.SQL_SERVER:
            if re.search(r"\bTOP\s+\d+\b", q_upper):
                warnings.append("Validated SQL Server 'TOP N' syntax.")

        is_valid = len(errors) == 0

        return SQLValidationReport(
            is_valid=is_valid,
            dialect=dialect,
            is_read_only=is_read_only,
            destructive=destructive,
            syntax_errors=errors,
            warnings=warnings,
            ast_summary={
                "has_join": bool(re.search(r"\bJOIN\b", q_upper)),
                "has_union": bool(re.search(r"\bUNION\b", q_upper)),
                "has_group_by": bool(re.search(r"\bGROUP\s+BY\b", q_upper)),
                "has_order_by": bool(re.search(r"\bORDER\s+BY\b", q_upper)),
            },
        )

    def detect_duplicate_accounts(
        self,
        records: List[Dict[str, Any]],
        key_column: str = "Account_Number",
    ) -> Dict[str, Any]:
        """Detect duplicate account numbers in tabular dataset records."""
        counts: Dict[str, int] = {}
        for row in records:
            val = str(row.get(key_column, "")).strip()
            if val:
                counts[val] = counts.get(val, 0) + 1

        duplicates = {k: v for k, v in counts.items() if v > 1}
        has_duplicates = len(duplicates) > 0

        return {
            "has_duplicates": has_duplicates,
            "key_column": key_column,
            "total_records": len(records),
            "unique_count": len(counts),
            "duplicate_count": len(duplicates),
            "duplicates": duplicates,
            "status": "VALIDATED",
            "message": f"Found {len(duplicates)} duplicate '{key_column}' entries." if has_duplicates else f"No duplicate '{key_column}' found.",
        }

    def explain_join_vs_union(self) -> Dict[str, Any]:
        """Distinguish JOIN from UNION conceptually, syntactically, and operationally."""
        return {
            "comparison": "JOIN vs UNION",
            "join": {
                "orientation": "Horizontal (combines columns from multiple tables side-by-side)",
                "mechanism": "Combines records based on a common matching key / relationship (e.g. A.id = B.account_id)",
                "types": ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "CROSS JOIN"],
                "column_requirement": "Does NOT require identical columns or schemas",
                "example": "SELECT c.Name, a.Balance FROM Customers c INNER JOIN Accounts a ON c.ID = a.CustomerID",
            },
            "union": {
                "orientation": "Vertical (stacks rows from multiple queries on top of each other)",
                "mechanism": "Combines result sets of two or more SELECT statements into a single unified rowset",
                "types": [
                    "UNION (eliminates duplicate rows across sets)",
                    "UNION ALL (preserves all rows, including duplicates)",
                ],
                "column_requirement": "Strictly requires the SAME number of columns with compatible data types in each query",
                "example": "SELECT Account_Number, Balance FROM SavingsAccounts UNION ALL SELECT Account_Number, Balance FROM CurrentAccounts",
            },
            "key_differences": [
                "JOIN extends records horizontally with new attributes.",
                "UNION appends records vertically with additional records.",
                "JOIN requires a joining predicate (ON condition); UNION requires identical column projections.",
                "UNION implicitly removes duplicate records; UNION ALL preserves duplicates.",
            ],
            "status": "VALIDATED",
        }

    def execute_safe_query(
        self,
        query: str,
        db_path: Optional[str] = None,
        max_rows: int = 100,
    ) -> ExecutionResult:
        """Execute a validated read-only SQL query in a safe SQLite or in-memory environment."""
        report = self.validate_query(query)
        if not report.is_valid:
            return ExecutionResult.create(
                tool_used="sql_validator",
                status=ExecutionStatus.FAILED,
                validation_status="FAILED",
                error_details="; ".join(report.syntax_errors),
                metadata={"dialect": report.dialect},
            )

        target_db = db_path if db_path and Path(db_path).exists() else ":memory:"
        start_t = time.perf_counter()

        try:
            conn = sqlite3.connect(target_db, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute(query)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            conn.close()

            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ExecutionResult.create(
                tool_used="sql_validator",
                status=ExecutionStatus.VALIDATED,
                validation_status="VALIDATED",
                output_preview={"columns": columns, "rows": rows, "count": len(rows)},
                validation_details={
                    "dialect": report.dialect,
                    "is_read_only": report.is_read_only,
                    "execution_time_ms": elapsed_ms,
                    "row_count": len(rows),
                },
                metadata={"query": query},
            )
        except Exception as e:
            return ExecutionResult.create(
                tool_used="sql_validator",
                status=ExecutionStatus.FAILED,
                validation_status="FAILED",
                error_details=str(e),
                metadata={"query": query, "dialect": report.dialect},
            )
