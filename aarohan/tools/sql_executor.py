"""AAROHAN AI - Safe SQL Execution Engine.

Executes read-only SQL queries against local SQLite databases and temporary in-memory
tabular tables. Enforces safety boundaries and reports configuration requirements for
remote enterprise databases.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aarohan.tools.sql_executor")

DEFAULT_DB_PATH = Path("data/memory/aarohan.sqlite")


class SQLExecutor:
    """Safe, bounded SQL execution engine for local datasets and SQLite databases."""

    def __init__(self, default_db: Optional[Path] = None) -> None:
        self.default_db = default_db or DEFAULT_DB_PATH

    def execute_query(
        self,
        query: str,
        db_path: Optional[str] = None,
        max_rows: int = 100,
        read_only: bool = True,
    ) -> Dict[str, Any]:
        """Execute a SQL query with bounds and safety checks."""
        q_strip = query.strip()
        q_upper = q_strip.upper()

        # Check for remote database patterns or connection intents
        if any(keyword in q_upper for keyword in ["CONNECT TO", "USE DATABASE", "HOST=", "SERVER="]):
            return {
                "success": False,
                "status": "CONFIGURATION_REQUIRED",
                "error": "Live connection to remote database servers requires configuring connection credentials.",
                "explanation": (
                    "AAROHAN can generate and optimize queries for PostgreSQL, MySQL, Snowflake, and Oracle. "
                    "However, direct network execution against an external server requires providing the host, "
                    "port, username, and password."
                ),
            }

        # Safety: enforce read-only if requested
        forbidden_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "ATTACH", "DETACH"]
        if read_only:
            forbidden_keywords.extend(["INSERT", "UPDATE", "CREATE", "REPLACE"])

        for word in forbidden_keywords:
            if re_word_match(word, q_upper):
                logger.warning("Blocked unsafe SQL operation '%s' in read-only mode", word)
                return {
                    "success": False,
                    "status": "FORBIDDEN",
                    "error": f"Operation '{word}' is restricted in safe read-only SQL execution.",
                }

        target_path = Path(db_path) if db_path else self.default_db
        if not target_path.exists():
            # If target database doesn't exist, create an in-memory database
            target_path = ":memory:"
        else:
            target_path = str(target_path)

        start_time = time.perf_counter()
        try:
            conn = sqlite3.connect(target_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute(q_strip)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            total_fetched = len(rows)

            conn.close()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "success": True,
                "status": "AVAILABLE",
                "query": q_strip,
                "columns": columns,
                "rows": rows,
                "row_count": total_fetched,
                "execution_time_ms": elapsed_ms,
                "database": str(target_path),
            }
        except Exception as e:
            logger.error("SQL execution failed: %s | Query: %s", e, q_strip)
            return {
                "success": False,
                "status": "ERROR",
                "error": str(e),
                "query": q_strip,
            }


def re_word_match(word: str, text: str) -> bool:
    import re
    return bool(re.search(rf"\b{re.escape(word)}\b", text))
