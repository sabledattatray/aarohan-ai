"""AAROHAN AI - Task Understanding & Tool Routing Layer.

Classifies incoming user prompts into 11 distinct operational task categories
and routes them to implemented system tools.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TaskType(str, enum.Enum):
    TEXT_GENERATION = "text_generation"
    MATHEMATICAL_CALCULATION = "mathematical_calculation"
    PYTHON_EXECUTION = "python_execution"
    SQL_QUERY = "sql_query"
    DATASET_CREATION = "dataset_creation"
    DATA_ANALYSIS = "data_analysis"
    CHART_GENERATION = "chart_generation"
    DOCUMENT_GENERATION = "document_generation"
    IMAGE_GENERATION = "image_generation"
    WEB_RESEARCH = "web_research"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"


@dataclass
class TaskRouting:
    task_type: TaskType
    tool_name: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class TaskRouter:
    """Intelligent task classifier and tool dispatcher."""

    def route(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> TaskRouting:
        q = query.lower().strip()

        # 1. Dataset Creation (CSV/Excel datasets)
        if any(w in q for w in ["dataset", "csv", "excel file", "xlsx", "sample data", "generate data", "create data"]) and any(
            w in q for w in ["generate", "create", "make", "give me", "provide", "download", "records", "rows"]
        ):
            # Extract row count if mentioned (e.g. 5 records, 100 rows, 1k records)
            row_match = re.search(r"(\d+)\s*(records?|rows?|entries|items)", q)
            col_match = re.search(r"(\d+)\s*columns?", q)
            rows = int(row_match.group(1)) if row_match else 100
            cols = int(col_match.group(1)) if col_match else None

            # Detect format
            fmt = "xlsx" if "excel" in q or "xlsx" in q else "csv"

            # Domain detection
            domain = "banking"
            if "sale" in q or "revenue" in q:
                domain = "sales"
            elif "ecom" in q or "order" in q or "product" in q:
                domain = "ecommerce"
            elif "emp" in q or "staff" in q or "attend" in q:
                domain = "employee"

            return TaskRouting(
                task_type=TaskType.DATASET_CREATION,
                tool_name="dataset_generator",
                confidence=0.95,
                parameters={"rows": rows, "columns": cols, "format": fmt, "domain": domain},
                reasoning=f"User requested {rows} rows {fmt.upper()} dataset in domain '{domain}'.",
            )

        # 2. Image Generation
        if any(w in q for w in ["generate image", "create image", "draw an image", "picture of", "photo of", "image of"]) or (
            "image" in q and any(w in q for w in ["generate", "create", "draw", "make", "render"])
        ):
            return TaskRouting(
                task_type=TaskType.IMAGE_GENERATION,
                tool_name="image_manager",
                confidence=0.95,
                parameters={"prompt": query},
                reasoning="User requested visual image generation.",
            )

        # 3. Document Generation (PDF / DOCX)
        if any(w in q for w in ["pdf", "report", "document", "docx"]) and any(
            w in q for w in ["generate", "create", "make", "download", "export", "compile"]
        ):
            return TaskRouting(
                task_type=TaskType.DOCUMENT_GENERATION,
                tool_name="pdf_generator",
                confidence=0.92,
                parameters={"title": query},
                reasoning="User requested structured document / PDF compilation.",
            )

        # 4. Chart Generation
        if any(w in q for w in ["chart", "plot", "graph", "histogram", "scatterplot", "bar chart", "line plot"]):
            return TaskRouting(
                task_type=TaskType.CHART_GENERATION,
                tool_name="analytics_engine.generate_chart",
                confidence=0.90,
                parameters={"prompt": query},
                reasoning="User requested graphical visualization / chart.",
            )

        # 5. Data Analysis
        if any(w in q for w in ["analyze", "profile", "summary statistics", "correlation", "missing values"]) and any(
            w in q for w in ["data", "dataset", "table", "dataframe", "csv", "excel"]
        ):
            return TaskRouting(
                task_type=TaskType.DATA_ANALYSIS,
                tool_name="analytics_engine.profile_dataframe",
                confidence=0.90,
                parameters={"prompt": query},
                reasoning="User requested statistical data profiling.",
            )

        # 6. SQL Query Generation / Execution
        if any(w in q for w in ["select ", "insert into", "group by", "inner join", "left join", "sql query", "sql server", "postgres", "sqlite", "mysql", "union vs join", "duplicate account"]):
            return TaskRouting(
                task_type=TaskType.SQL_QUERY,
                tool_name="sql_validator",
                confidence=0.92,
                parameters={"query": query},
                reasoning="User submitted or requested SQL query syntax, validation, or execution.",
            )

        # 7. Python Execution
        if any(w in q for w in ["run python", "execute python", "python code", "calculate with python", "```python"]):
            return TaskRouting(
                task_type=TaskType.PYTHON_EXECUTION,
                tool_name="code_sandbox",
                confidence=0.92,
                parameters={"code": query},
                reasoning="User requested execution of Python source code.",
            )

        # 8. Mathematical Calculation
        if any(w in q for w in ["calculate", "evaluate", "solve", "what is the sum", "what is the formula", "math", "derive"]) or re.search(r"[\d\.\)]\s*[\+\-\*\/\^\=]\s*[\d\.\(]", q):
            return TaskRouting(
                task_type=TaskType.MATHEMATICAL_CALCULATION,
                tool_name="math_verifier",
                confidence=0.88,
                parameters={"expression": query},
                reasoning="User requested mathematical computation and independent verification.",
            )

        # 9. Web Research
        if any(w in q for w in ["search the web", "look up online", "google", "web research", "find latest news", "browse web"]):
            return TaskRouting(
                task_type=TaskType.WEB_RESEARCH,
                tool_name="bounded_research_engine",
                confidence=0.85,
                parameters={"query": query},
                reasoning="User requested external web search.",
            )

        # 10. Knowledge Retrieval
        if any(w in q for w in ["who is", "what was", "in my notes", "remember when", "knowledge graph"]):
            return TaskRouting(
                task_type=TaskType.KNOWLEDGE_RETRIEVAL,
                tool_name="memory_store",
                confidence=0.80,
                parameters={"query": query},
                reasoning="User requested internal cognitive memory or graph retrieval.",
            )

        # Default: General Text Generation
        return TaskRouting(
            task_type=TaskType.TEXT_GENERATION,
            tool_name="foundation_engine",
            confidence=0.75,
            parameters={"prompt": query},
            reasoning="General conversational or text completion task.",
        )
