"""AAROHAN AI 2.0 - Data Analytics & Chart Engine.

Powered by DuckDB, Pandas, NumPy, and Matplotlib.
Executes real SQL queries on tabular files and generates charts.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Headless backend for thread-safe rendering

CHARTS_DIR = Path("artifacts/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


class AnalyticsEngine:
    """Enterprise tabular analytics and visualization engine."""

    def __init__(self) -> None:
        self.conn = duckdb.connect(database=":memory:")

    def load_dataset_into_table(self, file_path: str, table_name: str = "current_data") -> Dict[str, Any]:
        """Load a CSV, Parquet, or Excel file into a DuckDB in-memory table."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext in [".csv", ".tsv", ".txt"]:
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path.as_posix()}');")
        elif ext == ".parquet":
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path.as_posix()}');")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            self.conn.register(table_name, df)
        else:
            raise ValueError(f"Unsupported format for analytics: {ext}")

        # Fetch profile
        info = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        columns = [{"name": row[0], "type": row[1], "nullable": row[2]} for row in info]
        return {
            "table_name": table_name,
            "row_count": count,
            "column_count": len(columns),
            "columns": columns,
            "file_path": str(path),
        }

    def execute_query(self, sql_query: str, limit: int = 100) -> Dict[str, Any]:
        """Execute a read-only SQL query with safety enforcement."""
        cleaned = sql_query.strip()
        lower = cleaned.lower()
        # Enforce read-only constraint
        forbidden = ["drop ", "delete ", "update ", "alter ", "insert ", "truncate ", "create "]
        for f in forbidden:
            if lower.startswith(f) or f" {f}" in lower:
                raise PermissionError("Analytics SQL execution is strictly read-only.")

        rel = self.conn.execute(cleaned)
        cols = [desc[0] for desc in rel.description]
        rows = rel.fetchmany(limit)
        return {
            "columns": cols,
            "rows": rows,
            "row_count": len(rows),
            "query": sql_query,
        }

    def profile_dataframe(self, file_path: str) -> Dict[str, Any]:
        """Compute statistical summary, missing values, and column distribution."""
        path = Path(file_path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        numeric_summary = {}
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            if not series.empty:
                numeric_summary[col] = {
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        missing = {col: int(df[col].isnull().sum()) for col in df.columns}
        duplicates = int(df.duplicated().sum())

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": missing,
            "duplicates": duplicates,
            "numeric_summary": numeric_summary,
            "sample_records": df.head(5).to_dict(orient="records"),
        }

    def generate_chart(
        self,
        file_path: str,
        chart_type: str,
        x_col: str,
        y_col: Optional[str] = None,
        title: str = "AAROHAN Analytics Chart",
    ) -> Dict[str, Any]:
        """Generate a Matplotlib chart, save to disk and return base64."""
        path = Path(file_path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        cols = list(df.columns)
        if x_col not in cols:
            x_col = cols[0]
        if y_col and y_col not in cols:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            y_col = num_cols[0] if num_cols else None

        plt.figure(figsize=(9, 4.8), facecolor="#171717")
        ax = plt.subplot(111)
        ax.set_facecolor("#212121")

        # Styling
        plt.title(title, color="#ececec", fontsize=13, pad=12, fontweight="medium")
        plt.xlabel(x_col, color="#b4b4b4", fontsize=10)
        if y_col:
            plt.ylabel(y_col, color="#b4b4b4", fontsize=10)
        ax.tick_params(colors="#8e8e8e")
        for spine in ax.spines.values():
            spine.set_color("#383838")
        ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")

        # Plot by type
        if chart_type == "bar":
            if y_col:
                agg = df.groupby(x_col)[y_col].sum().head(12)
                agg.plot(kind="bar", ax=ax, color="#6366f1", edgecolor="#818cf8")
            else:
                df[x_col].value_counts().head(12).plot(kind="bar", ax=ax, color="#6366f1")
        elif chart_type == "line":
            if y_col:
                ax.plot(df[x_col].head(50), df[y_col].head(50), color="#10b981", marker="o", linewidth=2)
            else:
                ax.plot(df[x_col].head(50), color="#10b981", linewidth=2)
        elif chart_type == "hist":
            ax.hist(df[x_col].dropna(), bins=20, color="#8b5cf6", edgecolor="#171717")
        elif chart_type == "scatter" and y_col:
            ax.scatter(df[x_col].head(200), df[y_col].head(200), color="#38bdf8", alpha=0.8)
        else:
            # Default bar
            df[x_col].value_counts().head(10).plot(kind="bar", ax=ax, color="#6366f1")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        # Save image
        out_name = f"chart_{int(pd.Timestamp.now().timestamp())}.png"
        out_path = CHARTS_DIR / out_name
        plt.savefig(out_path, dpi=120, facecolor=plt.gcf().get_facecolor(), bbox_inches="tight")

        # Encode base64 for direct browser rendering
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, facecolor=plt.gcf().get_facecolor(), bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "chart_type": chart_type,
            "title": title,
            "file_path": str(out_path),
            "filename": out_name,
            "image_base64": f"data:image/png;base64,{b64}",
        }
