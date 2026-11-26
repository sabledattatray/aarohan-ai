"""AAROHAN AI - Document & PDF Generation Engine.

Compiles structured reports, executive summaries, and tables into publication-quality
downloadable PDF documents using ReportLab.
Saved to artifacts/documents/ and accessible via download endpoints.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aarohan.documents.generator")

DOCUMENTS_DIR = Path("artifacts/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


class PDFGenerator:
    """Compiles styled PDF documents with headers, paragraphs, and tables."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or DOCUMENTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        title: str,
        sections: List[Dict[str, Any]],
        author: str = "Datta Sable / AAROHAN AI",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a structured PDF report.

        sections is a list of dicts:
        [
            {"heading": "1. Executive Summary", "content": "Summary text..."},
            {"heading": "2. Data Breakdown", "content": "Text...", "table": [["Col1", "Col2"], ["Val1", "Val2"]]}
        ]
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as e:
            logger.error("ReportLab is not installed: %s", e)
            return {
                "success": False,
                "error": "PDF generation requires reportlab library. Install via: pip install reportlab",
            }

        if not filename:
            safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:30].strip("_")
            filename = f"report_{safe_title}_{uuid.uuid4().hex[:8]}.pdf"
        elif not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"

        file_path = self.output_dir / filename

        try:
            doc = SimpleDocTemplate(
                str(file_path),
                pagesize=letter,
                rightMargin=54,
                leftMargin=54,
                topMargin=54,
                bottomMargin=54,
            )

            styles = getSampleStyleSheet()

            # Custom Aarohan Brand Styles
            title_style = ParagraphStyle(
                "AarohanTitle",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=22,
                leading=26,
                textColor=colors.HexColor("#0f172a"),
                alignment=0,
                spaceAfter=6,
            )
            meta_style = ParagraphStyle(
                "AarohanMeta",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                textColor=colors.HexColor("#64748b"),
                spaceAfter=18,
            )
            heading_style = ParagraphStyle(
                "AarohanHeading",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=14,
                spaceAfter=6,
            )
            body_style = ParagraphStyle(
                "AarohanBody",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=10,
                leading=15,
                textColor=colors.HexColor("#334155"),
                spaceAfter=8,
            )

            story = []

            # Document Title & Metadata
            story.append(Paragraph(title, title_style))
            timestamp_str = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
            story.append(Paragraph(f"Created by: {author} &bull; Generated: {timestamp_str}", meta_style))
            story.append(Spacer(1, 10))

            # Sections
            for sec in sections:
                heading = sec.get("heading")
                if heading:
                    story.append(Paragraph(heading, heading_style))

                content = sec.get("content")
                if content:
                    paragraphs = content.split("\n\n")
                    for p in paragraphs:
                        clean_p = p.replace("\n", " ").strip()
                        if clean_p:
                            story.append(Paragraph(clean_p, body_style))

                table_data = sec.get("table")
                if table_data and isinstance(table_data, list) and len(table_data) > 0:
                    t = Table(table_data, hAlign="LEFT")
                    t.setStyle(
                        TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ])
                    )
                    story.append(Spacer(1, 6))
                    story.append(t)
                    story.append(Spacer(1, 10))

            doc.build(story)

            return {
                "success": True,
                "filename": filename,
                "file_path": str(file_path),
                "download_url": f"/api/documents/download/{filename}",
                "size_bytes": file_path.stat().st_size,
            }

        except Exception as e:
            logger.error("Failed to generate PDF '%s': %s", filename, e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
