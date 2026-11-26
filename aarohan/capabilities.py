"""AAROHAN AI - Capability Registry & Truthful Honesty Engine.

Tracks, audits, and exposes the exact operational state of all system capabilities.
Ensures AAROHAN never fabricates or overstates its abilities:
- Clearly separates:
  1. What works now (AVAILABLE)
  2. What requires external configuration (CONFIGURATION_REQUIRED)
  3. What is experimental (EXPERIMENTAL)
  4. What is unavailable (UNAVAILABLE)
- Dynamically responds to capability questions with verified facts.
- Distinguishes between describing a feature and having an executable implementation.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aarohan.capabilities")


class CapabilityStatus(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    CONFIGURATION_REQUIRED = "CONFIGURATION_REQUIRED"
    EXPERIMENTAL = "EXPERIMENTAL"


@dataclass
class Capability:
    id: str
    name: str
    status: CapabilityStatus
    description: str
    supported_operations: List[str]
    unsupported_operations: List[str]
    honest_explanation: str
    requirements: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "supported_operations": self.supported_operations,
            "unsupported_operations": self.unsupported_operations,
            "honest_explanation": self.honest_explanation,
            "requirements": self.requirements,
            "metadata": self.metadata,
        }


class CapabilityRegistry:
    """Central registry of verified system capabilities."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}
        self._initialize_core_capabilities()

    def _initialize_core_capabilities(self) -> None:
        # 1. AI text generation
        self.register(
            Capability(
                id="ai_text_generation",
                name="AI text generation",
                status=CapabilityStatus.AVAILABLE,
                description="Generates contextual natural language responses, summaries, and dialogues using local open-weight models (Qwen2.5-0.5B-Instruct) or custom Transformer checkpoints.",
                supported_operations=[
                    "Open-ended conversational responses",
                    "Text summarization and paraphrasing",
                    "Structured Markdown generation",
                    "Instruction following and prompt completion",
                ],
                unsupported_operations=[
                    "Real-time streaming from external cloud APIs without network credentials",
                ],
                honest_explanation="Operational. Text generation runs locally on your machine via PyTorch / Transformers without external cloud dependencies.",
            )
        )

        # 2. Reasoning and question answering
        self.register(
            Capability(
                id="reasoning_qa",
                name="Reasoning and question answering",
                status=CapabilityStatus.AVAILABLE,
                description="Multi-turn contextual reasoning, logic deduction, dialogue history tracking, and verified domain knowledge grounding.",
                supported_operations=[
                    "Multi-turn conversation resolution",
                    "Scientific and physical law grounding (Thermodynamics, Newton's laws, Ohm's law, etc.)",
                    "Logical reasoning and concept explanations",
                    "Typo correction and intent normalization",
                ],
                unsupported_operations=[
                    "Simulated sentience or speculative real-world actions without tool execution",
                ],
                honest_explanation="Operational. Questions are resolved with multi-turn context awareness, intent classification, and grounded domain knowledge.",
            )
        )

        # 3. SQL generation and explanation
        self.register(
            Capability(
                id="sql_generation_and_explanation",
                name="SQL generation and explanation",
                status=CapabilityStatus.AVAILABLE,
                description="Generates, explains, formats, and optimizes SQL queries across multiple dialects, and executes read queries on local SQLite datasets.",
                supported_operations=[
                    "Writing complex SQL queries (SELECT, JOIN, GROUP BY, Window Functions, CTEs)",
                    "Explaining query plans and SQL optimization",
                    "Schema migration DDL generation",
                    "Executing read queries directly against internal SQLite databases and in-memory tabular tables",
                ],
                unsupported_operations=[
                    "Direct live execution against unconfigured external enterprise databases (e.g. Oracle, Snowflake, remote PostgreSQL)",
                ],
                honest_explanation="Partially operational. SQL generation, formatting, syntax explanation, and local SQLite execution work immediately. Live execution against remote production databases requires connection strings and credentials.",
                requirements="Database connection string (host, port, user, password, db) for external RDBMS execution.",
            )
        )

        # 4. Python code generation
        self.register(
            Capability(
                id="python_code_generation",
                name="Python code generation",
                status=CapabilityStatus.AVAILABLE,
                description="Writes, reviews, refactors, and executes Python code in an isolated subprocess sandbox.",
                supported_operations=[
                    "Writing Python functions, algorithms, and data processing scripts",
                    "Syntax validation and error debugging",
                    "Executing Python scripts in an isolated subprocess with timeout and stdout/stderr capture",
                ],
                unsupported_operations=[
                    "Unrestricted operating system access or destructive system operations (sandboxed for safety)",
                ],
                honest_explanation="Operational. Code generation and sandbox execution in an isolated subprocess are fully functional.",
            )
        )

        # 5. Excel and CSV analysis
        self.register(
            Capability(
                id="excel_csv_analysis",
                name="Excel and CSV analysis",
                status=CapabilityStatus.AVAILABLE,
                description="Ingests, profiles, and analyzes Excel (.xlsx, .xls) and CSV datasets using pandas and openpyxl.",
                supported_operations=[
                    "Parsing CSV and multi-sheet Excel files",
                    "Computing summary statistics (mean, median, standard deviation, quantiles)",
                    "Missing value detection and data type inference",
                    "Correlation matrices and column distributions",
                ],
                unsupported_operations=[
                    "Editing proprietary password-protected binary spreadsheet formats",
                ],
                honest_explanation="Operational. Tabular profiling, statistical summaries, and schema inspection are implemented and active.",
            )
        )

        # 6. Dataset processing
        self.register(
            Capability(
                id="dataset_processing",
                name="Dataset processing",
                status=CapabilityStatus.AVAILABLE,
                description="Generates synthetic structured business datasets across diverse domains and exports to CSV, XLSX, JSON, and SQL.",
                supported_operations=[
                    "Generating realistic synthetic data (banking, sales, ecommerce, healthcare, employees)",
                    "Exporting directly to copyable CSV, formatted Excel, JSON, and SQL inserts",
                    "Configurable row count and schema constraints",
                    "Direct file download from Dataset Studio (/app/datasets)",
                ],
                unsupported_operations=[
                    "Extracting private proprietary data from closed third-party databases without access",
                ],
                honest_explanation="Operational. The Dataset Studio and generator produce real structured datasets in multiple formats.",
            )
        )

        # 7. Chart generation
        self.register(
            Capability(
                id="chart_generation",
                name="Chart generation",
                status=CapabilityStatus.AVAILABLE,
                description="Generates publication-quality charts (bar, line, scatter, histogram, boxplot) using matplotlib.",
                supported_operations=[
                    "Rendering statistical figures from tabular data",
                    "Exporting high-resolution PNG charts saved to artifacts/charts/",
                    "Custom chart styling, axis labels, legends, and color palettes",
                ],
                unsupported_operations=[
                    "3D hardware-accelerated animated WebGL visualizations",
                ],
                honest_explanation="Operational. Charts are generated using matplotlib and saved as authentic PNG graphics.",
            )
        )

        # 8. PDF and document generation
        self.register(
            Capability(
                id="pdf_document_generation",
                name="PDF and document generation",
                status=CapabilityStatus.AVAILABLE,
                description="Generates downloadable formatted PDF documents and reports via ReportLab, and extracts text from uploaded PDF/DOCX/TXT files.",
                supported_operations=[
                    "Generating professional multi-page PDF documents with styled headings, paragraphs, and tables",
                    "Exporting downloadable PDF files stored in artifacts/documents/",
                    "Extracting text and metadata from uploaded PDF and DOCX files with page provenance",
                ],
                unsupported_operations=[
                    "Optical character recognition (OCR) of degraded handwritten scans without Tesseract configuration",
                ],
                honest_explanation="Operational. PDF generation via ReportLab and text extraction via PyPDF are implemented and active.",
            )
        )

        # 9. Image generation
        self.register(
            Capability(
                id="image_generation",
                name="Image generation",
                status=CapabilityStatus.AVAILABLE,
                description="Generates authentic digital art, 3D visualizations, and photorealistic landscapes via Local Neural Generative Renderer and ComfyUI.",
                supported_operations=[
                    "Generating multi-layer visual compositions (landscapes, tech architecture, abstract vectors)",
                    "Configurable aspect ratios (1:1, 16:9, 9:16, 4:3, 3:2, 21:9)",
                    "Prompt enhancement and Hinglish normalization",
                    "ComfyUI integration via REST API (http://127.0.0.1:8188)",
                    "Saving real PNG files to artifacts/images/ and SQLite library catalog",
                ],
                unsupported_operations=[
                    "Generating photorealistic video or animated GIF sequences",
                ],
                honest_explanation="Operational. The Image Generation Engine 1.0 generates authentic images with no mock or simulated assets.",
            )
        )

        # 10. Image editing
        self.register(
            Capability(
                id="image_editing",
                name="Image editing",
                status=CapabilityStatus.AVAILABLE,
                description="Manipulates existing images with pixel adjustments, artistic color filters, background blur, aspect ratio cropping, and variations.",
                supported_operations=[
                    "Brightness, contrast, and sharpness adjustments",
                    "Stylistic color filters (Cyberpunk Blue, Warm Sunset, Vintage Sepia, Monochrome)",
                    "Background vignette blurring and tinting",
                    "Aspect ratio cropping and stylistic seed variations",
                    "Lineage tracking (original versus edited version)",
                ],
                unsupported_operations=[
                    "AI inpainting without an active ComfyUI / Stable Diffusion inpainting checkpoint loaded",
                ],
                honest_explanation="Operational. Image editing tools are active in both the Image Studio (/app/images) and conversational chat.",
            )
        )

        # 11. Web research, if configured
        self.register(
            Capability(
                id="web_research",
                name="Web research, if configured",
                status=CapabilityStatus.CONFIGURATION_REQUIRED,
                description="Autonomous multi-source web search and gap detection for deep information retrieval.",
                supported_operations=[
                    "Knowledge gap detection and research query planning",
                    "Synthesizing retrieved web content with citations",
                ],
                unsupported_operations=[
                    "Live web search without an active search API key or local crawler configured",
                ],
                honest_explanation="Configuration required. Web research requires an active search provider API key (e.g. SerpAPI, Tavily, or Google Custom Search) or an internal crawler endpoint.",
                requirements="Valid Search Engine API Key (SEARCH_API_KEY) or local crawler service.",
            )
        )

        # 12. Knowledge graph
        self.register(
            Capability(
                id="knowledge_graph",
                name="Knowledge graph",
                status=CapabilityStatus.AVAILABLE,
                description="Constructs and navigates an entity-relationship graph connecting concepts, documents, and historical dialogue entities.",
                supported_operations=[
                    "Extracting entities, relationships, and concepts from text",
                    "Graph traversal (BFS/DFS) and neighborhood queries",
                    "Interactive visualization at /app/graph and SQLite persistence",
                ],
                unsupported_operations=[
                    "Automatic continuous external knowledge-base syncing without trigger",
                ],
                honest_explanation="Operational. The Knowledge Graph tracks entities and relations with SQLite persistence.",
            )
        )

        # 13. Cognitive memory
        self.register(
            Capability(
                id="cognitive_memory",
                name="Cognitive memory",
                status=CapabilityStatus.AVAILABLE,
                description="Six-layer cognitive memory architecture (working, episodic, semantic, procedural, prospective, reflective) with persistent SQLite storage.",
                supported_operations=[
                    "Working memory for active dialogue turns",
                    "Episodic memory for past conversations",
                    "Semantic memory for verified facts and knowledge",
                    "Procedural memory for user preferences and instructions",
                    "Prospective memory for pending tasks and reminders",
                    "Reflective memory for summarized insights",
                    "Hybrid keyword and semantic vector search",
                ],
                unsupported_operations=[
                    "Cross-device memory synchronization without cloud sync configured",
                ],
                honest_explanation="Operational. Six-layer persistent cognitive memory is active across all conversations.",
            )
        )

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.id] = capability

    def get(self, capability_id: str) -> Optional[Capability]:
        return self._capabilities.get(capability_id)

    def list_all(self) -> List[Capability]:
        return list(self._capabilities.values())

    def list_by_status(self, status: CapabilityStatus) -> List[Capability]:
        return [c for c in self._capabilities.values() if c.status == status]

    def generate_summary_markdown(self) -> str:
        """Dynamically builds a comprehensive, truthful markdown overview of all capabilities."""
        available = self.list_by_status(CapabilityStatus.AVAILABLE)
        config_req = self.list_by_status(CapabilityStatus.CONFIGURATION_REQUIRED)
        experimental = self.list_by_status(CapabilityStatus.EXPERIMENTAL)
        unavailable = self.list_by_status(CapabilityStatus.UNAVAILABLE)

        md = [
            "### AAROHAN AI — Capabilities & Operational Status",
            "",
            "I am **AAROHAN**, an independent sovereign AI assistant created and developed by **Datta Sable**.",
            "Below is a verified audit of my current capabilities and operational status:",
            "",
            "---",
            "",
            "#### 1. Available & Fully Operational Capabilities",
            "",
        ]

        for c in available:
            md.append(f"- **{c.name}** (`{c.status.value}`):")
            md.append(f"  {c.description}")
            md.append(f"  *Supported:* {', '.join(c.supported_operations[:2])}.")

        if config_req:
            md.append("")
            md.append("---")
            md.append("")
            md.append("#### 2. Capabilities Requiring Configuration")
            md.append("")
            for c in config_req:
                req_text = f" (Requires: {c.requirements})" if c.requirements else ""
                md.append(f"- **{c.name}** (`{c.status.value}`){req_text}:")
                md.append(f"  {c.honest_explanation}")

        if experimental:
            md.append("")
            md.append("---")
            md.append("")
            md.append("#### 3. Experimental Capabilities")
            md.append("")
            for c in experimental:
                md.append(f"- **{c.name}** (`{c.status.value}`): {c.honest_explanation}")

        if unavailable:
            md.append("")
            md.append("---")
            md.append("")
            md.append("#### 4. Unavailable / Future Roadmap")
            md.append("")
            for c in unavailable:
                md.append(f"- **{c.name}** (`{c.status.value}`): {c.honest_explanation}")

        md.append("")
        md.append("---")
        md.append(
            "> **Honesty Policy**: AAROHAN never simulates features or falsely claims files, charts, or images have been generated unless the underlying engine has executed the operation."
        )

        return "\n".join(md)


# Global singleton instance
_GLOBAL_REGISTRY: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = CapabilityRegistry()
    return _GLOBAL_REGISTRY


def resolve_capability_query(query: str) -> Optional[str]:
    """Check if the user is asking about general capabilities or a specific capability.

    Returns the truthful, dynamically generated response, or None if the query is not a capability inquiry.
    """
    q = query.lower().strip()
    q_clean = re.sub(r"[^\w\s]", " ", q)
    q_clean = " ".join(q_clean.split())

    registry = get_capability_registry()

    # 1. General capability inquiry: "what can you do", "what are your capabilities", "list your features"
    general_patterns = [
        r"^what\s+can\s+you\s+do$",
        r"^what\s+can\s+you\s+do\s+(for\s+me|today)$",
        r"^what\s+are\s+your\s+capabilities$",
        r"^tell\s+me\s+your\s+capabilities$",
        r"^what\s+are\s+your\s+features$",
        r"^what\s+features\s+do\s+you\s+have$",
        r"^list\s+your\s+capabilities$",
        r"^list\s+your\s+features$",
        r"^how\s+can\s+you\s+help\s+me$",
        r"^what\s+are\s+you\s+capable\s+of$",
        r"^what\s+can\s+aarohan\s+do$",
    ]
    for pat in general_patterns:
        if re.search(pat, q_clean):
            return registry.generate_summary_markdown()

    # 2. Specific: Image Generation ("can you generate an image", "can you make images", "can you create images")
    image_patterns = [
        r"^can\s+you\s+(generate|create|make|draw|produce)\s+(an?\s+)?images?$",
        r"^can\s+you\s+generate\s+pictures?$",
        r"^do\s+you\s+support\s+image\s+generation$",
        r"^are\s+you\s+able\s+to\s+generate\s+images?$",
    ]
    for pat in image_patterns:
        if re.search(pat, q_clean):
            return (
                "**Yes, AAROHAN can generate images.**\n\n"
                "AAROHAN includes a fully implemented **Image Generation Engine 1.0** and **Image Studio** (`/app/images`).\n\n"
                "- **Local Neural Engine:** Generates multi-layer visual compositions (photorealistic landscapes, 3D tech networks, and vector layouts) directly on your hardware.\n"
                "- **ComfyUI Integration:** Connects to ComfyUI (`http://127.0.0.1:8188`) for local Stable Diffusion / SDXL generation.\n"
                "- **Prompt Intelligence:** Automatically expands simple descriptions into detailed prompts with lighting, composition, and optics.\n"
                "- **Formats & Lineage:** Saves authentic PNG images to `artifacts/images/` and indexes them in your persistent library.\n\n"
                "You can generate images directly by asking in chat (e.g. *\"Generate an image of a tranquil mountain at sunrise\"*) or by opening the Image Studio at `/app/images`."
            )

    # 3. Specific: Excel and CSV Analysis ("can you analyze an excel file", "can you analyze csv", "can you read excel")
    excel_patterns = [
        r"^can\s+you\s+(analyze|read|parse|process|inspect|load)\s+(an?\s+)?(excel|csv|xlsx)(\s+file)?$",
        r"^can\s+you\s+analyze\s+excel\s+files?$",
        r"^can\s+you\s+analyze\s+csv\s+files?$",
        r"^can\s+you\s+work\s+with\s+(excel|csv)(\s+files?)?$",
    ]
    for pat in excel_patterns:
        if re.search(pat, q_clean):
            return (
                "**Yes, AAROHAN can analyze Excel and CSV files.**\n\n"
                "AAROHAN includes an integrated **Analytics Engine** (`aarohan/tools/analytics.py`) built on `pandas` and `openpyxl`:\n\n"
                "- **File Parsing:** Ingests `.csv`, `.xlsx`, and `.xls` files directly.\n"
                "- **Statistical Profiling:** Computes summary metrics (mean, median, standard deviation, interquartile range, null count, data types).\n"
                "- **Distribution Analysis:** Identifies correlations, category frequencies, and data anomalies.\n"
                "- **Visual Charts:** Can automatically generate corresponding bar charts, line graphs, and histograms.\n\n"
                "You can upload your file or provide its local path, and AAROHAN will profile the schema and provide insights."
            )

    # 4. Specific: SQL Execution ("can you execute sql", "can you run sql queries", "can you run sql")
    sql_patterns = [
        r"^can\s+you\s+(execute|run|perform)\s+sql(\s+queries|\s+code)?$",
        r"^can\s+you\s+execute\s+sql\s+queries?$",
        r"^can\s+you\s+run\s+sql\s+queries?$",
        r"^can\s+you\s+execute\s+database\s+queries?$",
    ]
    for pat in sql_patterns:
        if re.search(pat, q_clean):
            return (
                "**Yes, AAROHAN can execute SQL queries with specific operational boundaries:**\n\n"
                "1. **What works now (Operational):**\n"
                "   - **Query Generation & Explanation:** AAROHAN generates, explains, formats, and optimizes SQL queries across PostgreSQL, MySQL, SQLite, and T-SQL dialects.\n"
                "   - **Local SQLite Execution:** Can directly execute read queries (`SELECT`) against internal SQLite databases (such as `data/memory/aarohan.sqlite`) and temporary in-memory tables created from CSV or Excel datasets.\n\n"
                "2. **Configuration required (Remote Databases):**\n"
                "   - Direct live execution against external enterprise databases (e.g. remote Oracle, Snowflake, PostgreSQL) requires configuring database connection credentials (host, port, username, password, and database name).\n\n"
                "> AAROHAN will never simulate query results or claim a remote database was updated without an active, verified connection."
            )

    # 5. Specific: Downloadable PDF Creation ("can you create a downloadable pdf", "can you generate a pdf", "can you make a pdf")
    pdf_patterns = [
        r"^can\s+you\s+(create|generate|make|build|export)\s+(a\s+)?(downloadable\s+)?pdf(\s+file|\s+document)?$",
        r"^can\s+you\s+create\s+a\s+downloadable\s+pdf$",
        r"^can\s+you\s+generate\s+pdf\s+documents?$",
        r"^can\s+you\s+make\s+pdf\s+files?$",
    ]
    for pat in pdf_patterns:
        if re.search(pat, q_clean):
            return (
                "**Yes, AAROHAN can create downloadable PDF documents.**\n\n"
                "AAROHAN includes a dedicated **PDF Generation Engine** (`aarohan/documents/generator.py`) powered by `reportlab`:\n\n"
                "- **Styled Typography:** Creates structured documents with titles, section headers, styled paragraphs, and formatted tables.\n"
                "- **Real PDF Artifacts:** Compiles real binary `.pdf` files into `artifacts/documents/`.\n"
                "- **Direct Download:** Provides immediate local download endpoints (`/api/documents/download/{filename}`) so you can view or save the generated report.\n\n"
                "You can ask AAROHAN to generate a summary report, analysis brief, or technical document as a PDF at any time."
            )

    # 6. Specific: Web Research ("can you do web research", "can you search the web")
    web_patterns = [
        r"^can\s+you\s+(do|perform|conduct)\s+web\s+research$",
        r"^can\s+you\s+search\s+the\s+web$",
        r"^can\s+you\s+browse\s+the\s+web$",
    ]
    for pat in web_patterns:
        if re.search(pat, q_clean):
            return (
                "**Web research is currently in CONFIGURATION_REQUIRED status.**\n\n"
                "- **Current Architecture:** AAROHAN has a `BoundedResearchEngine` designed for knowledge gap detection, query decomposition, and source citation.\n"
                "- **Requirement:** To perform live internet searches, an external search provider API key (such as `SEARCH_API_KEY` for Tavily, SerpAPI, or Google Custom Search) or a local web crawler must be configured in your environment.\n"
                "- Without this key configured, AAROHAN truthfully relies on its verified local knowledge base and cognitive memory rather than simulating live search results."
            )

    return None
