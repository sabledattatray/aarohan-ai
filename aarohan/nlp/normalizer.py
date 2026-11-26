"""Text normalization and spelling intelligence for Aarohan.

Preserves the original user query, protects technical terminology,
and normalizes common spelling errors, informalisms, and colloquial typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class NormalizedQuery:
    """Encapsulates the normalization output."""

    original: str
    normalized: str
    corrections: dict[str, str] = field(default_factory=dict)
    protected_tokens: list[str] = field(default_factory=list)

    @property
    def has_corrections(self) -> bool:
        return bool(self.corrections)

    @property
    def normalized_text(self) -> str:
        return self.normalized


class TextNormalizer:
    """Production-grade text normalizer with technical entity protection."""

    # Technical entities and acronyms that must NEVER be altered
    PROTECTED_TERMS: set[str] = {
        "sql",
        "powerbi",
        "power bi",
        "fabric",
        "microsoft fabric",
        "python",
        "pytorch",
        "cuda",
        "numpy",
        "pandas",
        "polars",
        "duckdb",
        "dax",
        "excel",
        "rope",
        "swiglu",
        "rmsnorm",
        "json",
        "jsonl",
        "csv",
        "xlsx",
        "parquet",
        "api",
        "rest",
        "sse",
        "html",
        "css",
        "javascript",
        "typescript",
        "aws",
        "azure",
        "gcp",
        "vram",
        "gpu",
        "cpu",
        "rtx",
        "2060",
        "qwen",
        "aarohan",
        "sqlite",
        "sqlite-vec",
        "bm25",
        "fts5",
    }

    # High-frequency typos and colloquial mappings
    TYPO_MAP: dict[str, str] = {
        # Core verbs
        "conect": "connect",
        "creat": "create",
        "crate": "create",
        "craete": "create",
        "explane": "explain",
        "explian": "explain",
        "generat": "generate",
        "downlod": "download",
        "dowload": "download",
        "dwnld": "download",
        "mak": "make",
        "writ": "write",
        "wrte": "write",
        "comput": "compute",
        "analize": "analyze",
        "analyse": "analyze",
        # Common nouns
        "datset": "dataset",
        "dtaset": "dataset",
        "dset": "dataset",
        "custmers": "customers",
        "customrs": "customers",
        "cust": "customer",
        "exel": "excel",
        "excle": "excel",
        "acount": "account",
        "acc": "account",
        "balence": "balance",
        "transction": "transaction",
        "transctions": "transactions",
        "intrest": "interest",
        "fuction": "function",
        "func": "function",
        "algoritm": "algorithm",
        "transfomer": "transformer",
        "transfromer": "transformer",
        "artifitial": "artificial",
        "inteligence": "intelligence",
        "intellegence": "intelligence",
        # Common prepositions / helper words
        "wit": "with",
        "wht": "what",
        "wat": "what",
        "hw": "how",
        "bcoz": "because",
        "bcz": "because",
        "becoz": "because",
        "plz": "please",
        "pls": "please",
        "wrk": "work",
        "wrks": "works",
        "ur": "your",
        "u": "you",
        "r": "are",
        "msg": "message",
        "diffrence": "difference",
        "diff": "difference",
        "shud": "should",
        "wud": "would",
        "cud": "could",
        "abt": "about",
        "info": "information",
    }

    # Phrase-level normalization (Indian English & colloquialisms)
    PHRASE_MAP: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bpower\s*bi\b", re.IGNORECASE), "Power BI"),
        (re.compile(r"\bsql\s*server\b", re.IGNORECASE), "SQL Server"),
        (re.compile(r"\bmicrosoft\s*fabric\b", re.IGNORECASE), "Microsoft Fabric"),
        (re.compile(r"\bmake\s+excel\s+and\s+give\s+download\b", re.IGNORECASE), "Create an Excel file and provide a download link"),
        (re.compile(r"\bdo\s+one\s+thing\b", re.IGNORECASE), "please"),
        (re.compile(r"\brevert\s+back\b", re.IGNORECASE), "reply"),
        (re.compile(r"\btell\s+me\s+about\b", re.IGNORECASE), "explain"),
        (re.compile(r"\b(first|second|third|zeroth|1st|2nd|3rd|0th|newton'?s?|ohm'?s?|boyle'?s?|charles'?|kepler'?s?|coulomb'?s?)\s+low\b", re.IGNORECASE), r"\g<1> law"),
        (re.compile(r"\blow\s+of\s+(thermodynamics|motion|conservation|gravity|physics|cooling|inertia|universal\s+gravitation)\b", re.IGNORECASE), r"law of \g<1>"),
        (re.compile(r"\b(scientific|physical|physics)\s+lows?\b", re.IGNORECASE), r"\g<1> laws"),
    ]

    def normalize(self, query: str) -> NormalizedQuery:
        """Normalize a query while preserving protected terms and original string."""
        if not query or not query.strip():
            return NormalizedQuery(original=query, normalized=query)

        original = query
        corrections: dict[str, str] = {}
        protected_found: list[str] = []

        # Check for code blocks or SQL queries to preserve intact
        if "```" in query or re.search(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\s+", query, re.IGNORECASE):
            return NormalizedQuery(original=original, normalized=original)

        working_text = query

        # Apply phrase-level normalization
        for pattern, replacement in self.PHRASE_MAP:
            if pattern.search(working_text):
                working_text = pattern.sub(replacement, working_text)

        # Word-level normalization
        tokens = re.split(r"(\s+|[^\w\s])", working_text)
        new_tokens: list[str] = []

        for token in tokens:
            lower_token = token.lower()

            if lower_token in self.PROTECTED_TERMS:
                protected_found.append(token)
                new_tokens.append(token)
            elif lower_token in self.TYPO_MAP:
                corrected = self.TYPO_MAP[lower_token]
                # Match title casing if input was capitalized
                if token.istitle():
                    corrected = corrected.capitalize()
                elif token.isupper() and len(token) > 1:
                    corrected = corrected.upper()
                corrections[token] = corrected
                new_tokens.append(corrected)
            else:
                new_tokens.append(token)

        normalized_str = "".join(new_tokens)

        # Clean redundant double spaces
        normalized_str = re.sub(r"\s{2,}", " ", normalized_str).strip()

        return NormalizedQuery(
            original=original,
            normalized=normalized_str,
            corrections=corrections,
            protected_tokens=protected_found,
        )
