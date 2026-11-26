"""Intent classification engine for Aarohan 2.0.

Detects single and multi-intent queries across 21 core intent categories
defined in Section 4.3 of the Master Engineering Transformation PRD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"
    KNOWLEDGE_QUESTION = "KNOWLEDGE_QUESTION"
    RESEARCH = "RESEARCH"
    WEB_SEARCH = "WEB_SEARCH"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
    DATASET_GENERATION = "DATASET_GENERATION"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    CHART_GENERATION = "CHART_GENERATION"
    EXCEL_GENERATION = "EXCEL_GENERATION"
    DOCUMENT_CREATION = "DOCUMENT_CREATION"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    CODE_GENERATION = "CODE_GENERATION"
    CODE_EXECUTION = "CODE_EXECUTION"
    SQL_ASSISTANCE = "SQL_ASSISTANCE"
    MATHEMATICAL_CALCULATION = "MATHEMATICAL_CALCULATION"
    SUMMARIZATION = "SUMMARIZATION"
    TRANSLATION = "TRANSLATION"
    MEMORY_RECALL = "MEMORY_RECALL"
    LEARNING_REQUEST = "LEARNING_REQUEST"
    MULTIMODAL_REQUEST = "MULTIMODAL_REQUEST"


@dataclass
class IntentMatch:
    intent: Intent
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class IntentResult:
    primary_intent: Intent
    intents: list[IntentMatch]
    is_multi_intent: bool

    @property
    def intent_names(self) -> list[str]:
        return [m.intent.value for m in self.intents]


class IntentClassifier:
    """Multi-intent classifier mapping user queries to operational action pipelines."""

    INTENT_PATTERNS: dict[Intent, list[str]] = {
        Intent.DATASET_GENERATION: [
            r"\b(create|generate|make|synthesize|build)\b.*\b(dataset|data set|dummy data|mock data|records|sample data)\b",
            r"\b\d+k?\s+(rows|records|transactions|customers|orders)\b",
            r"\b(banking|ecommerce|sales|employee|loan)\s+(data|dataset)\b",
        ],
        Intent.EXCEL_GENERATION: [
            r"\b(excel|xlsx|spreadsheet|csv file|workbook)\b",
            r"\b(export|download|save|convert)\b.*\b(excel|xlsx|spreadsheet|csv)\b",
            r"\b(pivot table|worksheets?)\b",
        ],
        Intent.IMAGE_GENERATION: [
            r"\b(create|generate|make|draw|render|produce)\b.*\b(image|picture|photo|illustration|drawing|visual|wallpaper|portrait|artwork)\b",
            r"\bimage of\b",
        ],
        Intent.CHART_GENERATION: [
            r"\b(chart|graph|plot|visualize|visualization|histogram|scatterplot|bar chart|line graph)\b",
            r"\b(plot|draw|show)\s+(a|the)?\s*(chart|graph|trend)\b",
        ],
        Intent.DATA_ANALYSIS: [
            r"\b(analyze|analyse|analysis|profile|clean|cleanse|correlation|summary statistics|eda|insights?)\b.*\b(data|dataset|csv|excel|file)\b",
            r"\b(top \d+|monthly revenue|mtd|lmtd|missing values|duplicates?)\b",
        ],
        Intent.SQL_ASSISTANCE: [
            r"\b(sql|query|queries|select|insert|update|delete|join|group by|order by|database|table|schema)\b",
            r"\b(sql server|postgresql|mysql|sqlite)\b",
        ],
        Intent.CODE_GENERATION: [
            r"\b(code|script|function|program|algorithm|implement|class|api endpoint)\b.*\b(python|javascript|typescript|c\+\+|dax|powershell)\b",
            r"\bwrite\s+(a|the)?\s*(python|code|script|function|routine)\b",
        ],
        Intent.CODE_EXECUTION: [
            r"\b(run|execute|test|eval|evaluate)\s+(this|the)?\s*(code|script|python)\b",
        ],
        Intent.RESEARCH: [
            r"\b(research|investigate|study|deep dive|literature)\b",
            r"\b(what is new in|latest updates? in|recent developments? in)\b",
        ],
        Intent.WEB_SEARCH: [
            r"\b(search the web|google|find online|current news|latest news|give me sources|provide sources|references)\b",
        ],
        Intent.MATHEMATICAL_CALCULATION: [
            r"\b(calculate|compute|solve|math|formula|equation|\+|\-|\*|\/|\^|\=)\b",
            r"\bwhat is \d+[\+\-\*\/]\d+\b",
        ],
        Intent.MEMORY_RECALL: [
            r"\b(do you remember|recall|last time|as i said before|remember when|stored in memory)\b",
        ],
        Intent.LEARNING_REQUEST: [
            r"\b(learn|teach me|study guide|curriculum|roadmap for|how do i start learning)\b",
        ],
        Intent.SUMMARIZATION: [
            r"\b(summarize|summarise|summary|tldr|brief|key takeaways)\b",
        ],
        Intent.TRANSLATION: [
            r"\b(translate|translation|in hindi|in spanish|in french|in marathi)\b",
        ],
        Intent.DOCUMENT_ANALYSIS: [
            r"\b(read this pdf|analyze document|extract from docx|read file)\b",
        ],
    }

    def classify(self, query: str) -> IntentResult:
        """Classify a query into primary and secondary operational intents."""
        normalized_lower = query.lower().strip()
        matches: list[IntentMatch] = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            matched_pats: list[str] = []
            for pat in patterns:
                if re.search(pat, normalized_lower):
                    matched_pats.append(pat)

            if matched_pats:
                # Score higher if multiple patterns match
                conf = min(0.7 + (len(matched_pats) * 0.15), 0.99)
                matches.append(IntentMatch(intent=intent, confidence=conf, matched_patterns=matched_pats))

        # Sort matches by confidence descending
        matches.sort(key=lambda x: x.confidence, reverse=True)

        # Default fallback
        if not matches:
            primary = Intent.GENERAL_CHAT if len(normalized_lower.split()) < 4 else Intent.KNOWLEDGE_QUESTION
            matches.append(IntentMatch(intent=primary, confidence=0.5, matched_patterns=["default_fallback"]))
        else:
            primary = matches[0].intent

        return IntentResult(
            primary_intent=primary,
            intents=matches,
            is_multi_intent=len(matches) > 1,
        )
