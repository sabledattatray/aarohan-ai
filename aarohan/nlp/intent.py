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
    MEMORY_RETRIEVE = "MEMORY_RETRIEVE"
    MEMORY_SAVE = "MEMORY_SAVE"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    LEARNING_REQUEST = "LEARNING_REQUEST"
    LEARNING_FEEDBACK = "LEARNING_FEEDBACK"
    LEARNING_EVALUATION = "LEARNING_EVALUATION"
    MODEL_TRAINING_REQUEST = "MODEL_TRAINING_REQUEST"
    MULTIMODAL_REQUEST = "MULTIMODAL_REQUEST"
    IDENTITY_AND_CREATOR = "IDENTITY_AND_CREATOR"
    SCIENTIFIC_LAW = "SCIENTIFIC_LAW"
    FACTUAL_QUESTION = "FACTUAL_QUESTION"
    REASONING = "REASONING"


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
            r"\b(make\s+(it|this)\s+(brighter|darker|more realistic|pop))\b",
            r"\b(change\s+(the\s+)?background|different\s+background)\b",
            r"\b(another\s+variation|generate\s+variation|different\s+color\s+palette|landscape\s+version|portrait\s+version)\b",
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
            r"\b(write|generate|explain|run|fix|optimize)\s+(a\s+)?sql\s+(query|queries|statement|command|joins?)\b",
            r"\bselect\s+[\*\w\s,]+\s+from\s+\w+\b",
            r"\b(insert\s+into|delete\s+from|create\s+table|alter\s+table|drop\s+table)\b",
            r"\b(sql\s+server|postgresql|mysql|sqlite\s+query|sql\s+join|sql\s+injection)\b",
            r"\bwrite\s+(a\s+)?(sql|query)\s+to\b",
            r"\bexplain\s+sql\s+(joins?|queries|indexes|transactions)\b",
        ],
        Intent.CODE_GENERATION: [
            r"\b(code|script|function|program|algorithm|implement|class|api endpoint)\b.*\b(python|javascript|typescript|c\+\+|dax|powershell)\b",
            r"\bwrite\s+(a|the)?\s*(python|code|script|function|routine|program)\b",
            r"\b(write|create|generate)\s+a\s+python\s+program\b",
            r"\bprogram\s+to\s+(calculate|compute|find|solve|sort|reverse)\b",
        ],
        Intent.FACTUAL_QUESTION: [
            r"\bwhat\s+is\s+(the\s+)?(capital|formula|chemical\s+formula|population|currency|definition|height|speed)\s+of\b",
            r"\bwho\s+(invented|discovered|wrote|founded|built|created|painted|composed)\b",
            r"\b(how\s+many\s+(planets|countries|states|bones|elements|continents|oceans))\b",
            r"\bwhen\s+was\s+.*(invented|born|founded|built|established)\b",
            r"\bwhat\s+(year|date)\s+did\b",
        ],
        Intent.REASONING: [
            r"\bexplain\s+the\s+difference\s+between\b",
            r"\bcompare\s+and\s+contrast\b",
            r"\b(why\s+does|why\s+is|how\s+come|what\s+causes)\b",
            r"\b(step\s+by\s+step|logical\s+reasoning|deduce|deduction|scenario\s+analysis)\b",
            r"\b(pros\s+and\s+cons|advantages\s+and\s+disadvantages)\b",
            r"\bexplain\s+how\s+.*works?\b",
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
            r"\bsearch\s+(?:about\s+)?me\s+(?:on\s+(?:the\s+)?(?:web|internet|google)|online)\b",
            r"\bsearch\s+(?:the\s+)?(?:web|internet|google)\s+(?:for|about)\b",
            r"\bsearch\s+(?:for\s+)?.+\s+(?:on\s+(?:the\s+)?(?:web|internet|google)|online)\b",
            r"\b(?:look\s+up|find)\s+.+\s+(?:on\s+(?:the\s+)?(?:web|internet|google)|online)\b",
            r"\bweb\s+search\b",
        ],
        Intent.MATHEMATICAL_CALCULATION: [
            r"\b(calculate|compute|solve|math|formula|equation)\b",
            r"\bwhat is \d+[\+\-\*\/]\d+\b",
        ],
        Intent.MEMORY_RECALL: [
            r"\b(do you remember|recall|last time|as i said before|remember when|stored in memory)\b",
        ],
        Intent.MEMORY_RETRIEVE: [
            r"\b(what\s+does\s+[A-Za-z0-9_\-]+\s+mean|what\s+is\s+the\s+meaning\s+of|retrieve\s+(the\s+)?(current\s+and\s+previous\s+)?meaning|what\s+is\s+stored\s+in\s+memory)\b",
            r"\b(retrieve\s+memory|recall\s+memory|search\s+memory)\b",
        ],
        Intent.MEMORY_SAVE: [
            r"\b(remember\s+that\s+[A-Za-z0-9_\-]+\s+means|save\s+this\s+fact|store\s+in\s+(persistent\s+)?memory)\b",
            r"\bkeep\s+(in\s+(your\s+)?memory|this\s+in\s+mind|that\s+in\s+mind)\b",
            r"\b(note\s+that|remember\s+that|keep\s+in\s+mind\s+that)\s+i\s+(am|live|stay|work|am\s+from)\b",
            r"\bsave\s+(this|the\s+following|my|that)\s+(fact|detail|info|information|location|address|preference)\b",
            r"\b(store|save|remember|note)\s+(my|this)\s+(location|address|city|home|preference|name|detail)\b",
            r"\b(i\s+(am|live|stay|reside|work)\s+(at|in|near))\b",
            r"\bthis\s+is\s+[A-Za-z]+\s+[A-Za-z]+\s*,?\s*(keep|store|save|remember|note)\b",
        ],
        Intent.MEMORY_UPDATE: [
            r"\b(correction:\s*[A-Za-z0-9_\-]+\s+(now\s+means|is\s+now)|update\s+memory:\s*[A-Za-z0-9_\-]+)\b",
        ],
        Intent.LEARNING_REQUEST: [
            r"\b(i\s+am\s+teaching\s+you|learn\s+these\s+rules|learning\s+experiment|learning\s+namespace|teach\s+you\s+a\s+new|new\s+decision-making\s+system|learn\s+and\s+save\s+these)\b",
            r"\brules:\s*\d+[\.\)]",
            r"\bnamespace:\s*[A-Za-z0-9_\-]+",
            r"\blearn\s+the\s+following\s+rules?\b",
        ],
        Intent.LEARNING_FEEDBACK: [
            r"\b(previous\s+decision\s+was\s+(in)?correct|decision\s+was\s+(wrong|incorrect)|apply\s+rule\s+\d+\s+before\s+rule\s+\d+|rule\s+\d+\s+should\s+override|feedback\s+on\s+(your\s+)?(previous\s+)?decision|that\s+was\s+incorrect\s+because)\b",
            r"\b(change\s+[A-Za-z0-9_\-]+\s+(?:deadline|value|policy)?\s*to\s+|update\s+[A-Za-z0-9_\-]+\s+(?:deadline|value|policy)?\s*to\s+|change\s+rule\s+|update\s+rule\s+)\b",
        ],
        Intent.LEARNING_EVALUATION: [
            r"\b(evaluate\s+(learned\s+)?rules?|benchmark\s+(the\s+)?namespace|run\s+(a\s+)?benchmark\s+on|test\s+(the\s+)?learned\s+rules?|evaluate\s+decision-making)\b",
        ],
        Intent.MODEL_TRAINING_REQUEST: [
            r"\b(fine-?tune\s+(the\s+)?model|train\s+(the\s+)?neural\s+weights?|run\s+continual\s+training|train\s+candidate\s+checkpoint|experience\s+replay\s+training)\b",
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
        Intent.IDENTITY_AND_CREATOR: [
            r"\b(your name|who are you|who made you|who created you|who developed you|who built you|creator|who is your creator|who owns you|who owns aarohan|anthropic|openai|what model powers you|underlying model)\b",
        ],
        Intent.SCIENTIFIC_LAW: [
            r"\b(law|low)\s+of\s+(thermodynamics|motion|conservation|universal gravitation|physics|cooling|inertia)\b",
            r"\b(first|second|third|zeroth|1st|2nd|3rd|0th)\s+(law|low)\s+of\s+thermodynamics\b",
            r"\b(newton'?s?|kepler'?s?|coulomb'?s?|ohm'?s?|snell'?s?|boyle'?s?|charles'?|ideal gas)\s+(law|formula|equation)\b",
            r"\bformula\s+of\s+(the\s+)?(first|second|third|zeroth|1st|2nd|3rd|0th)?\s*(law|low)\b",
            r"\b(schrodinger|heisenberg|maxwell|einstein|planck)\s+(equation|formula|principle|relation)\b",
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
