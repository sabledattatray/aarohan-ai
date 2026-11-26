"""Aarohan Intelligent Language Understanding and NLP Engine.

Provides:
  - Text normalization with technical terminology protection
  - Typo and spelling tolerance (English, Indian English, Hinglish, colloquialisms)
  - Multi-intent classification across 21 core intent categories
  - Contextual follow-up and entity resolution
"""

from aarohan.nlp.context import ContextResolver
from aarohan.nlp.intent import Intent, IntentClassifier, IntentResult
from aarohan.nlp.normalizer import NormalizedQuery, TextNormalizer

__all__ = [
    "TextNormalizer",
    "NormalizedQuery",
    "Intent",
    "IntentResult",
    "IntentClassifier",
    "ContextResolver",
]
