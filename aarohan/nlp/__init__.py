"""Aarohan Intelligent Language Understanding and NLP Engine.

Provides:
  - Text normalization with technical terminology protection
  - Typo and spelling tolerance (English, Indian English, Hinglish, colloquialisms)
  - Multi-intent classification across 21 core intent categories
  - Contextual follow-up and entity resolution
"""

from aarohan.capabilities import (
    Capability,
    CapabilityRegistry,
    CapabilityStatus,
    get_capability_registry,
    resolve_capability_query,
)
from aarohan.nlp.context import ContextResolver
from aarohan.nlp.identity import resolve_identity_query
from aarohan.nlp.intent import Intent, IntentClassifier, IntentResult
from aarohan.nlp.normalizer import NormalizedQuery, TextNormalizer
from aarohan.nlp.science import ScientificLaw, resolve_scientific_law

__all__ = [
    "TextNormalizer",
    "NormalizedQuery",
    "Intent",
    "IntentResult",
    "IntentClassifier",
    "ContextResolver",
    "resolve_identity_query",
    "ScientificLaw",
    "resolve_scientific_law",
    "Capability",
    "CapabilityRegistry",
    "CapabilityStatus",
    "get_capability_registry",
    "resolve_capability_query",
]
