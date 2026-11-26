"""Conversational context and follow-up resolution engine for Aarohan.

Resolves pronouns, anaphoric references ("this", "it", "that", "the second one"),
and tracks session entities to support natural multi-turn dialogues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ResolvedContext:
    query: str
    expanded_query: str
    has_follow_up: bool
    referenced_topic: str | None = None
    is_ambiguous: bool = False
    clarification_question: str | None = None

    @property
    def resolved_query(self) -> str:
        return self.expanded_query


class ContextResolver:
    """Resolves conversational references against previous dialogue turns."""

    ANAPHORIC_PATTERNS: list[str] = [
        r"\b(this|it|that|these|those)\b",
        r"\b(second|first|third|last|previous)\s+(one|option|item|point)\b",
        r"\b(add|append|remove|filter|convert|save|export)\s+(it|this)\b",
        r"^(and\s+)?(what\s+about|how\s+about)\b",
        r"^i\s+need\s+this\s+in\b",
        r"^(now|then)\s+(do|add|make|give|calculate)\b",
    ]

    def is_follow_up(self, query: str) -> bool:
        """Check if query is likely an anaphoric follow-up dependent on context."""
        lower = query.lower().strip()
        return any(bool(re.search(pat, lower)) for pat in self.ANAPHORIC_PATTERNS)

    def resolve(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ResolvedContext:
        """Resolve references in query using recent dialogue history."""
        if not history or not self.is_follow_up(query):
            return ResolvedContext(
                query=query,
                expanded_query=query,
                has_follow_up=False,
            )

        # Extract last assistant and user turns
        last_user_turn = ""
        last_assistant_turn = ""

        for turn in reversed(history):
            if turn.get("role") == "user" and not last_user_turn:
                last_user_turn = turn.get("content", "").strip()
            elif turn.get("role") == "assistant" and not last_assistant_turn:
                last_assistant_turn = turn.get("content", "").strip()

            if last_user_turn and last_assistant_turn:
                break

        # Check for ambiguous standalone short queries
        tokens = query.strip().split()
        if len(tokens) <= 2 and query.lower() in {"why", "how", "what", "more", "explain"}:
            return ResolvedContext(
                query=query,
                expanded_query=f"{query} regarding: {last_user_turn}",
                has_follow_up=True,
                referenced_topic=last_user_turn,
                is_ambiguous=False,
            )

        # Construct contextual expansion
        context_hint = f" [Context from previous turn: '{last_user_turn}']"
        expanded = f"{query}{context_hint}"

        return ResolvedContext(
            query=query,
            expanded_query=expanded,
            has_follow_up=True,
            referenced_topic=last_user_turn,
            is_ambiguous=False,
        )
