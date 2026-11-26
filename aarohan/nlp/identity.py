"""AAROHAN AI - Identity, Creator Attribution, and Model Awareness Engine.

Enforces accurate identity responses:
- Assistant Name: AAROHAN
- Creator: Datta Sable
- Project: AAROHAN AI
- Official GitHub: https://github.com/sabledattatray/aarohan-ai
- Underlying Model: Accurately identifies local Qwen2.5 or Custom Transformer checkpoint.
"""

from __future__ import annotations

import re
from typing import Optional


def resolve_identity_query(query: str, engine: str = "foundation") -> Optional[str]:
    """Check if query is asking about identity, creator, ownership, or underlying model.

    Returns the canonical truthful response, or None if the query is not an identity question.
    """
    q = query.lower().strip()
    # Normalize common contractions before stripping punctuation
    q = re.sub(r"\'s\b", " is", q)
    q = re.sub(r"\'re\b", " are", q)
    q = re.sub(r"\'m\b", " am", q)
    # Normalize punctuation and whitespace
    q_clean = re.sub(r"[^\w\s]", " ", q)
    q_clean = " ".join(q_clean.split())

    # 1. Name & Identity questions: "who are you", "what's your name", "what is your name", "what is aarohan"
    name_patterns = [
        r"^what\s+(is|\'s|s)\s+your\s+name$",
        r"^whats\s+your\s+name$",
        r"^what\s+is\s+your\s+name\s+please$",
        r"^who\s+are\s+you$",
        r"^what\s+are\s+you\s+called$",
        r"^tell\s+me\s+your\s+name$",
        r"^your\s+name$",
        r"^what\s+is\s+aarohan$",
        r"^who\s+is\s+aarohan$",
    ]
    for pat in name_patterns:
        if re.search(pat, q_clean):
            return "My name is AAROHAN, an AI assistant created by Datta Sable."

    # 2. Creator questions: "who is your creator", "who created you", "who made you", "who developed you", "who created aarohan"
    creator_patterns = [
        r"^who\s+(is|\'s)\s+your\s+creator$",
        r"^who\s+created\s+you$",
        r"^who\s+made\s+you$",
        r"^who\s+developed\s+you$",
        r"^who\s+built\s+you$",
        r"^who\s+designed\s+you$",
        r"^who\s+is\s+your\s+maker$",
        r"^who\s+is\s+your\s+developer$",
        r"^who\s+is\s+your\s+author$",
        r"^who\s+(is|\'s)\s+the\s+creator\s+of\s+aarohan$",
        r"^who\s+created\s+aarohan$",
        r"^who\s+made\s+aarohan$",
        r"^who\s+developed\s+aarohan$",
        r"^who\s+built\s+aarohan$",
        r"^tell\s+me\s+who\s+created\s+you$",
    ]
    for pat in creator_patterns:
        if re.search(pat, q_clean):
            return "I was created and developed by Datta Sable."

    # 3. Ownership & Creator bio: "who owns aarohan", "who owns you", "who is datta sable"
    owner_patterns = [
        r"^who\s+owns\s+aarohan$",
        r"^who\s+owns\s+you$",
        r"^who\s+(is|\'s)\s+the\s+owner\s+of\s+aarohan$",
        r"^who\s+is\s+your\s+owner$",
        r"^who\s+owns\s+this\s+(project|ai|assistant|software)$",
    ]
    for pat in owner_patterns:
        if re.search(pat, q_clean):
            return "AAROHAN is an independent AI project developed by Datta Sable."

    if re.search(r"^who\s+(is|\'s)\s+datta\s+sable$", q_clean):
        return (
            "Datta Sable is the creator and lead developer of AAROHAN AI, "
            "an independent sovereign AI assistant project (https://github.com/sabledattatray/aarohan-ai)."
        )

    # 4. Third-party creator inquiries (Anthropic, OpenAI, Google, etc.)
    third_party_patterns = [
        r"^are\s+you\s+created\s+by\s+(anthropic|openai|google)(\s+or\s+(anthropic|openai|google))?$",
        r"^are\s+you\s+made\s+by\s+(anthropic|openai|google)(\s+or\s+(anthropic|openai|google))?$",
        r"^were\s+you\s+created\s+by\s+(anthropic|openai|google)(\s+or\s+(anthropic|openai|google))?$",
        r"^were\s+you\s+made\s+by\s+(anthropic|openai|google)(\s+or\s+(anthropic|openai|google))?$",
        r"^(did|have)\s+(anthropic|openai|google)\s+(create|make|develop)\s+you$",
        r"^are\s+you\s+(chatgpt|claude|gemini|gpt4|gpt3)$",
    ]
    for pat in third_party_patterns:
        if re.search(pat, q_clean):
            return (
                "No. AAROHAN is a separate AI application developed by Datta Sable. "
                "It may use third-party AI models or APIs as underlying technology, depending on its configuration."
            )

    # 5. Model inquiries: "what model powers you", "which model powers you", "what model are you using", "which model powers you"
    model_patterns = [
        r"^what\s+model\s+powers\s+you$",
        r"^which\s+model\s+powers\s+you$",
        r"^what\s+model\s+(are\s+you\s+using|is\s+this|is\s+powering\s+you)$",
        r"^what\s+is\s+your\s+(underlying\s+)?model$",
        r"^which\s+model\s+are\s+you$",
        r"^what\s+llm\s+powers\s+you$",
        r"^what\s+architecture\s+powers\s+you$",
        r"^which\s+model\s+is\s+powering\s+you$",
    ]
    for pat in model_patterns:
        if re.search(pat, q_clean):
            if engine == "foundation":
                return (
                    "AAROHAN is currently configured to run with the Qwen2.5-0.5B-Instruct open-weight foundation model "
                    "developed by the Qwen / Alibaba Cloud team, running locally on your hardware."
                )
            elif engine == "scratch":
                return (
                    "AAROHAN is currently configured to run with the custom pre-trained 1.57M parameter from-scratch "
                    "Transformer checkpoint developed within the AAROHAN project by Datta Sable."
                )
            else:
                return "The underlying model information is currently unavailable or unconfigured."

    return None
