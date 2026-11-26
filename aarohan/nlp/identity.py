"""AAROHAN AI - Identity, Creator Attribution, and Model Awareness Engine.

Enforces accurate identity responses:
- Assistant Name: AAROHAN
- Creator: Datta Sable
- User Persona / Owner: Datta Sable (creator & lead engineer)
- Project: AAROHAN AI
- Official GitHub & Website: https://github.com/sabledattatray/aarohan-ai
- Personal Website: https://dattasable.com/
- Underlying Model: Accurately identifies local Qwen2.5 or Custom Transformer checkpoint.
"""

from __future__ import annotations

import re
from typing import Optional


def resolve_identity_query(query: str, engine: str = "foundation") -> Optional[str]:
    """Check if query is asking about identity, creator, ownership, or underlying model.

    Returns the canonical truthful response, or None if the query is not an identity question.
    """
    raw_q = query.strip()
    q = raw_q.lower()

    # -------------------------------------------------------------------------
    # 0. Multi-Question / Compound Identity & Profile Inquiries
    # -------------------------------------------------------------------------
    # Compound profile prompt (e.g. "Tell me everything you currently know about me, Datta Sable...")
    is_compound_profile = bool(
        any(k in q for k in [
            "tell me everything you know",
            "everything you currently know",
            "everything you know about me",
            "everything you know about datta sable",
            "tell me everything about me",
            "tell me everything about datta sable",
            "tell me all about me",
            "tell me all about datta sable",
            "what do you know about me",
            "what all do you know about me",
            "what do you know about datta sable",
        ])
        or (
            ("tell me everything" in q or "everything you know" in q)
            and any(k in q for k in ["datta sable", "about me", "background", "profile"])
        )
    )
    if is_compound_profile:
        return (
            "### 👤 Human User Profile — Datta Sable\n\n"
            "1. **Full Name:** Datta Sable\n"
            "2. **Professional Background:** MIS and Business Intelligence Specialist with extensive BFSI (Banking, Financial Services, and Insurance) and fintech domain expertise; Independent AI Engineer & Systems Architect; Creator and Lead Developer of AAROHAN AI.\n"
            "3. **Technical Skills:** Advanced Excel, SQL Server, Power BI, Power Query, Python, Causal Transformers, Cognitive Memory Architectures, Local Neural Inference & Automated MIS Reporting.\n"
            "4. **Personal Website:** [https://dattasable.com/](https://dattasable.com/)\n"
            "   - **Official Project Repository:** [https://github.com/sabledattatray/aarohan-ai](https://github.com/sabledattatray/aarohan-ai)\n"
            "   - **GitHub Profile:** [https://github.com/sabledattatray](https://github.com/sabledattatray)\n"
            "5. **Who Developed AAROHAN:** AAROHAN was created and developed independently by Datta Sable.\n"
            "6. **Information Not Verified:** Private personal credentials, off-record personal communications, or unprovided private data cannot be verified."
        )

    # User statement confirming/asserting personal website
    if re.search(r"\b(datta\s+sable|creator).*website.*(dattasable\.com|https?://)", q):
        return (
            "Confirmed. Datta Sable's verified personal website is [https://dattasable.com/](https://dattasable.com/), "
            "and the official AAROHAN AI repository is [https://github.com/sabledattatray/aarohan-ai](https://github.com/sabledattatray/aarohan-ai)."
        )

    # Multi-item numbered identity questions (e.g. "1. What is my name? 2. What is your name? ...")
    has_my_name = bool(re.search(r"\b(my\s+name|who\s+am\s+i|and\s+my|and\s+mine)\b", q))
    has_your_name = bool(re.search(r"\b(your\s+name|who\s+are\s+you|and\s+your|and\s+yours)\b", q))
    has_creator = bool(re.search(r"\b(who\s+created\s+you|who\s+developed|who\s+is\s+your\s+creator|creator)\b", q))
    has_website = bool(re.search(r"\b(my\s+website|your\s+website|website)\b", q))
    if sum([has_my_name, has_your_name, has_creator, has_website]) >= 2 and any(ch in raw_q for ch in ["\n", "1.", "2.", "?", ";"]):
        parts = []
        if has_my_name:
            parts.append("- **Your Name (Human User):** Datta Sable, the creator and lead developer of AAROHAN AI.")
        if has_your_name:
            parts.append("- **My Name (AI Assistant):** AAROHAN, an independent sovereign cognitive AI assistant.")
        if has_creator:
            parts.append("- **Creator & Developer:** Created and developed by Datta Sable.")
        if has_website:
            parts.append(
                "- **Websites:**\n"
                "  - Datta Sable's Personal Website: [https://dattasable.com/](https://dattasable.com/)\n"
                "  - AAROHAN AI Official Project Repository: [https://github.com/sabledattatray/aarohan-ai](https://github.com/sabledattatray/aarohan-ai)\n"
                "  - Datta Sable's GitHub Profile: [https://github.com/sabledattatray](https://github.com/sabledattatray)"
            )
        return "### 📋 Identity & System Context\n\n" + "\n".join(parts)

    # -------------------------------------------------------------------------
    # Single-Query Normalization
    # -------------------------------------------------------------------------
    # Strip leading list numbers like "1. ", "1) ", "1 - ", "#1 ", "q1: " BEFORE stripping punctuation
    q = re.sub(r"^\s*(?:#?\d+[\.\)\-:]|[a-zA-Z][\.\)\-:]|[-*•]|q\d*[:\.\s])\s*", "", q)
    # Also handle bare digits at start if followed by word
    q = re.sub(r"^\s*\d+\s+", "", q)
    # Normalize common contractions before stripping punctuation
    q = re.sub(r"\'s\b", " is", q)
    q = re.sub(r"\'re\b", " are", q)
    q = re.sub(r"\'m\b", " am", q)
    # Normalize colloquial contractions, abbreviations, and common typos
    q = re.sub(r"\bwhats\b", "what is", q)
    q = re.sub(r"\bwhos\b", "who is", q)
    q = re.sub(r"\bwheres\b", "where is", q)
    q = re.sub(r"\bur\b", "your", q)
    q = re.sub(r"\bu\b", "you", q)
    q = re.sub(r"\bnam\b", "name", q)
    # Normalize punctuation and whitespace
    q_clean = re.sub(r"[^\w\s]", " ", q)
    q_clean = " ".join(q_clean.split())

    # -------------------------------------------------------------------------
    # 1. User Name & User Identity questions
    # -------------------------------------------------------------------------
    user_name_patterns = [
        r"^what\s+(is|\'s|s)\s+my\s+name$",
        r"^whats\s+my\s+name$",
        r"^tell\s+me\s+my\s+name$",
        r"^who\s+am\s+i$",
        r"^my\s+name$",
        r"^and\s+my$",
        r"^and\s+mine$",
        r"^and\s+my\s+name$",
        r"^and\s+what\s+about\s+my\s+name$",
        r"^what\s+about\s+my\s+name$",
        r"^do\s+you\s+know\s+my\s+name$",
        r"^what\s+do\s+you\s+call\s+me$",
        r"^who\s+is\s+talking\s+to\s+you$",
        r"^who\s+are\s+you\s+talking\s+to$",
        r"^who\s+is\s+speaking\s+with\s+you$",
        r"^who\s+are\s+you\s+speaking\s+with$",
        r"^who\s+is\s+chatting\s+with\s+you$",
        r"^who\s+are\s+you\s+chatting\s+with$",
        r"^who\s+is\s+interacting\s+with\s+you$",
        r"^who\s+is\s+this\s+user$",
        r"^who\s+is\s+the\s+user$",
    ]
    for pat in user_name_patterns:
        if re.search(pat, q_clean):
            if any(k in q_clean for k in ["talking to you", "you talking to", "speaking with you", "chatting with you"]):
                return "I am interacting with Datta Sable, the creator and lead developer of AAROHAN AI."
            return "Your name is Datta Sable, the creator and lead developer of AAROHAN AI."

    # -------------------------------------------------------------------------
    # 2. Assistant Name & Identity questions
    # -------------------------------------------------------------------------
    name_patterns = [
        r"^what\s+(is|\'s|s)\s+your\s+name$",
        r"^whats\s+your\s+name$",
        r"^what\s+is\s+your\s+name\s+please$",
        r"^who\s+are\s+you$",
        r"^what\s+are\s+you\s+called$",
        r"^tell\s+me\s+your\s+name$",
        r"^your\s+name$",
        r"^and\s+your$",
        r"^and\s+yours$",
        r"^and\s+your\s+name$",
        r"^what\s+is\s+aarohan$",
        r"^who\s+is\s+aarohan$",
        r"^who\s+am\s+i\s+(talking|chatting|speaking)\s+(to|with)$",
    ]
    for pat in name_patterns:
        if re.search(pat, q_clean):
            if "who am i" in q_clean:
                return "You are chatting with AAROHAN, an AI assistant created by Datta Sable."
            return "My name is AAROHAN, an AI assistant created by Datta Sable."

    # -------------------------------------------------------------------------
    # 3. Creator questions
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4. User & Creator Background: "what is my professional background", "who is datta sable"
    # -------------------------------------------------------------------------
    user_bg_patterns = [
        r"^what\s+(is|\'s|s)\s+(?:my|your\s+creator(?:\'s)?|datta\s+sable(?:\'s)?)\s+(?:professional\s+)?background$",
        r"^tell\s+me\s+about\s+(?:my|your\s+creator(?:\'s)?|datta\s+sable(?:\'s)?)\s+(?:professional\s+)?background$",
        r"^what\s+is\s+(?:my|your\s+creator(?:\'s)?|datta\s+sable(?:\'s)?)\s+profession$",
        r"^what\s+do\s+i\s+do$",
        r"^who\s+is\s+datta\s+sable$",
        r"^what\s+is\s+datta\s+sable\s+(?:professional\s+)?background$",
        r"^tell\s+me\s+(?:more\s+)?about\s+datta\s+sable$",
        r"^tell\s+me\s+(?:more\s+)?about\s+(?:myself|me)$",
        r"^who\s+is\s+your\s+creator\s+and\s+what\s+is\s+his\s+background$",
        r"^datta\s+sable$",
        r"^what\s+do\s+you\s+know\s+about\s+datta\s+sable$",
        r"^what\s+is\s+my\s+(?:job|role|work)$",
        r"^(?:my\s+)?profession$",
        r"^(?:my\s+)?(?:job|role|work)$",
        r"^(?:my\s+)?background$",
    ]
    for pat in user_bg_patterns:
        if re.search(pat, q_clean):
            return (
                "### 👤 Datta Sable — Professional Profile\n\n"
                "**Datta Sable** is an independent AI engineer, systems architect, and **MIS & Business Intelligence specialist** "
                "with extensive **BFSI (Banking, Financial Services, and Insurance) and fintech** domain expertise.\n\n"
                "- **Role:** Creator and Lead Developer of **AAROHAN AI**.\n"
                "- **Personal Website:** [https://dattasable.com/](https://dattasable.com/)\n"
                "- **Official Project Repository:** [https://github.com/sabledattatray/aarohan-ai](https://github.com/sabledattatray/aarohan-ai)\n"
                "- **GitHub Profile:** [https://github.com/sabledattatray](https://github.com/sabledattatray)\n"
                "- **Domain Specializations:** Sovereign local-first AI systems, Causal Transformer architectures, Cognitive Memory & Continual Learning pipelines, Automated MIS Reporting, and Multimodal platforms.\n"
                "- **Core Technical Skills:** Advanced Excel, SQL Server, Power BI, Power Query, Python, PyTorch CUDA, and Data Analytics.\n"
                "- **Corporate Independence:** Independent developer and researcher; not affiliated with Anthropic, OpenAI, or Google."
            )

    # -------------------------------------------------------------------------
    # 4b. Location & Residence questions: "where do i stay", "where do i live"
    # -------------------------------------------------------------------------
    user_location_patterns = [
        r"^(?:where\s+(?:do\s+i|does\s+datta\s+sable|does\s+the\s+creator|does\s+your\s+creator)\s+(?:live|stay|reside|stay\s+at)|where\s+i\s+(?:stay|live)|where\s+you\s+think\s+i\s+stay)$",
        r"^(?:where\s+(?:am\s+i|is\s+datta\s+sable|is\s+the\s+creator)\s+(?:staying|living|based|located|from))$",
        r"^(?:what\s+is\s+(?:my|datta\s+sable(?:\'s)?)\s+(?:location|city|address|residence|place))$",
        r"^(?:my\s+(?:location|address|city|residence))$",
        r"^(?:do\s+you\s+know\s+where\s+i\s+(?:stay|live))$",
    ]
    for pat in user_location_patterns:
        if re.search(pat, q_clean):
            return "According to verified records, you (Datta Sable) stay at Badlapur East."

    assistant_location_patterns = [
        r"^(?:where\s+do\s+you\s+(?:live|stay|reside)|where\s+are\s+you\s+(?:located|staying|living|based|hosted|from)|what\s+is\s+your\s+location)$",
        r"^(?:where\s+does\s+aarohan\s+(?:live|stay|run|reside))$",
    ]
    for pat in assistant_location_patterns:
        if re.search(pat, q_clean):
            return "I am AAROHAN, an independent AI running locally on your hardware. I operate directly on your local workstation."

    # -------------------------------------------------------------------------
    # 5. Website & Repository questions
    # -------------------------------------------------------------------------
    website_patterns = [
        r"^(?:what\s+(?:is|\'s|s)|whats)\s+(?:my|your|the|the\s+official|aarohan|datta\s+sable)\s+website$",
        r"^(?:what\s+(?:is|\'s|s)|whats)\s+(?:the|my|your)\s+(?:url|link)$",
        r"^(?:where\s+(?:is|\'s|s)|wheres)\s+(?:the|my|your)\s+website$",
        r"^what\s+(?:is|\'s|s)\s+the\s+github(\s+repo|\s+repository|\s+link|\s+url)?$",
        r"^whats\s+the\s+github(\s+repo|\s+repository|\s+link|\s+url)?$",
        r"^show\s+me\s+(?:the|my|your)\s+website$",
        r"^tell\s+me\s+(?:the|your|my)\s+website$",
        r"^(?:my|your|the|datta\s+sable)?\s*website\s*(?:url|link)?$",
        r"^my\s+website$",
        r"^your\s+website$",
        r"^and\s+my\s+website$",
        r"^and\s+your\s+website$",
        r"^datta\s+sable\s+website$",
    ]
    for pat in website_patterns:
        if re.search(pat, q_clean):
            return (
                "Datta Sable's personal website:\n"
                "🌐 [https://dattasable.com/](https://dattasable.com/)\n\n"
                "AAROHAN AI official repository & project website:\n"
                "🔗 [https://github.com/sabledattatray/aarohan-ai](https://github.com/sabledattatray/aarohan-ai)\n\n"
                "Datta Sable's official GitHub profile:\n"
                "🔗 [https://github.com/sabledattatray](https://github.com/sabledattatray)"
            )

    # -------------------------------------------------------------------------
    # 6. Ownership & Project nature
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7. Third-party creator inquiries
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 8. Model inquiries
    # -------------------------------------------------------------------------
    model_patterns = [
        r"^(?:what|which)\s+(?:ai\s+|llm\s+|language\s+)?model\s+(?:powers|is\s+powering)\s+you(?:\s+(?:right\s+now|currently|today|at\s+the\s+moment))?$",
        r"^(?:what|which)\s+(?:ai\s+|llm\s+|language\s+)?model\s+(?:are\s+you\s+using|are\s+you\s+running|is\s+this|is\s+running|is\s+active)(?:\s+(?:right\s+now|currently|today))?$",
        r"^(?:what|which)\s+llm\s+(?:are\s+you\s+using|are\s+you\s+running|is\s+this|is\s+running|is\s+active)$",
        r"^(?:what|which)\s+(?:ai\s+|llm\s+|language\s+)?model\s+is\s+running$",
        r"^what\s+model\s+are\s+you\s+running$",
        r"^what\s+model\s+is\s+this$",
        r"^what\s+is\s+your\s+(?:current\s+)?(?:underlying\s+)?(?:ai\s+|llm\s+)?model$",
        r"^which\s+model\s+are\s+you(?:\s+using|\s+running)?$",
        r"^what\s+(?:ai\s+|llm\s+)?architecture\s+powers\s+you$",
    ]
    for pat in model_patterns:
        if re.search(pat, q_clean):
            if engine == "foundation":
                return (
                    "AAROHAN is currently configured to run with the Qwen2.5-0.5B-Instruct open-weight foundation model "
                    "developed by the Qwen / Alibaba Cloud team, running locally on your hardware via CUDA (NVIDIA GeForce RTX 2060)."
                )
            elif engine == "scratch":
                return (
                    "AAROHAN is currently configured to run with the custom pre-trained 1.57M parameter from-scratch "
                    "Transformer checkpoint developed within the AAROHAN project by Datta Sable, running locally on your hardware."
                )
            else:
                return "The underlying model information is currently unavailable or unconfigured."

    return None
