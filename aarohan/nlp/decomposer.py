"""AAROHAN AI - Multi-Question Task Decomposition & Neuro-Symbolic Routing Layer.

Decomposes complex, multi-question or multi-part compound queries into discrete sub-tasks,
routes each sub-task to its optimal deterministic engine (Rule Evaluator, Math Verifier,
Identity Resolver, Capability Engine, Scientific Grounding) or local neural foundation model,
and recombines verified sub-results into a coherent, truthful composite response.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional


class MultiTaskDecomposer:
    """Decomposes compound queries and routes subtasks to deterministic or neural handlers."""

    def __init__(
        self,
        learning_engine: Any = None,
        math_verifier: Any = None,
        foundation_engine: Any = None,
    ):
        self.learning_engine = learning_engine
        self.math_verifier = math_verifier
        self.foundation_engine = foundation_engine

    def split_compound_query(self, query: str) -> List[str]:
        """Splits multi-question or compound prompts into independent sub-queries.

        Supports:
        - Numbered lines (1. ... 2. ...)
        - Lettered lines (a. ... b. ...)
        - Bullet lines (- ... * ...)
        - Inline numbered lists (1. ... 2. ...)
        - Multi-question queries separated by question marks and conjunctions (? Also, / ? And)
        """
        q = query.strip()
        lines = [line.strip() for line in q.splitlines() if line.strip()]

        # 1. Multi-line lists with bullet or numbering
        if len(lines) >= 2 and all(
            re.match(r"^(?:#?\d+[\.\)\-:]|[a-zA-Z][\.\)\-:]|[-*•])\s+", line) for line in lines
        ):
            return [
                re.sub(r"^(?:#?\d+[\.\)\-:]|[a-zA-Z][\.\)\-:]|[-*•])\s*", "", line).strip()
                for line in lines
            ]

        # 2. In-line numbered items: "1. What is X? 2. What is Y?"
        if re.search(r"\b1[\.\)]\s+.*\b2[\.\)]\s+", q):
            parts = re.split(r"\b\d+[\.\)]\s+", q)
            cleaned = [p.strip() for p in parts if p.strip()]
            if len(cleaned) >= 2:
                return cleaned

        # 3. Multi-question text separated by '?' and conjunctions or newlines
        if "?" in q:
            # Match ? followed by conjunctions or newlines
            q_parts = [
                re.sub(r"^[,\s\-\.\?]+", "", p).strip()
                for p in re.split(r"\?\s*(?:also|and|next|\n)+\s*", q, flags=re.IGNORECASE)
                if p.strip()
            ]
            if len(q_parts) >= 2:
                return [p if p.endswith("?") else p + "?" for p in q_parts]

        return [q]

    def extract_math_expression(self, sub_query: str) -> Optional[str]:
        """Extracts arithmetic expression if sub-query is an arithmetic inquiry."""
        q = sub_query.strip()
        m = re.search(
            r"(?:what\s+is\s+|calculate\s+|evaluate\s+|compute\s+|solve\s+)?([0-9\.\(\)\s\+\-\*\/\^\%]+[0-9\.\)])(?:\s*\?)?",
            q,
            re.IGNORECASE,
        )
        if m:
            expr = m.group(1).strip()
            # Ensure it has an operator and at least one digit
            if any(op in expr for op in ["+", "-", "*", "/", "^", "%"]) and re.search(r"\d", expr):
                return expr
        return None

    def execute_subtask(
        self,
        sub_query: str,
        *,
        engine: str = "foundation",
        user_scope: str = "default_user",
        identity_resolver_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        capability_resolver_fn: Optional[Callable[[str], Optional[str]]] = None,
        science_resolver_fn: Optional[Callable[[str], Optional[Any]]] = None,
        neural_generator_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Executes a single subtask through the deterministic hierarchy, falling back to neural."""
        sq = sub_query.strip()

        # 1. Deterministic Rule Engine
        if self.learning_engine:
            rule_res = self.learning_engine.evaluate_rule_query(sq, user_scope=user_scope)
            if rule_res and rule_res.get("status") in ("EVALUATED", "UNKNOWN_RULE", "UNKNOWN_STATUS"):
                return {
                    "query": sq,
                    "response": rule_res["response"],
                    "source": "aarohan_learning_engine",
                    "model_invoked": False,
                    "generated_tokens": 0,
                    "status": rule_res["status"],
                }

        # 2. Deterministic Identity Resolver
        if identity_resolver_fn:
            ident_res = identity_resolver_fn(sq, engine)
            if ident_res:
                return {
                    "query": sq,
                    "response": ident_res,
                    "source": "aarohan_identity_resolver",
                    "model_invoked": False,
                    "generated_tokens": 0,
                    "status": "RESOLVED",
                }

        # 3. Deterministic Math Verifier
        math_expr = self.extract_math_expression(sq)
        if math_expr and self.math_verifier:
            math_res = self.math_verifier.evaluate_and_verify(math_expr)
            if math_res.is_verified and math_res.primary_result is not None:
                resp_text = (
                    f"**Expression:** `{math_res.expression}`\n"
                    f"- **Result:** **{math_res.primary_result}**\n"
                    f"- **Verification:** Independently verified via {math_res.independent_check_method}\n"
                    f"- **Status:** `VALIDATED` ✅"
                )
                return {
                    "query": sq,
                    "response": resp_text,
                    "source": "aarohan_math_verifier",
                    "model_invoked": False,
                    "generated_tokens": 0,
                    "status": "VALIDATED",
                }

        # 4. Capability Engine
        if capability_resolver_fn:
            cap_res = capability_resolver_fn(sq)
            if cap_res:
                return {
                    "query": sq,
                    "response": cap_res,
                    "source": "aarohan_capabilities_engine",
                    "model_invoked": False,
                    "generated_tokens": 0,
                    "status": "RESOLVED",
                }

        # 5. Scientific Grounding Engine
        if science_resolver_fn:
            sci_res = science_resolver_fn(sq)
            if sci_res:
                return {
                    "query": sq,
                    "response": sci_res.detailed_markdown,
                    "source": "aarohan_scientific_grounding",
                    "model_invoked": False,
                    "generated_tokens": 0,
                    "status": "RESOLVED",
                }

        # 6. Local Neural Foundation Engine (Fallback for creative, conversational, or general knowledge)
        if neural_generator_fn:
            try:
                gen_res = neural_generator_fn(sq)
                return {
                    "query": sq,
                    "response": gen_res.get("response", ""),
                    "source": gen_res.get("model_name", "foundation_engine"),
                    "model_invoked": True,
                    "generated_tokens": gen_res.get("generated_tokens", 0),
                    "status": "GENERATED",
                }
            except Exception as e:
                return {
                    "query": sq,
                    "response": f"*(Neural generation error for this subtask: {e})*",
                    "source": "neural_error",
                    "model_invoked": True,
                    "generated_tokens": 0,
                    "status": "ERROR",
                }

        return {
            "query": sq,
            "response": "*(No handler available for subtask)*",
            "source": "unhandled",
            "model_invoked": False,
            "generated_tokens": 0,
            "status": "UNHANDLED",
        }

    def decompose_and_process(
        self,
        query: str,
        *,
        engine: str = "foundation",
        user_scope: str = "default_user",
        identity_resolver_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        capability_resolver_fn: Optional[Callable[[str], Optional[str]]] = None,
        science_resolver_fn: Optional[Callable[[str], Optional[Any]]] = None,
        neural_generator_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Decomposes compound query and processes each subtask independently."""
        sub_queries = self.split_compound_query(query)
        if len(sub_queries) <= 1:
            return {"is_compound": False, "subtasks": []}

        subtask_results = []
        any_model_invoked = False
        total_tokens = 0
        sources = []

        for sq in sub_queries:
            res = self.execute_subtask(
                sq,
                engine=engine,
                user_scope=user_scope,
                identity_resolver_fn=identity_resolver_fn,
                capability_resolver_fn=capability_resolver_fn,
                science_resolver_fn=science_resolver_fn,
                neural_generator_fn=neural_generator_fn,
            )
            subtask_results.append(res)
            if res.get("model_invoked"):
                any_model_invoked = True
                total_tokens += res.get("generated_tokens", 0)
            sources.append(res.get("source", "unknown"))

        # Assemble clean, structured composite markdown response
        lines = [
            f"### 📋 Multi-Task Decomposed & Verified Response\n\n"
            f"> Executed **{len(subtask_results)} independent sub-tasks** via neuro-symbolic routing.\n"
        ]
        for i, st in enumerate(subtask_results, 1):
            src_label = st.get("source", "engine")
            inv_label = "Local Neural Weights" if st.get("model_invoked") else "Deterministic Grounding"
            lines.append(f"#### {i}. {st['query']}\n")
            lines.append(st["response"].strip() + "\n")
            lines.append(f"*(Resolution: `{src_label}` | Mode: {inv_label})*\n")

        composite_text = "\n".join(lines).strip()

        return {
            "is_compound": True,
            "subtasks_count": len(subtask_results),
            "response": composite_text,
            "model_invoked": any_model_invoked,
            "generated_tokens": total_tokens,
            "sources": sources,
            "subtasks": subtask_results,
        }
