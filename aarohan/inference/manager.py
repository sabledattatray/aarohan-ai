"""Inference Manager coordinating model loading, routing, and lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from aarohan.inference.base import (
    BaseInferenceEngine,
    GenerationResult,
)
from aarohan.inference.custom import CustomTransformerInferenceEngine
from aarohan.inference.foundation import FoundationInferenceEngine

logger = logging.getLogger(__name__)


class InferenceManager:
    """Coordinates local model providers, lifecycle, and execution routing."""

    def __init__(
        self,
        *,
        default_foundation_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        custom_checkpoint_path: Optional[Path] = None,
        custom_tokenizer_path: Optional[Path] = None,
    ) -> None:
        self._engines: Dict[str, BaseInferenceEngine] = {}

        # 1. Foundation engine (Default chat with LoRA alignment)
        self._foundation_engine = FoundationInferenceEngine(model_name_or_path=default_foundation_model)
        self.register_engine("foundation", self._foundation_engine)
        self.register_engine("qwen", self._foundation_engine)

        # 2. Dedicated Coder engine (High-precision websites, Python, JavaScript, SQL)
        try:
            self._coder_engine = FoundationInferenceEngine(model_name_or_path="Qwen/Qwen2.5-Coder-1.5B-Instruct")
            self.register_engine("coder", self._coder_engine)
            self.register_engine("code", self._coder_engine)
        except Exception as e:
            logger.info("Coder engine deferred until downloaded: %s", e)
            self._coder_engine = self._foundation_engine

        # 2. Custom checkpoint engine (if paths provided)
        if custom_checkpoint_path and custom_tokenizer_path:
            self._custom_engine = CustomTransformerInferenceEngine(
                checkpoint_path=custom_checkpoint_path,
                tokenizer_path=custom_tokenizer_path,
            )
            self.register_engine("scratch", self._custom_engine)
            self.register_engine("custom", self._custom_engine)
        else:
            self._custom_engine = None

    def register_engine(self, name: str, engine: BaseInferenceEngine) -> None:
        self._engines[name.lower()] = engine

    def get_engine(self, name: str = "foundation") -> BaseInferenceEngine:
        normalized = name.lower()
        if normalized in self._engines:
            return self._engines[normalized]
        raise ValueError(f"Unknown engine '{name}'. Available: {list(self._engines.keys())}")

    def list_models(self) -> List[Dict[str, Any]]:
        seen = set()
        models = []
        for name, engine in self._engines.items():
            info = engine.get_model_info()
            if info.model_id not in seen:
                seen.add(info.model_id)
                m_dict = info.to_dict()
                m_dict["alias"] = name
                models.append(m_dict)
        return models

    def health_check_all(self) -> Dict[str, Any]:
        results = {}
        for name, engine in self._engines.items():
            results[name] = engine.health_check()
        return results

    def _resolve_engine_name(self, engine_name: str, prompt: str) -> str:
        """Route to appropriate engine based on task requirements."""
        eng = engine_name.lower()
        if eng == "auto":
            lower_p = prompt.lower()
            code_keywords = ["website", "html", "css", "javascript", "python", "sql", "script", "code", "table", "dataset", "csv", "query"]
            if any(k in lower_p for k in code_keywords) and "coder" in self._engines:
                return "coder"
            return "foundation"
        if eng in ("coder", "code") and "coder" not in self._engines:
            return "foundation"
        return eng if eng in self._engines else "foundation"

    def generate(
        self,
        prompt: str,
        *,
        engine_name: str = "foundation",
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        resolved_engine = self._resolve_engine_name(engine_name, prompt)
        engine = self.get_engine(resolved_engine)
        try:
            return engine.generate(
                prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Generation with engine '%s' failed: %s", resolved_engine, e)
            if resolved_engine in ("coder", "code"):
                logger.info("Falling back to foundation engine...")
                return self._foundation_engine.generate(
                    prompt,
                    system_prompt=system_prompt,
                    history=history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            if resolved_engine in ("foundation", "qwen") and self._custom_engine:
                logger.info("Falling back to custom transformer engine...")
                res = self._custom_engine.generate(
                    prompt,
                    system_prompt=system_prompt,
                    history=history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                res.warning = f"Primary engine '{resolved_engine}' unavailable ({e}). Fallback applied."
                return res
            raise

    def stream(
        self,
        prompt: str,
        *,
        engine_name: str = "foundation",
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        resolved_engine = self._resolve_engine_name(engine_name, prompt)
        try:
            engine = self.get_engine(resolved_engine)
            return engine.stream(
                prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Stream with engine '%s' failed: %s, falling back to foundation", resolved_engine, e)
            return self._foundation_engine.stream(
                prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def unload_all(self) -> None:
        for engine in set(self._engines.values()):
            engine.unload()
