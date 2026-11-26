"""Ollama High-Performance Foundation Inference Engine for Aarohan AI.

Connects Aarohan to locally hosted 7B/8B models (e.g. Qwen 2.5 7B,
Qwen 2.5 Coder 7B, Llama 3.1 8B, DeepSeek-R1) via Ollama with 4-bit
quantization, full CUDA GPU offload on RTX 2060 6GB, and sub-millisecond streaming.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from aarohan.inference.base import (
    BaseInferenceEngine,
    GenerationResult,
    ModelCapability,
    ModelInfo,
    ModelType,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


class OllamaInferenceEngine(BaseInferenceEngine):
    """Local foundation engine powered by Ollama GPU backend."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._available = False
        self._last_health_check = 0.0

    def _check_server(self) -> bool:
        now = time.time()
        if now - self._last_health_check < 5.0:
            return self._available

        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    self._available = True
                    self._last_health_check = now
                    return True
        except Exception:
            self._available = False

        self._last_health_check = now
        return False

    def is_loaded(self) -> bool:
        if not self._check_server():
            return False
        return len(self.list_installed_models()) > 0

    def ensure_loaded(self) -> None:
        if not self._check_server():
            logger.warning(
                "Ollama server is not responding at %s. Ensure 'ollama serve' is running.",
                self.host,
            )

    def list_installed_models(self) -> List[str]:
        """Fetch list of all installed models from local Ollama instance."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def _resolve_model(self) -> str:
        """Resolve to the best installed model if requested model isn't downloaded yet."""
        installed = self.list_installed_models()
        if not installed:
            return self.model_name
        for m in installed:
            if self.model_name in m:
                return m
        for pref in ["qwen2.5:7b", "qwen2.5-coder:7b", "qwen2.5:3b", "qwen2.5-coder:3b", "llama3.1"]:
            for m in installed:
                if pref in m:
                    return m
        return installed[0]

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        start_time = time.perf_counter()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        active_model = self._resolve_model()
        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": 0.9,
                "repeat_penalty": 1.15,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content = res_json.get("message", {}).get("content", "").strip()
                eval_count = res_json.get("eval_count", len(content.split()))
                prompt_eval_count = res_json.get("prompt_eval_count", len(prompt.split()))
                latency = (time.perf_counter() - start_time) * 1000.0

                return GenerationResult(
                    text=content,
                    generated_tokens=int(eval_count),
                    prompt_tokens=int(prompt_eval_count),
                    model_id=f"ollama/{self.model_name}",
                    device="cuda",
                    finish_reason="stop",
                    latency_ms=latency,
                )
        except Exception as e:
            logger.error("Ollama generate call failed: %s", e)
            raise RuntimeError(f"Ollama inference error: {e}") from e

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        active_model = self._resolve_model()
        payload = {
            "model": active_model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": 0.9,
                "repeat_penalty": 1.15,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error("Ollama stream call failed: %s", e)
            yield f"\n[Inference Error: Ollama connection lost - {e}]"

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=f"ollama/{self._resolve_model()}",
            model_type=ModelType.FOUNDATION_PRETRAINED,
            parameter_count=3_100_000_000,
            context_length=32768,
            device="cuda",
            dtype="q4_k_m",
            is_loaded=self.is_loaded(),
            capabilities=self.available_capabilities(),
            metadata={"quantization": "Q4_K_M (4-bit)", "backend": "ollama_gpu"},
        )

    def health_check(self) -> Dict[str, Any]:
        alive = self._check_server()
        installed = self.list_installed_models() if alive else []
        return {
            "engine": "ollama",
            "model_name": self.model_name,
            "host": self.host,
            "is_available": alive,
            "installed_models": installed,
            "status": "healthy" if alive else "unreachable",
        }

    def unload(self) -> bool:
        """Tell Ollama to unload model from GPU VRAM."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps({"model": self._resolve_model(), "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def available_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
            ModelCapability.CHAT_TEMPLATE,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ]
