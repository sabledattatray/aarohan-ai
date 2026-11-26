"""Custom Transformer Engine implementing BaseInferenceEngine for AAROHANLM."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

from aarohan.chat_template import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    validate_dialogue,
)
from aarohan.inference.base import (
    BaseInferenceEngine,
    GenerationResult,
    ModelCapability,
    ModelInfo,
    ModelType,
)
from aarohan.model import AAROHANConfig, AAROHANLM
from aarohan.tokenizer import load_tokenizer, sha256_file

logger = logging.getLogger(__name__)


class CustomTransformerInferenceEngine(BaseInferenceEngine):
    """Independently trained custom Causal Transformer inference provider."""

    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AAROHANLM] = None
        self.tokenizer: Any = None
        self.config: Optional[AAROHANConfig] = None
        self._load_lock = threading.Lock()
        self._parameter_count: int = 0

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        with self._load_lock:
            if self.is_loaded():
                return
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            if not self.tokenizer_path.is_file():
                raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

            logger.info("Loading custom checkpoint %s on %s...", self.checkpoint_path.name, self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

            expected_tok_hash = checkpoint.get("tokenizer_sha256")
            actual_tok_hash = sha256_file(self.tokenizer_path)
            if expected_tok_hash and expected_tok_hash != actual_tok_hash:
                raise ValueError("Tokenizer SHA-256 hash mismatch with checkpoint")

            self.tokenizer = load_tokenizer(self.tokenizer_path)
            self.config = AAROHANConfig(**checkpoint["config"])
            self.model = AAROHANLM(self.config)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

            self._parameter_count = sum(p.numel() for p in self.model.parameters())
            logger.info("Custom model loaded with %d parameters on %s", self._parameter_count, self.device)

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        validate_dialogue(messages)
        prompt_str = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "system":
                prompt_str += f"{SYSTEM_TOKEN}\n{content}\n"
            elif role == "user":
                prompt_str += f"{USER_TOKEN}\n{content}\n"
            elif role == "assistant":
                prompt_str += f"{ASSISTANT_TOKEN}\n{content}{EOS_TOKEN}\n"

        prompt_str += f"{ASSISTANT_TOKEN}\n"
        return prompt_str

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        start_time = time.perf_counter()
        self.ensure_loaded()
        assert self.model is not None and self.tokenizer is not None and self.config is not None

        prompt_str = self._format_prompt(prompt, system_prompt, history)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False).ids
        if not prompt_ids:
            raise ValueError("Prompt produced no token IDs")

        eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_tokens: List[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length :]
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / max(temperature, 1e-5)

                top_k = 20
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_id = int(next_token[0, 0].item())

                if next_id == eos_id:
                    break

                generated_tokens.append(next_id)
                token_ids = torch.cat((token_ids, next_token), dim=1)

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - start_time) * 1000.0
        warning = (
            "This response was generated by an independently trained experimental SFT model checkpoint. "
            "Output may be limited in reasoning capability."
        )

        return GenerationResult(
            text=response_text,
            generated_tokens=len(generated_tokens),
            prompt_tokens=len(prompt_ids),
            model_id=f"aarohan_custom_{self.checkpoint_path.stem}",
            device=self.device,
            finish_reason="stop",
            latency_ms=latency,
            warning=warning,
        )

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        self.ensure_loaded()
        assert self.model is not None and self.tokenizer is not None and self.config is not None

        prompt_str = self._format_prompt(prompt, system_prompt, history)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False).ids
        eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length :]
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / max(temperature, 1e-5)

                top_k = 20
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_id = int(next_token[0, 0].item())

                if next_id == eos_id:
                    break

                token_ids = torch.cat((token_ids, next_token), dim=1)
                token_text = self.tokenizer.decode([next_id], skip_special_tokens=True)
                yield token_text

    def get_model_info(self) -> ModelInfo:
        ctx_len = self.config.context_length if self.config else 128
        return ModelInfo(
            model_id=f"aarohan_custom_{self.checkpoint_path.stem}",
            model_type=ModelType.CUSTOM_TRANSFORMER,
            parameter_count=self._parameter_count or 1_570_000,
            context_length=ctx_len,
            device=self.device,
            dtype="float32",
            is_loaded=self.is_loaded(),
            capabilities=self.available_capabilities(),
            metadata={
                "architecture": "Causal Transformer (RoPE, SwiGLU, RMSNorm)",
                "checkpoint": str(self.checkpoint_path),
                "tokenizer": str(self.tokenizer_path),
            },
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_loaded() else "ready_to_load",
            "model_id": f"aarohan_custom_{self.checkpoint_path.stem}",
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "tokenizer_exists": self.tokenizer_path.is_file(),
            "device": self.device,
            "is_loaded": self.is_loaded(),
            "parameter_count": self._parameter_count,
        }

    def unload(self) -> bool:
        with self._load_lock:
            if not self.is_loaded():
                return True
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded custom model %s", self.checkpoint_path.name)
            return True

    def available_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
            ModelCapability.CHAT_TEMPLATE,
        ]
