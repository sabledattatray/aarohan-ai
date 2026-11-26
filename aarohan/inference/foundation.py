"""Foundation Model Engine implementing BaseInferenceEngine."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import torch

from aarohan.inference.base import (
    BaseInferenceEngine,
    GenerationResult,
    ModelCapability,
    ModelInfo,
    ModelType,
)

logger = logging.getLogger(__name__)

DEFAULT_FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class FoundationInferenceEngine(BaseInferenceEngine):
    """High-speed local open-weights foundation model inference provider."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_FOUNDATION_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Any = None
        self.model: Any = None
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

            # Check if existing singleton in FoundationEngine already holds the model
            try:
                from aarohan.foundation_engine import FoundationEngine
                fe = FoundationEngine.get_instance(self.model_name_or_path)
                if fe.is_loaded():
                    self.tokenizer = fe.tokenizer
                    self.model = fe.model
                    self.device = fe.device
                    self._parameter_count = sum(p.numel() for p in self.model.parameters())
                    logger.info("Reused active foundation model from FoundationEngine singleton (%d params on %s)", self._parameter_count, self.device)
                    return
            except Exception as e:
                logger.debug("Singleton reuse check: %s", e)

            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading foundation model %s on %s...", self.model_name_or_path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=dtype,
                ).to(self.device)
            except Exception as e:
                if self.device == "cuda":
                    logger.warning("CUDA load failed (%s); falling back to CPU", e)
                    self.device = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        torch_dtype=torch.float32,
                    ).to("cpu")
                else:
                    raise

            self.model.eval()
            self._parameter_count = sum(p.numel() for p in self.model.parameters())
            logger.info("Foundation model loaded with %d parameters on %s", self._parameter_count, self.device)

            # Sync back to FoundationEngine singleton for consistency
            try:
                from aarohan.foundation_engine import FoundationEngine
                fe = FoundationEngine.get_instance(self.model_name_or_path)
                fe.tokenizer = self.tokenizer
                fe.model = self.model
                fe.device = self.device
            except Exception as e:
                logger.debug("Singleton sync back: %s", e)

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
        assert self.tokenizer is not None and self.model is not None

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                do_sample=temperature > 0.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        latency = (time.perf_counter() - start_time) * 1000.0

        return GenerationResult(
            text=response_text,
            generated_tokens=int(len(new_tokens)),
            prompt_tokens=int(prompt_len),
            model_id=self.model_name_or_path,
            device=self.device,
            finish_reason="stop",
            latency_ms=latency,
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
        assert self.tokenizer is not None and self.model is not None
        from transformers import TextIteratorStreamer

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            do_sample=temperature > 0.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_name_or_path,
            model_type=ModelType.FOUNDATION_PRETRAINED,
            parameter_count=self._parameter_count or 490_000_000,
            context_length=32768,
            device=self.device,
            dtype="float16" if self.device == "cuda" else "float32",
            is_loaded=self.is_loaded(),
            capabilities=self.available_capabilities(),
            metadata={"framework": "PyTorch / HuggingFace Transformers", "provider": "local_open_weights"},
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_loaded() else "ready_to_load",
            "model_id": self.model_name_or_path,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
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
            logger.info("Unloaded foundation model %s", self.model_name_or_path)
            return True

    def available_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
            ModelCapability.CHAT_TEMPLATE,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ]
