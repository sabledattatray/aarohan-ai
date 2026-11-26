"""AAROHAN Foundation Model Engine.

Provides seamless loading, inference, and real-time streaming for local, 100% free,
open-weights foundation models (e.g. Qwen2.5, TinyLlama, SmolLM) running directly
on local GPU (CUDA) or CPU without any commercial API costs or external telemetry.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Generator

import torch

logger = logging.getLogger(__name__)

DEFAULT_FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LORA_ADAPTER = "checkpoints/aarohan-chat-lora"


class FoundationEngine:
    _instance: FoundationEngine | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_FOUNDATION_MODEL,
        device: str | None = None,
        lora_adapter_path: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        if self.lora_adapter_path is None and Path(DEFAULT_LORA_ADAPTER).exists():
            self.lora_adapter_path = DEFAULT_LORA_ADAPTER
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Any = None
        self.model: Any = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        model_name_or_path: str = DEFAULT_FOUNDATION_MODEL,
        lora_adapter_path: str | None = None,
    ) -> FoundationEngine:
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_name_or_path != model_name_or_path
                or (lora_adapter_path is not None and cls._instance.lora_adapter_path != lora_adapter_path)
            ):
                cls._instance = cls(model_name_or_path=model_name_or_path, lora_adapter_path=lora_adapter_path)
            return cls._instance

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        with self._load_lock:
            if self.is_loaded():
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading foundation model %s on %s...", self.model_name_or_path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model_loader = getattr(AutoModelForCausalLM, "from_pretrained")
            base_model = model_loader(
                self.model_name_or_path,
                torch_dtype=dtype,
            ).to(self.device)

            if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
                try:
                    from peft import PeftModel
                    logger.info("Attaching Aarohan LoRA chat adapter from %s...", self.lora_adapter_path)
                    self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
                    logger.info("Aarohan LoRA chat adapter active!")
                except Exception as e:
                    logger.warning("Could not attach LoRA adapter (%s), using base model: %s", self.lora_adapter_path, e)
                    self.model = base_model
            else:
                self.model = base_model

            self.model.eval()
            logger.info("Foundation model %s successfully loaded on %s.", self.model_name_or_path, self.device)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        self.ensure_loaded()
        assert self.tokenizer is not None and self.model is not None

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "response": response_text,
            "generated_tokens": int(len(new_tokens)),
            "model_name": self.model_name_or_path,
            "device": self.device,
        }

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        self.ensure_loaded()
        assert self.tokenizer is not None and self.model is not None
        from transformers import TextIteratorStreamer

        messages: list[dict[str, str]] = []
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
            do_sample=temperature > 0.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
