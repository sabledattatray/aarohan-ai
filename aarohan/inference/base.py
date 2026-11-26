"""Unified provider-independent model interface for AAROHAN AI.

Defines the contract that every model provider (Foundation open-weights or
independently trained custom Causal Transformer) must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class ModelCapability(str, Enum):
    TEXT_GENERATION = "TEXT_GENERATION"
    STREAMING = "STREAMING"
    CHAT_TEMPLATE = "CHAT_TEMPLATE"
    CODE_GENERATION = "CODE_GENERATION"
    REASONING = "REASONING"
    INSTRUCTION_FOLLOWING = "INSTRUCTION_FOLLOWING"
    EMBEDDING = "EMBEDDING"


class ModelType(str, Enum):
    FOUNDATION_PRETRAINED = "FOUNDATION_PRETRAINED"
    CUSTOM_TRANSFORMER = "CUSTOM_TRANSFORMER"


@dataclass
class ModelInfo:
    model_id: str
    model_type: ModelType
    parameter_count: int
    context_length: int
    device: str
    dtype: str
    is_loaded: bool
    capabilities: List[ModelCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "device": self.device,
            "dtype": self.dtype,
            "is_loaded": self.is_loaded,
            "capabilities": [c.value for c in self.capabilities],
            "metadata": self.metadata,
        }


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int
    prompt_tokens: int
    model_id: str
    device: str
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "model_id": self.model_id,
            "device": self.device,
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 2),
            "warning": self.warning,
        }


class BaseInferenceEngine(ABC):
    """Abstract base class establishing the unified inference contract."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generates a complete response."""
        ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Yields chunks of text in real-time streaming."""
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Returns structured metadata about model architecture and capabilities."""
        ...

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Verifies runtime health, device placement, and loading state."""
        ...

    @abstractmethod
    def unload(self) -> bool:
        """Frees model weights from VRAM/RAM memory."""
        ...

    @abstractmethod
    def available_capabilities(self) -> List[ModelCapability]:
        """Lists verified operational capabilities."""
        ...
