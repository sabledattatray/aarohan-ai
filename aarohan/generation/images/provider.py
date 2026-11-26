"""Modular Image Generation Provider Architecture for Aarohan 2.0.

Defines the core interface and data structures required by all image generation,
editing, and transformation backends (ComfyUI, Local Diffusers, and Neural Synthesizers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ImageModelInfo:
    """Metadata describing an image generation or editing model."""

    model_name: str
    model_type: str  # text2img, img2img, inpaint, upscale
    provider: str
    file_size_gb: float = 0.0
    model_path: str = ""
    is_installed: bool = False
    is_active: bool = False
    hardware_compatibility: str = "RTX 2060 (6GB) / CPU"
    license_info: str = "CreativeML OpenRAIL-M / Apache 2.0"
    description: str = ""


@dataclass
class ProviderHealth:
    """Health and connection status of an image backend provider."""

    provider_id: str
    display_name: str
    is_available: bool
    endpoint_url: str = ""
    device: str = "cpu"
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    queue_remaining: int = 0
    active_model: str = ""
    error_message: Optional[str] = None


@dataclass
class GenerationParams:
    """Input parameters for a text-to-image generation job."""

    prompt: str
    enhanced_prompt: str = ""
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, watermark, text, out of frame, bad anatomy"
    )
    width: int = 768
    height: int = 768
    aspect_ratio: str = "1:1"
    steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    sampler: str = "Euler a"
    style: str = "cinematic"
    category: str = "general"
    model_name: str = "standard"


@dataclass
class EditParams:
    """Input parameters for an image-to-image or image-editing job."""

    source_image_path: str
    operation: str  # img2img, variation, brightness, contrast, background, crop, inpaint
    prompt: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    strength: float = 0.7  # 0.0 to 1.0 transformation strength
    seed: Optional[int] = None
    aspect_ratio: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageJob:
    """Represents an asynchronous image generation or editing task."""

    job_id: str
    provider_id: str
    status: JobStatus
    prompt: str
    enhanced_prompt: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    output_filename: Optional[str] = None
    output_file_path: Optional[str] = None
    output_url: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    seed: Optional[int] = None
    dimensions: str = "768x768"
    aspect_ratio: str = "1:1"
    model_name: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageGenerationProvider(ABC):
    """Abstract Base Class for all Aarohan image generation providers."""

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Verify provider availability and system resources."""
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[ImageModelInfo]:
        """List models supported or hosted by this provider."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, params: GenerationParams) -> ImageJob:
        """Submit and execute a text-to-image generation job."""
        raise NotImplementedError

    @abstractmethod
    def edit(self, params: EditParams) -> ImageJob:
        """Submit and execute an image editing / img2img job."""
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[ImageJob]:
        """Query execution progress and status of a job."""
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued job if supported."""
        raise NotImplementedError

    @abstractmethod
    def retrieve_result(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve output image metadata and file paths."""
        raise NotImplementedError
