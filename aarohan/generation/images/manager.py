"""Unified Image Studio Manager for Aarohan AI.

Coordinates provider orchestration (ComfyUI vs. Local Neural Studio),
prompt enhancement, image editing, job execution queues, and persistent
SQLite library registration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from aarohan.generation.images.comfyui import ComfyUIProvider
from aarohan.generation.images.editing import ImageEditor
from aarohan.generation.images.enhancer import PromptEnhancer
from aarohan.generation.images.library import ImageLibraryDB
from aarohan.generation.images.local_pipeline import LocalNeuralProvider
from aarohan.generation.images.provider import (
    EditParams,
    GenerationParams,
    ImageGenerationProvider,
    ImageJob,
    ImageModelInfo,
    JobStatus,
    ProviderHealth,
)

logger = logging.getLogger(__name__)


class ImageStudioManager:
    """Singleton coordinator for all image generation, editing, and storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.library_db = ImageLibraryDB(db_path=db_path)
        self.local_provider = LocalNeuralProvider()
        self.comfy_provider = ComfyUIProvider()
        self._active_provider_id = "aarohan_local"

    @property
    def active_provider(self) -> ImageGenerationProvider:
        """Return active provider; auto-fall back to local if ComfyUI is unavailable."""
        if self._active_provider_id == "comfyui":
            health = self.comfy_provider.health_check()
            if health.is_available:
                return self.comfy_provider
            logger.info("ComfyUI unavailable; routing to local neural engine.")
        return self.local_provider

    def set_active_provider(self, provider_id: str) -> bool:
        if provider_id in ("comfyui", "aarohan_local"):
            self._active_provider_id = provider_id
            return True
        return False

    def get_health(self) -> dict[str, Any]:
        """Aggregate health across all installed providers."""
        local_h = self.local_provider.health_check()
        comfy_h = self.comfy_provider.health_check()
        return {
            "status": "operational",
            "active_provider": self._active_provider_id,
            "providers": [
                {
                    "id": local_h.provider_id,
                    "name": local_h.display_name,
                    "available": local_h.is_available,
                    "device": local_h.device,
                    "vram_used_mb": local_h.vram_used_mb,
                    "vram_total_mb": local_h.vram_total_mb,
                },
                {
                    "id": comfy_h.provider_id,
                    "name": comfy_h.display_name,
                    "available": comfy_h.is_available,
                    "endpoint": comfy_h.endpoint_url,
                    "device": comfy_h.device,
                    "vram_used_mb": comfy_h.vram_used_mb,
                    "vram_total_mb": comfy_h.vram_total_mb,
                    "queue": comfy_h.queue_remaining,
                    "error": comfy_h.error_message,
                },
            ],
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List all models across local and ComfyUI catalogs."""
        local_models = self.local_provider.list_models()
        comfy_models = self.comfy_provider.list_models()
        combined = local_models + comfy_models
        return [
            {
                "model_name": m.model_name,
                "model_type": m.model_type,
                "provider": m.provider,
                "file_size_gb": m.file_size_gb,
                "is_installed": m.is_installed,
                "is_active": m.is_active,
                "hardware_compatibility": m.hardware_compatibility,
                "license": m.license_info,
                "description": m.description,
            }
            for m in combined
        ]

    def enhance_prompt(
        self, prompt: str, style: Optional[str] = None, aspect_ratio: Optional[str] = None
    ) -> dict[str, Any]:
        """Enhance natural language prompt."""
        res = PromptEnhancer.enhance(prompt, style_override=style, aspect_ratio=aspect_ratio)
        return {
            "original_prompt": res.original_prompt,
            "enhanced_prompt": res.enhanced_prompt,
            "negative_prompt": res.negative_prompt,
            "detected_category": res.detected_category,
            "detected_style": res.detected_style,
            "suggested_aspect_ratio": res.suggested_aspect_ratio,
            "lighting": res.lighting,
            "composition": res.composition,
        }

    def generate(
        self,
        prompt: str,
        enhanced_prompt: Optional[str] = None,
        aspect_ratio: str = "1:1",
        style: str = "cinematic",
        seed: Optional[int] = None,
        provider_id: Optional[str] = None,
        model_name: str = "standard",
    ) -> dict[str, Any]:
        """Orchestrate text-to-image generation and persist to library."""
        # Auto-enhance if enhanced prompt is not provided
        if not enhanced_prompt:
            enhanced_info = PromptEnhancer.enhance(prompt, style_override=style, aspect_ratio=aspect_ratio)
            final_enhanced = enhanced_info.enhanced_prompt
            final_category = enhanced_info.detected_category
        else:
            final_enhanced = enhanced_prompt
            final_category = PromptEnhancer.detect_category(prompt)

        provider = self.comfy_provider if provider_id == "comfyui" else self.active_provider

        # Dimensions lookup
        dims = LocalNeuralProvider.ASPECT_RATIO_MAP.get(aspect_ratio, (768, 768))
        params = GenerationParams(
            prompt=prompt,
            enhanced_prompt=final_enhanced,
            aspect_ratio=aspect_ratio,
            width=dims[0],
            height=dims[1],
            style=style,
            seed=seed,
            category=final_category,
            model_name=model_name,
        )

        job = provider.generate(params)

        # Record job state
        self.library_db.record_job(
            job_id=job.job_id,
            provider_id=job.provider_id,
            status=job.status.value,
            prompt=job.prompt,
            enhanced_prompt=job.enhanced_prompt,
            progress=job.progress,
            error_message=job.error_message,
            output_filename=job.output_filename,
            output_file_path=job.output_file_path,
        )

        # If completed immediately, persist to image_library table
        if job.status == JobStatus.COMPLETED and job.output_filename:
            lib_record = self.library_db.add_image(
                filename=job.output_filename,
                file_path=job.output_file_path or "",
                prompt=prompt,
                enhanced_prompt=final_enhanced,
                category=final_category,
                style=style,
                aspect_ratio=aspect_ratio,
                dimensions=job.dimensions,
                seed=job.seed,
                provider=job.provider_id,
                model_name=model_name,
            )
            return {
                "job_id": job.job_id,
                "status": "completed",
                "filename": job.output_filename,
                "url": job.output_url,
                "file_path": job.output_file_path,
                "image_base64": job.thumbnail_base64,
                "dimensions": job.dimensions,
                "aspect_ratio": job.aspect_ratio,
                "prompt": prompt,
                "enhanced_prompt": final_enhanced,
                "library_id": lib_record.get("id"),
            }

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "prompt": prompt,
            "enhanced_prompt": final_enhanced,
            "error": job.error_message,
        }

    def edit(
        self,
        source_image: str,
        operation: str,
        prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Orchestrate image manipulation / variation."""
        params = EditParams(
            source_image_path=source_image,
            operation=operation,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            parameters=parameters or {},
        )
        job = self.local_provider.edit(params)

        # Record job
        self.library_db.record_job(
            job_id=job.job_id,
            provider_id=job.provider_id,
            status=job.status.value,
            prompt=job.prompt,
            progress=job.progress,
            error_message=job.error_message,
            output_filename=job.output_filename,
            output_file_path=job.output_file_path,
        )

        if job.status == JobStatus.COMPLETED and job.output_filename:
            lib_record = self.library_db.add_image(
                filename=job.output_filename,
                file_path=job.output_file_path or "",
                prompt=prompt or f"Edit ({operation}) of {source_image}",
                category="edit",
                style=operation,
                aspect_ratio=aspect_ratio or "1:1",
                dimensions=job.dimensions,
                provider="aarohan_editor",
                original_image_id=source_image,
            )
            return {
                "job_id": job.job_id,
                "status": "completed",
                "filename": job.output_filename,
                "url": job.output_url,
                "file_path": job.output_file_path,
                "original_filename": job.metadata.get("original_filename"),
                "operation": operation,
                "library_id": lib_record.get("id"),
            }

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "error": job.error_message,
        }
