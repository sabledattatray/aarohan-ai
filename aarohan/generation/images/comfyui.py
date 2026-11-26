"""ComfyUI Provider Adapter for Aarohan Image Studio.

Connects to a local or remote ComfyUI instance (default: http://127.0.0.1:8188)
via its documented REST API. Implements health checks, model discovery,
workflow synthesis, queue monitoring, job cancellation, and image retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

ARTIFACTS_IMG_DIR = Path("artifacts/images")
ARTIFACTS_IMG_DIR.mkdir(parents=True, exist_ok=True)


class ComfyUIProvider(ImageGenerationProvider):
    """Adapter for ComfyUI REST/WebSocket interface."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = (
            endpoint_url
            or os.environ.get("COMFYUI_URL")
            or "http://127.0.0.1:8188"
        ).rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._jobs: dict[str, ImageJob] = {}

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict[str, Any]] = None,
        timeout: float = 3.0,
    ) -> Optional[dict[str, Any]]:
        """Execute HTTP request against ComfyUI instance."""
        url = f"{self.endpoint_url}/{endpoint.lstrip('/')}"
        headers = {"User-Agent": "Aarohan-AI-ImageStudio/1.0"}
        body = None
        if data is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_bytes = resp.read()
                if resp_bytes:
                    return json.loads(resp_bytes.decode("utf-8"))
                return {}
        except Exception as e:
            logger.debug("ComfyUI request failed (%s %s): %s", method, url, e)
            return None

    def health_check(self) -> ProviderHealth:
        """Check if ComfyUI is running and report GPU telemetry."""
        stats = self._request("system_stats", timeout=2.0)
        if not stats or "system" not in stats:
            return ProviderHealth(
                provider_id="comfyui",
                display_name="ComfyUI Local Studio",
                is_available=False,
                endpoint_url=self.endpoint_url,
                error_message=f"ComfyUI offline or unreachable at {self.endpoint_url}. Launch ComfyUI to enable local diffusion models.",
            )

        devices = stats.get("devices", [])
        vram_total = 0.0
        vram_free = 0.0
        device_name = "CPU"
        if devices:
            d = devices[0]
            device_name = d.get("name", "GPU")
            vram_total = d.get("vram_total", 0) / (1024 * 1024)
            vram_free = d.get("vram_free", 0) / (1024 * 1024)

        return ProviderHealth(
            provider_id="comfyui",
            display_name="ComfyUI Local Studio",
            is_available=True,
            endpoint_url=self.endpoint_url,
            device=device_name,
            vram_used_mb=max(0.0, vram_total - vram_free),
            vram_total_mb=vram_total,
            queue_remaining=stats.get("exec_info", {}).get("queue_remaining", 0),
        )

    def list_models(self) -> list[ImageModelInfo]:
        """Discover checkpoints installed in ComfyUI."""
        object_info = self._request("object_info/CheckpointLoaderSimple", timeout=3.0)
        models = []
        if object_info and "CheckpointLoaderSimple" in object_info:
            input_info = object_info["CheckpointLoaderSimple"].get("input", {}).get("required", {})
            ckpt_names = input_info.get("ckpt_name", [[]])[0]
            for name in ckpt_names:
                models.append(
                    ImageModelInfo(
                        model_name=name,
                        model_type="text2img",
                        provider="comfyui",
                        is_installed=True,
                        is_active=True,
                        description=f"Local ComfyUI checkpoint: {name}",
                    )
                )

        if not models:
            # Baseline catalog definitions
            models = [
                ImageModelInfo(
                    model_name="v1-5-pruned-emaonly.safetensors",
                    model_type="text2img",
                    provider="comfyui",
                    file_size_gb=3.97,
                    is_installed=False,
                    hardware_compatibility="NVIDIA RTX 2060 (6GB) / FP16",
                    license_info="CreativeML OpenRAIL-M",
                    description="Stable Diffusion 1.5 baseline checkpoint. Highly optimized for 6GB VRAM.",
                ),
                ImageModelInfo(
                    model_name="sd_xl_base_1.0.safetensors",
                    model_type="text2img",
                    provider="comfyui",
                    file_size_gb=6.94,
                    is_installed=False,
                    hardware_compatibility="NVIDIA RTX 2060 (6GB) with LowVRAM/CPU-offload",
                    license_info="CreativeML OpenRAIL++-M",
                    description="SDXL 1.0 base model for high-resolution 1024x1024 imagery.",
                ),
            ]
        return models

    def _build_txt2img_workflow(self, params: GenerationParams) -> dict[str, Any]:
        """Construct a standard ComfyUI text-to-image workflow prompt."""
        seed = params.seed if params.seed is not None else 42
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": params.guidance_scale,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler_ancestral" if "euler" in params.sampler.lower() else "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": params.steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": params.model_name or "v1-5-pruned-emaonly.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": params.height,
                    "width": params.width,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": params.enhanced_prompt or params.prompt,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": params.negative_prompt,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "Aarohan_ComfyUI",
                    "images": ["8", 0],
                },
            },
        }

    def generate(self, params: GenerationParams) -> ImageJob:
        """Submit text-to-image job to ComfyUI queue."""
        job_id = str(uuid.uuid4())
        workflow = self._build_txt2img_workflow(params)
        payload = {"prompt": workflow, "client_id": self.client_id}

        res = self._request("prompt", method="POST", data=payload)
        if not res or "prompt_id" not in res:
            job = ImageJob(
                job_id=job_id,
                provider_id="comfyui",
                status=JobStatus.FAILED,
                prompt=params.prompt,
                enhanced_prompt=params.enhanced_prompt,
                error_message="Failed to submit workflow to ComfyUI. Instance may be offline or busy.",
                dimensions=f"{params.width}x{params.height}",
                aspect_ratio=params.aspect_ratio,
                model_name=params.model_name,
            )
            self._jobs[job_id] = job
            return job

        prompt_id = res["prompt_id"]
        job = ImageJob(
            job_id=job_id,
            provider_id="comfyui",
            status=JobStatus.PROCESSING,
            prompt=params.prompt,
            enhanced_prompt=params.enhanced_prompt,
            seed=params.seed,
            dimensions=f"{params.width}x{params.height}",
            aspect_ratio=params.aspect_ratio,
            model_name=params.model_name,
            metadata={"comfy_prompt_id": prompt_id},
        )
        self._jobs[job_id] = job
        return job

    def edit(self, params: EditParams) -> ImageJob:
        """Submit image-to-image or inpainting workflow to ComfyUI."""
        job_id = str(uuid.uuid4())
        job = ImageJob(
            job_id=job_id,
            provider_id="comfyui",
            status=JobStatus.FAILED,
            prompt=params.prompt or "image edit",
            error_message="Direct ComfyUI img2img pipeline requires uploaded input node setup.",
        )
        self._jobs[job_id] = job
        return job

    def get_job_status(self, job_id: str) -> Optional[ImageJob]:
        """Check status of job and download outputs when completed."""
        job = self._jobs.get(job_id)
        if not job or job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job

        prompt_id = job.metadata.get("comfy_prompt_id")
        if not prompt_id:
            return job

        history = self._request(f"history/{prompt_id}", timeout=2.0)
        if history and prompt_id in history:
            item = history[prompt_id]
            outputs = item.get("outputs", {})
            for node_id, node_out in outputs.items():
                if "images" in node_out:
                    images = node_out["images"]
                    if images:
                        first_img = images[0]
                        filename = first_img["filename"]
                        subfolder = first_img.get("subfolder", "")
                        img_type = first_img.get("type", "output")

                        # Download to local artifacts directory
                        view_url = f"{self.endpoint_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                        target_path = ARTIFACTS_IMG_DIR / filename
                        try:
                            urllib.request.urlretrieve(view_url, target_path)
                            job.status = JobStatus.COMPLETED
                            job.completed_at = datetime.now().isoformat()
                            job.progress = 1.0
                            job.output_filename = filename
                            job.output_file_path = str(target_path)
                            job.output_url = f"/api/images/view/{filename}"
                            return job
                        except Exception as e:
                            logger.error("Failed to retrieve ComfyUI image: %s", e)

        # Still processing or in queue
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Interrupt active execution."""
        res = self._request("interrupt", method="POST")
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED
            self._jobs[job_id].completed_at = datetime.now().isoformat()
        return res is not None

    def retrieve_result(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self.get_job_status(job_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None
        return {
            "job_id": job.job_id,
            "filename": job.output_filename,
            "file_path": job.output_file_path,
            "url": job.output_url,
            "dimensions": job.dimensions,
            "aspect_ratio": job.aspect_ratio,
        }
