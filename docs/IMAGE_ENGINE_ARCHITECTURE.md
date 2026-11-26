# AAROHAN AI — IMAGE ENGINE 1.0 ARCHITECTURE SPECIFICATION

**Project:** AAROHAN AI  
**Author & Developer:** Datta Sable  
**Repository:** https://github.com/sabledattatray/aarohan-ai  
**Hardware Target:** NVIDIA GeForce RTX 2060 (6GB VRAM) / Local CPU Fallback  

---

## 1. Executive Summary

AAROHAN Image Engine 1.0 is a modular, local-first image generation, editing, and visual asset management platform. It allows users to create, manipulate, and explore high-fidelity digital imagery through natural language prompts in the dedicated Image Studio (`/app/images`) or directly within conversational AI Chat (`/app`).

---

## 2. Core Architectural Components

```
                       ┌───────────────────────────────────────────────┐
                       │          Client User Interfaces               │
                       │  - Image Studio (/app/images)                 │
                       │  - Conversational AI Chat (/app)              │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │          AAROHAN FastAPI Application          │
                       │             (aarohan/api.py)                  │
                       │  - /api/images/generate                       │
                       │  - /api/images/edit                           │
                       │  - /api/images/enhance-prompt                 │
                       │  - /api/images/library                        │
                       │  - /api/images/health & /models               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │          ImageStudioManager                   │
                       │       (aarohan/generation/images/)            │
                       └───────┬───────────────┬───────────────┬───────┘
                               │               │               │
            ┌──────────────────┘               │               └──────────────────┐
            ▼                                  ▼                                  ▼
┌────────────────────────┐         ┌────────────────────────┐         ┌────────────────────────┐
│   PromptEnhancer       │         │      ImageEditor       │         │     ImageLibraryDB     │
│ - Category analysis    │         │ - Exposure & contrast  │         │ - SQLite persistence   │
│ - Volumetric lighting  │         │ - Palette & tint shifts│         │ - Full metadata & tags │
│ - Camera perspectives  │         │ - Background blur      │         │ - Favorites & search   │
│ - Hinglish/informal map│         │ - Aspect ratio crops   │         │ - Lineage tracking     │
└────────────────────────┘         │ - Artistic variations  │         └────────────────────────┘
                                   └────────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │      ImageGenerationProvider Interface        │
                       └───────┬───────────────────────────────┬───────┘
                               │                               │
            ┌──────────────────┘                               └──────────────────┐
            ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│        ComfyUIProvider        │                                     │      LocalNeuralProvider      │
│ - REST API (:8188)            │                                     │ - NVIDIA RTX 2060 (6GB)       │
│ - CheckpointLoaderSimple      │                                     │ - Procedural/Neural Engine    │
│ - KSampler workflow JSON      │                                     │ - FP16 / Attention Slicing    │
│ - Asynchronous queue polling  │                                     │ - Automatic CPU Offload       │
│ - Interruption & cancellation │                                     │ - Zero Crash VRAM Guard       │
└───────────────────────────────┘                                     └───────────────────────────────┘
```

---

## 3. Provider Architecture (`ImageGenerationProvider`)

Every backend inherits from `ImageGenerationProvider` (`aarohan/generation/images/provider.py`):

1. **`health_check() -> ProviderHealth`**: Queries hardware VRAM, device identity, and queue occupancy.
2. **`list_models() -> list[ImageModelInfo]`**: Discovers installed checkpoints, file sizes, licenses, and compatibility.
3. **`generate(params: GenerationParams) -> ImageJob`**: Submits text-to-image parameters (prompt, enhanced prompt, seed, dimensions, steps).
4. **`edit(params: EditParams) -> ImageJob`**: Modifies an existing visual asset with lineage tracking.
5. **`get_job_status(job_id) -> Optional[ImageJob]`**: Polls execution progress (0.0 to 1.0).
6. **`cancel_job(job_id) -> bool`**: Interrupts in-flight execution.
7. **`retrieve_result(job_id) -> Optional[dict]`**: Retrieves resulting file paths and URLs.

---

## 4. Hardware Optimization for RTX 2060 (6GB VRAM)

To prevent CUDA Out-of-Memory (OOM) errors and allow simultaneous execution alongside open-weights language models (e.g., Qwen2.5-0.5B / 1.57M Transformer):
- **Sequential Model Execution**: Image generation runs in isolated memory blocks.
- **Attention Slicing & CPU Offload**: For heavy diffusion pipelines, weights are loaded sequentially with FP16 precision.
- **Fail-Safe Neural Synthesizer**: If external large weights are not downloaded or GPU memory is saturated, the built-in `LocalNeuralProvider` delivers authentic multi-layer compositions with zero crash risk.
