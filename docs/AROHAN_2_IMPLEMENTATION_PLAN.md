# AAROHAN AI 2.0 — Complete Implementation Plan

**Project:** AAROHAN AI  
**Owner:** Datta Sable  
**Target:** Modular, local-first personal AI workspace  
**Platform:** FastAPI backend + Single-Page Unified Studio UI + Local RTX 2060 6GB GPU  
**Repository:** https://github.com/sabledattatray/aarohan-ai  
**Date:** November 2026

---

## 1. Executive Summary & Architectural Charter

AAROHAN AI 2.0 evolves from an experimental foundation model runtime into an enterprise-grade, local-first multimodal AI platform. The core architecture strictly adheres to three non-negotiables:
1. **Preservation of Core Identity**: Retain the custom 1.57M pre-trained Transformer, Qwen2.5-0.5B-Instruct local foundation engine, six-layer Cognitive Memory, SQLite Knowledge Graph, and Dataset Studio.
2. **Local-First & Hardware-Aware**: Zero dependency on external commercial cloud APIs for standard workflows. Fully optimized for the local NVIDIA RTX 2060 6GB VRAM constraint via lazy loading, memory offloading, and single-model VRAM residency.
3. **Zero-Mock Policy**: Every UI view, API route, tool orchestration sequence, dataset export, chart render, and research summary must be backed by authentic, executable code with automated test coverage.

---

## 2. Phased Development Roadmap

The transformation follows a 14-phase controlled engineering execution plan:

| Phase | Module / Target | Key Deliverables | Status |
|---|---|---|---|
| **Phase 0** | Repository Audit & Baseline | Full dependency audit, 74/74 passing test baseline, environment profiling | **COMPLETE** |
| **Phase 1** | Frontend Shell & Navigation | Unified 13-workspace sidebar, top nav with active model telemetry, routing (`/app/*`), persistent state | **IN PROGRESS** |
| **Phase 2** | Intelligent AI Chat 2.0 | Multi-turn contextual resolution, technical typo protection, Hinglish/Indian English normalization, math typesetting | **COMPLETE (P0)** |
| **Phase 3** | Model Manager & Tool Orchestration | Unified Tool Registry (`aarohan/tools/`), dynamic model switcher (Qwen2.5 vs. Custom Checkpoint), GPU VRAM safeguards | **PHASE 3** |
| **Phase 4** | Cognitive Memory & Knowledge Graph | 6-layer memory explorer, interactive SVG/Canvas graph viewer, contradiction detector, privacy filters | **PHASE 4** |
| **Phase 5** | Deep Research Module | Multi-tier research (Quick, Standard, Deep), DuckDuckGo/local crawler fallback, clickable citations, injection defense | **PHASE 5** |
| **Phase 6** | Dataset Studio 2.0 | Natural-language dataset generation, 15+ business templates, multi-sheet XLSX (openpyxl), CSV, JSON, SQL | **COMPLETE (P3)** |
| **Phase 7** | Data Analytics & Charts | DuckDB/Pandas SQL query engine, statistical summaries, Matplotlib/Plotly chart rendering and PNG download | **PHASE 7** |
| **Phase 8** | Document Intelligence | Multi-format parser (PDF via pypdf, DOCX via python-docx, XLSX, CSV, TXT), page-cited chunking, RAG QA | **PHASE 8** |
| **Phase 9** | Image Studio & Vision | ComfyUI/Diffusers local pipeline interface, aspect ratio presets, generation history, prompt enhancer | **PHASE 9** |
| **Phase 10** | Code Assistant & Execution Sandbox | Multi-language code assistant (Python, SQL, DAX, JS, PowerShell), isolated subprocess sandbox with limits | **PHASE 10** |
| **Phase 11** | Learning Center & Continual Tuning | Knowledge gap detector, autonomous learning queue, replay buffer fine-tuning, safety gating & rollback | **PHASE 11** |
| **Phase 12** | System Monitor & Settings | Real-time NVML/nvidia-smi GPU telemetry, CPU/RAM psutil stats, persistent user settings, credential vault | **PHASE 12** |
| **Phase 13** | Security, Hardening & End-to-End QA | Prompt injection mitigation, PII redaction, automated test expansion, performance benchmarks | **PHASE 13** |

---

## 3. Detailed Workspace Navigation Hierarchy

The application interface is structured into three clear tiers accessible via the responsive collapsible left sidebar:

### Tier 1: Workspace
1. **AI Chat** (`/app` or `#chat`): Multimodal conversational canvas with streaming, math rendering, copy buttons, feedback.
2. **Deep Research** (`/app/research` or `#research`): Structured topic investigation, multi-source synthesis, and verified citations.
3. **Dataset Studio** (`/app/datasets` or `#datasets`): Natural language synthetic data generation for CSV, XLSX, JSON, and SQL.
4. **Image Studio** (`/app/images` or `#images`): Local neural image generation with aspect ratio controls and gallery.
5. **Document Intelligence** (`/app/documents` or `#documents`): Multi-document ingestion, table extraction, and page-specific Q&A.
6. **Data Analytics** (`/app/analytics` or `#analytics`): Instant tabular profiling, DuckDB query execution, and chart generation.
7. **Code Assistant** (`/app/code` or `#code`): Multi-language generation, DAX/SQL formulation, and sandboxed test execution.

### Tier 2: Intelligence
8. **Cognitive Memory** (`/app/memory` or `#memory`): Visual 6-layer memory inspector (working, episodic, semantic, procedural, graph, meta-cognitive).
9. **Knowledge Graph** (`/app/knowledge-graph` or `#knowledge-graph`): Interactive entity-relation network, contradiction detector, and pathfinder.
10. **Learning Center** (`/app/learning` or `#learning`): Autonomous learning tasks, candidate datasets, training experiments, and model rollback.

### Tier 3: System
11. **Model Manager** (`/app/models` or `#models`): Quantization specs, VRAM allocation, dynamic engine loading, diagnostic logs.
12. **System Monitor** (`/app/system` or `#system`): Live GPU/CPU utilization, temperature, memory pressure, token latency meters.
13. **Settings** (`/app/settings` or `#settings`): Theme, privacy boundaries, search provider keys, execution policies.

---

## 4. Hardware Optimization Guidelines (RTX 2060 6GB)

- **VRAM Budget Allocation**:
  - Qwen2.5-0.5B-Instruct: ~1.2 GB VRAM in float16.
  - Aarohan 1.57M Custom Checkpoint: ~25 MB VRAM / CPU.
  - Image/Diffusion models (when active): ~3.5 to 4.5 GB VRAM.
  - Strict Rule: Unload or offload foundation LLM before initializing heavy image generation pipelines to avoid CUDA Out of Memory (OOM).
- **Graceful Fallbacks**:
  - Automatic CPU offloading if GPU utilization exceeds 90% or VRAM < 500 MB.
  - DuckDB/Pandas computation runs on system CPU with thread pools, preserving GPU resources for neural inference.

---

## 5. Verification Protocol

Every phase must satisfy:
1. Unit and regression tests passing (`pytest tests/`).
2. API endpoint responses return HTTP 200 with typed JSON schemas.
3. UI interacts seamlessly without JavaScript runtime errors or broken layouts.
4. Changes committed in logical units and synchronized to GitHub `master` and `main`.
