# Aarohan AI — Live Repository Gap Analysis

**Author:** Aarohan Engineering Team  
**Audit Scope:** Full codebase, API endpoints, frontend UI, model architectures, memory, and automated test suite.

---

## 1. Executive Summary

Aarohan has achieved an operational sovereign AI platform running 100% locally on user hardware (AMD CPU, 16GB RAM, NVIDIA GeForce RTX 2060 6GB CUDA). 

The platform supports:
- Native inference from original checkpoints and local open-weights foundation models (`Qwen/Qwen2.5-0.5B-Instruct`).
- 6-layer cognitive memory with hybrid search (dense embeddings + SQLite FTS5) and Knowledge Graph.
- Pitch-black ChatGPT-style studio user interface with SSE streaming and KaTeX math rendering.
- Intelligent language engine with typo tolerance, entity protection, 21-intent classification, and context resolution.
- Synthetic Dataset Studio exporting to Excel (`.xlsx`), CSV, JSON, and SQL.
- Safety filters for prompt injection, PII redaction, and audit logging.
- 74 automated unit and integration tests passing in under 20 seconds.

However, to realize the full capability of a truly autonomous sovereign AI assistant, several gaps must be closed across prioritized stages.

---

## 2. Priority Gaps & Action Plan

### P0: Intelligent Query Understanding & Context Resolution (Core NLP Engine)
- **Current State:** Implemented in `aarohan/nlp/` with `normalizer.py`, `intent.py`, and `context.py`. Protects technical terminology and normalizes colloquial typos. Resolves anaphoric references across dialogue turns.
- **Identified Gaps:**
  1. Needs broader phrase normalization for multi-word scientific and colloquial patterns.
  2. Intent classification confidence thresholds should be exposed in API telemetry and UI status chips.
  3. Context resolver needs explicit entity memory linking for named topics across turns.
- **Implementation Status:** `HARDENED & TESTED`.

---

### P1: Model Orchestration & Reliable Answer Quality
- **Current State:** Foundation engine (`Qwen2.5-0.5B-Instruct`) and Custom Checkpoint (`aarohan-sft-001.pt` 1.57M parameters) are selectable via UI dropdown.
- **Identified Gaps:**
  1. Grounding: System prompt must continue to enforce zero hallucinations about external providers (Anthropic, OpenAI, etc.).
  2. Model size limitation: The 0.5B model is fast but lightweight; complex multi-step reasoning requires retrieved context injection or a larger quantized checkpoint (e.g. Qwen2.5-1.5B/3B GGUF/AWQ if VRAM permits).
  3. Response streaming error recovery: Resilient fallback from CUDA foundation engine to local checkpoint without dropping connection.
- **Implementation Status:** `NEXT PHASE (P1)`.

---

### P2: Web Research, Real Source Retrieval & Clickable Citations
- **Current State:** Bounded research CLI exists (`scripts/ingest_document.py`, `aarohan/research.py`), but web search is documented rather than connected to live HTTP search providers.
- **Identified Gaps:**
  1. Connect a real offline-first search provider (DuckDuckGo HTML or SearXNG / bounded fetch).
  2. Clickable bracketed citations (`[1]`, `[2]`) in chat that display extracted source URLs, page numbers, and license metadata.
  3. Fact verifier comparing model claims against retrieved source snippets.
- **Implementation Status:** `PLANNED (P2)`.

---

### P3: Dataset Studio & Python Analytics
- **Current State:** `aarohan/generation/datasets/` generates realistic Banking, Sales, E-commerce, and Employee datasets to `.xlsx`, `.csv`, `.json`, and `.sql` with download endpoints.
- **Identified Gaps:**
  1. Pandas integration for on-the-fly profiling (distribution summaries, correlation matrices, null counts).
  2. Matplotlib / Plotly chart generation saving PNG artifacts with inline previews.
- **Implementation Status:** `PARTIALLY IMPLEMENTED (Datasets Complete; Analytics Next)`.

---

### P4: Multimodal Vision & Document Intelligence
- **Current State:** Text model generates SVG diagrams and Python plotting code. PDF and HTML files are ingested via CLI.
- **Identified Gaps:**
  1. Frontend drag-and-drop file upload pill for PDFs, CSVs, and images.
  2. Local vision model plugin architecture (e.g., small vision encoder or OCR pipeline) for image analysis.
- **Implementation Status:** `PLANNED (P4)`.

---

### P5: Sandboxed Code Execution & Continual Learning Improvements
- **Current State:** Code highlighting and copy button in frontend; `aarohan/continual.py` has RegressionGate, RollbackManager, and ReplayBuffer.
- **Identified Gaps:**
  1. Secure Python sandboxed subprocess runner with execution timeouts and output capture.
  2. Automated fine-tuning loop triggering training when Learning Queue tasks accumulate past threshold.
- **Implementation Status:** `PLANNED (P5)`.
