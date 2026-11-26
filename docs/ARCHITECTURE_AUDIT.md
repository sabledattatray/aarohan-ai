# Aarohan 2.0 — Architecture Audit & Repository Baseline

**Audit Date:** November 2026  
**Auditor:** Principal AI Architect (Antigravity)  
**Project:** Aarohan AI  
**Owner:** Datta Sable  
**Target:** Transformation into Aarohan 2.0 Complete Cognitive AI Assistant  
**Baseline Test Status:** 66/66 Passing Tests (`pytest`)

---

## 1. Executive Summary

A comprehensive audit was performed across all directories and subsystems in the Aarohan repository (`aarohan/`, `scripts/`, `tests/`, `docs/`, `data/`).

The codebase possesses an exceptionally solid foundation:
- **Causal Transformer Model**: Working native PyTorch architecture (RoPE, SwiGLU, RMSNorm) with training loops, SFT, checkpoint management, and validation.
- **Cognitive Memory Architecture**: Working 6-layer memory model with SQLite storage, vector search (`sqlite-vec`), full-text search, and knowledge graph relations.
- **Foundation Engine**: High-speed local foundation model (`Qwen2.5-0.5B-Instruct` CUDA FP16) with streaming SSE.
- **ChatGPT-Style Web Studio**: Pitch-black responsive UI on FastAPI with multi-turn chat and controls.

However, to elevate Aarohan into the **Aarohan 2.0 Complete Cognitive Assistant** as outlined in the Master Engineering Transformation PRD, substantial enhancements are required in:
1. **Language Understanding Engine (`aarohan/nlp/`)**: Intelligent text normalization, spelling/typo resilience, and multi-intent classification.
2. **Reasoning & Tool Orchestration (`aarohan/agent/`)**: Unified tool registry, plan generation, and intelligent dispatch.
3. **Dataset Generation Studio (`aarohan/generation/datasets/`)**: Synthetic CSV/XLSX/JSON generator with realistic distributions.
4. **Data Analytics & Code Engine (`aarohan/analytics/`, `aarohan/coding/`)**: Real computation with Pandas/DuckDB and safe sandboxed execution.
5. **Web Research (`aarohan/research/`)**: Live web research, page extraction, and verified citations.

---

## 2. Granular Component Audit Matrix

| Component | Status | Working Features | Gaps & Technical Debt | Recommended Action |
|:---|:---|:---|:---|:---|
| **Causal Transformer** | IMPLEMENTED | Native RoPE, SwiGLU, RMSNorm, SFT, tiny training loops | Small parameter size (1.57M) limits open-domain general prose | Retain and preserve; keep available via dual-engine routing |
| **Foundation Engine** | IMPLEMENTED | Qwen2.5-0.5B on CUDA FP16, SSE streaming, multi-turn history | System prompt requires stronger proactive behavior | Upgrade prompt and integrate with the NLP intent router |
| **Cognitive Memory** | IMPLEMENTED | 6 memory layers, vector + BM25 search, knowledge graph | Lacks query-driven automatic memory synthesis and decay scoring | Expand with recency/importance scoring and auto-ingest |
| **NLP Engine** | MISSING | Basic regex prompt injection screening | Lacks spelling correction, intent classification, query rewriting | **Phase 1 Priority**: Build `aarohan/nlp/` engine |
| **Orchestrator / Agent**| PARTIAL | Simple direct routing between foundation & scratch | No multi-step planning or tool selection | **Phase 1 Priority**: Build `aarohan/agent/` |
| **Dataset Studio** | MISSING | Mentioned in PRD | No automated generator for CSV/XLSX/JSON | **Phase 4 Priority**: Build synthetic data engine |
| **Code & Analytics** | PARTIAL | Code blocks highlighted in UI | No sandboxed Python/SQL execution or real calculations | **Phase 6 Priority**: Add sandboxed execution engine |
| **Web Research** | PARTIAL | Bounded research engine in `aarohan/research.py` | Lacks live search provider interface and citation extractors | **Phase 3 Priority**: Expand search providers |
| **Web Studio UI** | IMPLEMENTED | Pitch-black ChatGPT aesthetic, dropdown, modals | Lacks dedicated views for Dataset Studio and Analytics | Enhance sidebar with direct sub-modals |

---

## 3. Reusable Assets
- `aarohan/model.py`: Robust, well-tested causal transformer.
- `aarohan/memory.py`: Clean multi-layer memory storage and knowledge graph.
- `aarohan/foundation_engine.py`: High-throughput local CUDA generator.
- `aarohan/api.py`: Production-grade FastAPI ASGI service.
- `tests/`: 66 comprehensive regression tests ensuring zero regressions.

---

## 4. Immediate Roadmap (Phase 1)
- Construct `aarohan/nlp/` with:
  - `normalizer.py`: Spelling correction, Hinglish / colloquial handling, typo resolution without altering original user prompts.
  - `intent.py`: Multi-intent classification across 20+ intents (Chat, Dataset Generation, Code, Research, SQL, Excel, Calculation, etc.).
  - `context.py`: Follow-up resolution and pronoun/entity tracking.
- Wire `aarohan/nlp/` into the chat inference pipeline.
- Verify with unit and integration tests.
