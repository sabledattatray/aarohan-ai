# AAROHAN AI 2.0 — System Architecture Document

**System Version:** 2.0.0  
**Target Environment:** Local Workstation (Windows, NVIDIA RTX 2060 6GB CUDA)  
**Primary Engine:** Dual-Inference Pipeline (Qwen2.5-0.5B-Instruct + 1.57M Custom Transformer)  
**Protocol:** Local-First, Zero-Commercial API Dependency  

---

## 1. High-Level System Architecture

AAROHAN 2.0 is designed as a modular, local-first artificial intelligence platform comprising five decoupled sub-systems:

```
+-----------------------------------------------------------------------------------+
|                           AAROHAN STUDIO FRONTEND                                  |
|   (Responsive Dark UI, 13 Workspaces, KaTeX Math, Code Blocks, Real-time Charts) |
+-----------------------------------------+-----------------------------------------+
                                          | HTTP / SSE Stream
                                          v
+-----------------------------------------------------------------------------------+
|                               FASTAPI API GATEWAY                                 |
|     (/app/* routing, /api/chat/stream, /api/dataset/*, /api/tools/*, /api/memory) |
+-----------------------------------------+-----------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          v                               v                               v
+-----------------------+   +---------------------------+   +-----------------------+
|  NLP & INTENT ENGINE  |   |   UNIFIED TOOL REGISTRY   |   |  COGNITIVE MEMORY & KG|
|  - Normalizer / Typos |   |   - Dataset Generator     |   |  - 6-Layer Memory     |
|  - 21-Intent Router   |   |   - Python Analytics      |   |  - Vector Embeddings  |
|  - Context Resolver   |   |   - Code Sandbox          |   |  - SQLite KG Graph    |
|  - Safety / Injection |   |   - Research / Web Search |   |  - Contradiction Scan |
+-----------+-----------+   +-------------+-------------+   +-----------+-----------+
            |                             |                             |
            +-----------------------------+-----------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            DUAL MODEL INFERENCE RUNTIME                           |
|  [Engine 1] Qwen2.5-0.5B-Instruct (Local HuggingFace, FP16, CUDA 13.2)            |
|  [Engine 2] Aarohan 1.57M Pre-trained Transformer Checkpoint (Custom PyTorch)     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Structure

```
aarohan/
|-- api.py                     # FastAPI entrypoint, unified studio UI renderer, routes
|-- foundation_engine.py       # Qwen2.5 local inference pipeline, streaming generator
|-- model.py                   # Custom pre-trained Transformer architecture (RoPE, SwiGLU)
|-- memory.py                  # 6-layer Cognitive Memory, SQLite vector storage, Graph
|-- research.py                # Autonomous research engine, knowledge gap detector
|-- continual.py               # Replay buffer, continual learning, regression gate
|-- scheduler.py               # Autonomous learning task orchestrator
|-- evaluator.py               # Multi-domain benchmark evaluation engine
|-- safety.py                  # Prompt injection defense, PII masking, audit log
|-- nlp/
|   |-- normalizer.py          # Spelling correction, phonetic and Hinglish normalizer
|   |-- intent.py              # 21-class intent classification engine
|   |-- context.py             # Conversation context and anaphora resolution
|-- generation/
|   |-- datasets/
|   |   |-- generator.py       # 15 business data templates, openpyxl XLSX, CSV, SQL
|-- tools/
|   |-- registry.py            # Unified Tool Registry and orchestrator
|   |-- analytics.py           # DuckDB / Pandas analytics and Matplotlib chart generator
|   |-- sandbox.py             # Sandboxed Python execution environment
```

---

## 3. Database & Relational Storage

The system stores all state locally in SQLite (`data/memory/aarohan.sqlite`) with tables:
- `memories`: `id`, `layer`, `content`, `category`, `importance`, `verification_status`, `source`, `embedding_blob`, `created_at`.
- `kg_entities`: `name`, `entity_type`, `properties`, `created_at`.
- `kg_relations`: `source_entity`, `target_entity`, `relation_type`, `confidence`, `source_reference`.
- `conversations`: `id`, `user_id`, `title`, `pinned`, `created_at`, `updated_at`.
- `messages`: `id`, `conversation_id`, `role`, `content`, `raw_tokens`, `created_at`.
- `research_records`: `id`, `query`, `summary`, `citations`, `depth`, `created_at`.
- `dataset_records`: `id`, `prompt`, `row_count`, `format`, `file_path`, `created_at`.
- `learning_tasks`: `id`, `title`, `gap_type`, `priority`, `status`, `created_at`.
- `training_experiments`: `id`, `checkpoint_name`, `loss`, `eval_score`, `status`.

---

## 4. Hardware Awareness & Resource Management

- **Target GPU**: NVIDIA GeForce RTX 2060 6GB GDDR6.
- **Model Load Policies**:
  - The Qwen2.5-0.5B model occupies ~1.2 GB VRAM.
  - Custom Transformer occupies ~25 MB.
  - Analytics and SQL executions run exclusively on CPU via DuckDB/Pandas, avoiding GPU memory spikes.
  - Dynamic model unload/offload prior to invoking any secondary generative model.

---

## 5. Security & Isolation Architecture

1. **Host Protection**: No arbitrary AI-generated code is executed directly on the host shell. Code execution occurs in an isolated subprocess with strict timeouts, CPU quotas, memory ceilings, and read-only filesystem policies.
2. **Untrusted Data Isolation**: All external research web text and uploaded document contents are scrubbed by `aarohan/safety.py` to prevent prompt injection and indirect system instructions.
3. **Audit Trail**: Every inference, tool execution, and memory modification is timestamped and logged in `artifacts/audit.log`.
