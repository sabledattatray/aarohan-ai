# AAROHAN AI — REST API REFERENCE

**Platform:** AAROHAN AI 2.0  
**Architect:** Datta Sable (`https://dattasable.com/`)  
**Base URL:** `http://127.0.0.1:8000`  
**Authentication:** Header `Authorization: Bearer <token>` or default local authenticated session.

---

## 1. Inference & Chat Endpoints

### 1.1 `POST /api/chat`
Execute a synchronous chat inference with intent classification, memory retrieval, and grounding.

**Request Body:**
```json
{
  "query": "What is the capital of India?",
  "engine": "foundation",
  "retrieval_mode": "hybrid",
  "memory_limit": 5,
  "max_new_tokens": 128
}
```

**Response Body:**
```json
{
  "response": "The capital of India is New Delhi.",
  "engine": "foundation",
  "knowledge_source": "model_weights",
  "intent": "FACTUAL_QUESTION",
  "retrieved_memory": [],
  "verification": {
    "status": "verified",
    "grounding_score": 1.0,
    "claims_verified": 1
  }
}
```

### 1.2 `POST /api/chat/stream`
Stream tokens via Server-Sent Events (SSE).

**SSE Event Sequence:**
1. `data: {"type": "context", "engine": "foundation", "response_source": "foundation_model", "retrieved_memory": [...]}`
2. `data: {"type": "token", "token": "The"}`
3. `data: {"type": "token", "token": " capital"}`
4. `data: {"type": "done", "engine": "foundation", "checkpoint": "qwen2.5-0.5b"}`
5. Error (if blocked): `data: {"type": "error", "reason": "Input flagged by safety filter."}`

---

## 2. Model Registry & Health Endpoints

### 2.1 `GET /api/models`
Returns list of registered inference engines, hardware residency, and capabilities.

**Response Body:**
```json
{
  "active_engine": "foundation",
  "models": [
    {
      "name": "Qwen/Qwen2.5-0.5B-Instruct",
      "model_type": "Foundation Model",
      "alias": "foundation",
      "device": "cuda",
      "total_parameters": 494032768,
      "loaded": true,
      "capabilities": ["TEXT_GENERATION", "STREAMING", "CODE_GENERATION"]
    },
    {
      "name": "aarohan-release-candidate",
      "model_type": "Custom Neural Transformer",
      "alias": "scratch",
      "device": "cuda",
      "total_parameters": 1572864,
      "loaded": false,
      "capabilities": ["TEXT_GENERATION", "STREAMING"]
    }
  ],
  "health": {
    "foundation": {"is_loaded": true, "device": "cuda"},
    "scratch": {"checkpoint_exists": true, "device": "cuda"}
  }
}
```

---

## 3. Cognitive Memory Endpoints

- `GET /api/memory/search?query=knowledge&limit=25`: Hybrid keyword and semantic memory search.
- `GET /api/memory/layers`: Real-time breakdown of memory records across all 6 cognitive layers.
- `POST /api/memory`: Persist an explicit memory record with importance, source, and TTL.
- `DELETE /api/memory/{id}`: Delete an individual memory record.
- `POST /api/memory/purge-expired`: Trigger garbage collection on expired TTL memories.

---

## 4. Knowledge Graph Endpoints

- `GET /api/graph`: Returns entities and relational triples.
- `POST /api/graph/entities`: Register a normalized named entity.
- `POST /api/graph/relations`: Insert a typed relation edge (`verified` or `inferred`).
- `POST /api/graph/extract`: Extract entities and relations from text using relational verb constraints.
- `GET /api/graph/paths?source=A&target=B`: Execute multi-hop BFS path traversal.
- `GET /api/graph/contradictions`: Scan graph for conflicting edges.

---

## 5. Autonomous Research & Learning Endpoints

- `GET /api/learning-tasks`: List active and pending research tasks.
- `POST /api/learning-tasks`: Create a new autonomous research task.
- `POST /api/learning-tasks/{id}/approve`: Grant human approval for sensitive or high-priority research.
- `POST /api/learning-tasks/{id}/retry`: Retry a failed research task.
- `POST /api/research/query`: Execute instant deep research against verified knowledge bases.

---

## 6. RAG Pipeline Endpoints

- `POST /api/rag/ingest`: Ingest, chunk, and embed an uploaded local document.
- `POST /api/rag/retrieve`: Perform hybrid dense vector + BM25 search over ingested chunks.
- `POST /api/rag/evaluate`: Compute Precision@K, Recall@K, and MRR metrics on test queries.

---

## 7. System Telemetry & Safety Endpoints

- `GET /api/system/telemetry`: Hardware resource usage (CPU %, RAM GB, VRAM MB, SQLite size KB).
- `POST /api/code/execute`: Execute Python code in a sandboxed process with timeouts.
