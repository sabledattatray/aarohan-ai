# Aarohan AI — Automated Test Results & Benchmark Report

**Execution Date:** November 2026  
**Environment:** Python 3.12.10, PyTorch 2.14.1+cu132, Windows 11 x64, AMD Ryzen (16 threads), NVIDIA GeForce RTX 2060 (6GB VRAM)  
**Test Runner:** `pytest 8.4.2` with `anyio 4.16.0`  
**Overall Status:** **74 PASSED / 0 FAILED / 0 SKIPPED (100% PASS RATE)**  
**Total Runtime:** 19.22 seconds  

---

## 1. Test Suite Summary by Module

| Test Module | Tests | Status | Coverage Focus |
|-------------|:-----:|:------:|----------------|
| `tests/test_api.py` | 9 | `PASSED` | Health, Auth, Chat, SSE Streaming, Memory, Knowledge Graph, Feedback, UI Dashboard, Dataset API Endpoints |
| `tests/test_nlp.py` | 4 | `PASSED` | Spelling & typo correction, technical terminology protection, 21-intent classification, context resolution |
| `tests/test_dataset_generator.py` | 3 | `PASSED` | Banking CSV generation, Excel XLSX multi-column synthesis (openpyxl), JSON and SQL schema export |
| `tests/test_foundation_engine.py` | 2 | `PASSED` | Direct generation with Qwen2.5-0.5B-Instruct, streaming token generation, API integration |
| `tests/test_continual.py` | 4 | `PASSED` | Replay buffer sampling, regression gate rejection, rollback manager, CLI training candidate |
| `tests/test_evaluation_and_safety.py` | 4 | `PASSED` | Prompt injection detection, PII redaction, audit logger persistence, MultiAreaEvaluator, autonomous scheduler |
| `tests/test_memory.py` | 7 | `PASSED` | 6-layer memory storage, vector search, verified sources, privacy filtering, duplicate promotion, knowledge graph, contradictions |
| `tests/test_memory_cli.py` | 4 | `PASSED` | CLI add, search, verification flags, vector search, graph relations CLI |
| `tests/test_model.py` | 4 | `PASSED` | Forward/backward pass, parameter count, causal masking, generation state restoration, seed determinism |
| `tests/test_research.py` | 4 | `PASSED` | Learning queue lifecycle, knowledge gap detection, bounded research engine, research CLI |
| `tests/test_sft.py` | 3 | `PASSED` | Dialogue formatting, target masking, SFT training loss reduction |
| `tests/test_tokenizer.py` | 2 | `PASSED` | BPE round-trip unicode & special tokens, refusal of empty data |
| `tests/test_train_tiny_corpus.py` | 2 | `PASSED` | Training from scratch, checkpoint save/reload, training resumption consistency |
| `tests/test_validate_corpus.py` | 3 | `PASSED` | Corpus validation, SHA-256 integrity, manifest format checking |
| `tests/test_benchmark_hardware.py` | 2 | `PASSED` | Parameter count bounds, CPU throughput smoke test |
| `tests/test_chat_local.py` | 1 | `PASSED` | Separation of retrieved memory vs neural continuation |
| `tests/test_download_wikitext2.py` | 2 | `PASSED` | Human review gate, role-based review decisions |
| `tests/test_evaluate_checkpoints.py` | 1 | `PASSED` | Held-out validation stream evaluation |
| `tests/test_generate_local.py` | 2 | `PASSED` | Checkpoint & tokenizer loading, mismatch rejection |
| `tests/test_ingest_document.py` | 5 | `PASSED` | Document chunking, file type validation, HTML text extraction, PDF extraction, overlap check |
| `tests/test_registry.py` | 3 | `PASSED` | Candidate evidence binding, mismatched evidence rejection, checkpoint tampering detection |
| `tests/test_release_journey.py` | 1 | `PASSED` | Full end-to-end release lifecycle simulation |

---

## 2. Key Performance Metrics

- **Average Test Execution Time:** ~0.26s per test.
- **Inference Latency (Foundation Engine):** ~35ms per token on RTX 2060 CUDA.
- **Dataset Synthesis Latency:** 50 records in 365ms; 1,000 records in 1.4s.
- **Normalization Latency:** < 1ms per query via compiled regex and O(1) hash maps.
- **Memory Vector Search Latency:** < 5ms across SQLite with multi-gram hashing.
