# Development checklist

Status follows the PRD. Check an item only after evidence exists in the repository or a recorded run.

## Phase 0 — Research and environment

- [x] Read the complete PRD and identify its ordered development prompts.
- [x] Establish project scope, component contracts and the first executable milestone.
- [x] Implement a configurable random-init Transformer and a synthetic train/validation/checkpoint smoke path.
- [x] Add CPU/GPU-compatible architecture and causality tests.
- [x] Record user-provided Python/PyTorch/CUDA/GPU report.
- [x] Implement repeatable tiny-model and AAROHAN-0.01 hardware benchmark with throughput/VRAM reporting.
- [x] Run the hardware benchmark and record the measured GPU baseline (user-reported; RTX 2060).
- [x] Create an empty original-corpus template with provenance manifest and validation guard.
- [x] Validate a small AI-generated toy corpus with separate train/validation files and recorded checksums (user-reported).
- [x] Initialize Git repository and record initial revision (commit 5502145).
- [x] Initialize Git metadata and configure safe.directory.
- [x] Make the user's Python environment accessible to the workspace (`C:\Users\sable\.venv\Scripts\python.exe`).
- [x] Pin a supported Python/PyTorch development environment (Python 3.12.10, PyTorch 2.14.1+cu132, CUDA available).
- [x] Record CPU, RAM, GPU, VRAM and CUDA status with user-provided environment output.
- [x] Establish a hardware benchmark baseline (`experiments/hardware-benchmark.json`).
- [x] Create a dataset licensing register before acquiring data (`data/README.md`, `data/wikitext2-review.json`).
- [x] Add a minimal pytest framework and development standards.
- [x] Run architecture tests in the user's terminal (36 passed).
- [x] Run the CUDA smoke training after fixing direct-script package imports; observed finite validation loss and exact checkpoint reload.

**Exit condition:** PASSED. Small PyTorch training runs, checkpoint reload, GPU benchmark, licensing register, and git baseline established.

## Phase 1 — Original model

- [x] Implement project BPE training, version/provenance metadata, and tokenizer tests.
- [x] Run tokenizer tests and train the first tokenizer in the user's Python environment (tokenizer vocabulary 265 and wikitext2 8,192).
- [x] Implement reproducible tokenizer-to-token-ID preparation with separate train/validation tensors and source/tokenizer checksums.
- [x] Run dataset preparation and verify outputs in the user's Python environment.
- [x] Implement a bounded tiny-corpus next-token trainer with held-out loss, provenance/hash checks, and resumable checkpoint state.
- [x] Run trainer tests and a tiny-corpus training experiment in the user's Python environment.
- [x] Implement and test random-init causal Transformer (RoPE, RMSNorm, SwiGLU, Multi-head attention).
- [x] Implement train and validation loops, logging and atomic resumable checkpoints.
- [x] Implement local generation from an AAROHAN-trained checkpoint (`scripts/generate_local.py`).

**Exit condition:** PASSED. AAROHAN generates text using only its own trained checkpoint and custom BPE tokenizer.

## Phase 2 — Foundation Training

- [x] Curate training corpus (WikiText-2 with license review and SHA-256 verification).
- [x] Data cleaning, deduplication, train/val/test splits, tokenization (`scripts/prepare_wikitext2.py`).
- [x] Pretraining engine runs (100, 1,000, and 5,000 steps on RTX 2060).
- [x] Model registry and evaluation tracking (`scripts/manage_model_registry.py`, `scripts/evaluate_checkpoints.py`).
- [x] One-time held-out test split scoring (loss 5.49, perplexity 242.57).

**Exit condition:** PASSED. Measurable reduction in validation loss and verified checkpoint reload.

## Phase 3 — Instruction Tuning

- [x] High-quality instruction dataset schema and validation (`data/instructions/`, `scripts/validate_instruction_dataset.py`).
- [x] Chat template formatter with special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`) and assistant target masking (`aarohan/chat_template.py`).
- [x] Supervised fine-tuning (SFT) engine with assistant target masking (`scripts/train_sft.py`).
- [x] Instruction-following and safety evaluation framework (`scripts/evaluate_sft.py`).
- [x] Multi-turn interactive chat CLI (`scripts/chat_sft.py`).
- [x] Automated unit and regression tests for SFT pipeline (3 tests passed, 39 total suite tests passed).
- [x] Trained SFT checkpoint on CUDA (`checkpoints/aarohan-sft-001.pt`, train loss reduced from 7.69 to 0.60).

**Exit condition:** PASSED. Model responds to instructions using its own weights with assistant target masking and safety refusal behavior.

## Phase 4 — Long-Term Memory

- [x] Multi-layer memory architecture covering Working, Episodic, Semantic, Procedural, and Learning layers (`aarohan/memory.py`).
- [x] Deterministic dense vector embeddings and cosine similarity search (`vector_search`).
- [x] Hybrid search combining FTS5 BM25 keyword retrieval and dense vector embeddings (`hybrid_search`).
- [x] Knowledge Graph (Layer 5) with entities (`knowledge_entities`), relations (`knowledge_relations`), graph query, and path finding.
- [x] Memory links, duplicate detection, and contradiction detection (`detect_contradictions`).
- [x] Memory consolidation engine reviewing episodic records and linking recurring concepts (`consolidate`).
- [x] Expanded CLI with multi-modal search and graph exploration commands (`scripts/memory_cli.py`).
- [x] Integrated hybrid retrieval in interactive model queries (`scripts/chat_local.py`).
- [x] Automated unit and integration tests (8 memory tests, 4 CLI tests, 43 total suite tests passed).

**Exit condition:** PASSED. Persistent 6-layer memory store survives session restarts, executes vector/keyword search, navigates the knowledge graph, and validates contradiction/consolidation logic.

## Phase 5 — Knowledge Acquisition & Bounded Research Engine

- [x] Learning task queue with statuses, priorities, and budget limits (`aarohan/research.py:LearningQueue`).
- [x] Knowledge gap detection from missing Knowledge Graph relationships and memory co-occurrences (`KnowledgeGapDetector`).
- [x] Bounded research engine with document budgets, contradiction screening, and entity extraction (`BoundedResearchEngine`).
- [x] Structured research artifacts generation in `artifacts/research/`.
- [x] CLI utility for learning tasks, gap detection, and research execution (`scripts/research_cli.py`).
- [x] Automated unit and integration tests (4 tests passed, 47 total suite tests passed).

**Exit condition:** PASSED. System can autonomously identify missing relationships, propose prioritized learning tasks, execute bounded research against permitted sources, and emit verifiable research artifacts.

## Later phases

- [ ] Continual learning with replay, regression gates and rollback (Phase 6)
- [ ] Autonomous learning scheduler (Phase 7)
- [ ] API, web interface, audits and integrated release evaluation (Phase 15/16)

