# AAROHAN architecture and module contracts

## Status and scope

This document establishes Phase 0 boundaries from sections 3, 4, 19 and 21 of the PRD. It describes intended interfaces and distinguishes implemented code from validated behavior. The first executable milestone is a small CPU-capable next-token training run on a tiny, locally supplied, license-documented corpus, followed by validation and checkpoint reload. The current synthetic smoke path checks pipeline mechanics only. No external model service is required.

## System flow

```text
Approved sources -> license/provenance manifest -> normalization and filtering
                 -> project tokenizer -> tokenized train/validation partitions
                 -> random-init decoder Transformer -> bounded training run
                 -> checkpoint + run metadata -> validation and regression report

Chat / documents -> orchestrator -> local model
                                -> persistent memory and sourced knowledge (later phase)
                                -> response with provenance and uncertainty (later phase)

Evaluation failures -> learning task proposal -> approved research and validation
                     -> candidate training data -> candidate checkpoint
                     -> independent evaluation -> explicit promotion or rollback
```

## Component contracts

| Component | Owns | Must not own |
|---|---|---|
| Data registry and pipeline | Source/license manifests, normalization, filtering, splits, hashes and token counts | Model weights or undocumented source ingestion |
| Tokenizer | Vocabulary, special tokens, encode/decode and immutable version metadata | Model training decisions or silent vocabulary changes |
| Model | Configurable decoder-only Transformer with random initialization, forward pass and local generation | External inference calls, persistent memory, or autonomous actions |
| Trainer | Optimization, validation, deterministic configuration, resume state and run logs | Selecting/promoting production checkpoints |
| Checkpoint registry | Immutable checkpoint artifacts and links to tokenizer, config, data and evaluation | Overwriting a prior stable checkpoint |
| Evaluation | Held-out metrics and regression comparisons by model version | Solely model-generated grading |
| Memory/knowledge (later) | Persistent records, sources, confidence, privacy and retrieval | Treating generated text as verified fact or silently changing model weights |
| Research/learning (later) | Bounded, auditable proposals from approved sources | Arbitrary command execution or unreviewed production updates |
| API/UI (later) | User interaction and clear origin labels for model, memory and documents | Silent commercial-model fallback |

## Initial model boundary

Implement a causal decoder Transformer in PyTorch: token embeddings, RoPE, causal multi-head self-attention, RMSNorm, SwiGLU, residual blocks and vocabulary projection. Initialization is random and testable with a fixed seed. The PRD's 8-block, width-512 configuration is a proposal to benchmark, not a machine-independent default. Begin with a deliberately small configuration and increase only after measured memory and throughput checks.

Tokenizer artifacts must be versioned with a model checkpoint. A checkpoint is incomplete without model configuration, tokenizer version, optimizer/scheduler state (for resumable training), step, RNG state, data manifest/checksum, code revision and validation results.

## Data and trust boundaries

Every source record carries source identifier, license, collection date and processing version. Train/validation/evaluation partitions remain separated. Documents and model outputs are untrusted inputs; document instructions are never executed. Research findings retain citations to source records. Personal data and license restrictions must be handled before training use.

## First executable milestone and acceptance

1. Environment report records Python, PyTorch, device, RAM where available and CUDA status.
2. A tiny local corpus is accompanied by a source/license manifest and split reproducibly.
3. Tokenizer encode/decode and special-token behavior are checked.
4. A small random-init model has an asserted parameter count and passes shape, causal-mask, forward and backward checks.
5. A short CPU training run produces finite train and validation losses.
6. An atomic checkpoint reload reproduces model outputs and contains enough state to resume; no existing stable artifact is overwritten.

The milestone demonstrates pipeline integrity only; it does not demonstrate useful language ability. Do not scale training until this path is reproducible.

## Hardware baseline (user-reported, 2026-11-25)

Python 3.12.10, Windows 11 x64, AMD family 23 model 113 CPU with 16 logical CPUs, 17,058,258,944 bytes RAM (~15.9 GiB), PyTorch 2.14.1+cu132, CUDA 13.2, NVIDIA GeForce RTX 2060 with 6,442,123,264 bytes reported VRAM (~6 GiB). This is suitable for architecture experiments and tiny training runs. Batch size, context length and model width must remain modest and be benchmarked; it is not evidence that the PRD's larger proposed configuration fits.

## Tokenizer milestone

`aarohan.tokenizer.train_bpe` trains a project BPE tokenizer from the training file only, using byte-level pre-tokenization for Unicode coverage and all nine required special tokens. It saves `tokenizer.json` plus metadata including the training file SHA-256 and provenance manifest SHA-256. The CLI refuses to proceed unless the corpus validator passes. A 256-token vocabulary on the tiny toy corpus is solely an end-to-end test artifact, not a useful language tokenizer.

`scripts/prepare_tokenized_data.py` validates provenance before encoding, uses the saved tokenizer, and writes separate 1D `torch.long` train and validation tensors. Its metadata pins the exact corpus manifest, split files, and tokenizer by SHA-256. Validation text is encoded only after tokenizer training and never enters tokenizer fitting.

`scripts/train_tiny_corpus.py` checks those hashes before training, optimizes on random context windows from the training stream, reports held-out validation loss without updating on it, and writes an atomic checkpoint with optimizer and RNG state. On the current 474-token / 235-token AI-generated toy corpus, this is only a smoke experiment and is expected to overfit; it is not a model capability result.

The trainer supports `--resume-from checkpoint.pt`; `--steps` is the total target step count, not the number of extra steps. Resume verifies tokenizer/config compatibility and restores model, optimizer, Python RNG, Torch RNG, and CUDA RNG state where applicable. A deterministic CPU test compares a resumed run against an uninterrupted run.

Layer count, attention heads, and width are configurable through `--layers`, `--heads`, and `--width`. The 5–10M-parameter architecture-validation configuration can be trained on the current tokenizer by using 4 layers, 6 heads, and width 288; with vocabulary 265, this is about 5.4M parameters. Use a context length no greater than the available token stream for validation.

The first user-reported CUDA run completed on RTX 2060 with a 148,352-parameter model, 100 steps, context 32 and batch 8. Training loss changed from 5.598822 to 3.092667; held-out validation loss was 3.144345; checkpoint reload max logit difference was 0.0. Treat these values as a reproducibility record for the toy run only.

`scripts/generate_local.py` loads the local model checkpoint and exact tokenizer hash recorded in that checkpoint, rejects mismatches, and generates a sampled continuation without a network call. Output from the toy checkpoint is an integration demonstration only.

`scripts/evaluate_checkpoints.py` scores checkpoints on the complete held-out token stream and reports token-weighted cross-entropy, perplexity, and loss delta against the first checkpoint. It verifies tokenizer and validation-source identity before comparison and never updates model weights.

`aarohan.registry` stores append-only candidate records bound to checkpoint and evaluation-report SHA-256 plus matching tokenizer/data evidence. An explicit `activate` command rechecks both artifact hashes before changing only the local active checkpoint pointer; `rollback` restores the prior pointer. This is local experiment bookkeeping, not production deployment.

`scripts/benchmark_hardware.py` benchmarks the tiny pipeline model and a roughly 6.5M-parameter AAROHAN-0.01 candidate on fixed random token batches. It records throughput, finite loss, and peak CUDA allocation/reservation. These measurements characterize hardware capacity only; they are not quality metrics.

`aarohan.memory.MemoryStore` is the first local episodic-memory baseline: SQLite persistence, keyword search (FTS5 when available), source/confidence/privacy metadata, duplicate handling, correction history, and deletion. New records default to unverified; marking one verified requires a source and reviewer. This does not yet meet the PRD's PostgreSQL/pgvector semantic-memory target, which remains a later storage adapter milestone.

The `scripts/memory_cli.py` command-line interface exercises that store without an application server. See `docs/memory.md` for local usage and privacy guidance.

Memory keyword search can optionally filter by source label or filename, so results can be scoped to a specific ingested document.

`scripts/chat_local.py` demonstrates the orchestrator boundary by returning source-attributed memory and local model text as separate fields. It intentionally does not feed retrieved document text into the current toy model; grounded generation remains future work.

`scripts/ingest_document.py` adds bounded TXT/Markdown/HTML/PDF ingestion that preserves document hashes, chunk IDs, and PDF page numbers in source fields; it stores extracted text as unverified memory and blocks sensitive-classified documents. See `docs/document-ingestion.md`.

The evaluation comparison of the 100-step and resumed 120-step toy checkpoints scored all 234 validation targets: loss/perplexity 3.168759/23.7779 versus 3.018733/20.4654. The trainer now includes the final partial validation window and has a regression check against evaluator loss. These numbers only describe this tiny synthetic corpus.

