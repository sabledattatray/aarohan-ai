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
- [ ] Initialize Git repository and record initial revision.
- [x] Initialize Git metadata; first status check is currently blocked by Git's dubious-ownership safeguard in the Codex sandbox.
- [ ] Make the user's Python environment accessible to the workspace or run the checks from the project terminal.
- [ ] Pin a supported Python/PyTorch development environment.
- [x] Record CPU, RAM, GPU, VRAM and CUDA status with user-provided environment output.
- [ ] Establish a hardware benchmark baseline.
- [ ] Create a dataset licensing register before acquiring data.
- [x] Add a minimal pytest framework and development standards.
- [x] Run architecture tests in the user's terminal (4 passed, user-reported).
- [x] Run the CUDA smoke training after fixing direct-script package imports; observed finite validation loss and exact checkpoint reload (user-reported; details in environment record).

**Exit condition:** A small PyTorch training run and checkpoint reload have been reported successful. The CUDA synthetic smoke path is complete; the broader PRD Phase 0 exit also calls for a hardware benchmark baseline and a dataset license register, which remain open.

## Phase 1 — Original model

- [x] Implement project BPE training, version/provenance metadata, and tokenizer tests.
- [x] Run tokenizer tests and train the first tokenizer in the user's Python environment (9 tests passed; user-reported tokenizer vocabulary 265).
- [x] Implement reproducible tokenizer-to-token-ID preparation with separate train/validation tensors and source/tokenizer checksums.
- [x] Run dataset preparation and verify outputs in the user's Python environment (474 train tokens, 235 validation tokens; user-reported).
- [x] Implement a bounded tiny-corpus next-token trainer with held-out loss, provenance/hash checks, and resumable checkpoint state.
- [x] Run trainer tests and a tiny-corpus training experiment in the user's Python environment (11 tests passed; CUDA checkpoint reload exact; user-reported).
- [ ] Implement and test random-init causal Transformer.
- [ ] Implement train and validation loops, logging and atomic resumable checkpoints.
- [ ] Implement local generation from an AAROHAN-trained checkpoint.

## Later phases

- [ ] Foundation data pipeline and pretraining
- [ ] Instruction tuning
- [ ] Persistent, source-aware memory and knowledge
- [ ] Bounded research and learning-gap queue
- [ ] Continual learning with replay, regression gates and rollback
- [ ] API, web interface, audits and integrated release evaluation

