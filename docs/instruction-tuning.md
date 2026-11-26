# AAROHAN Supervised Fine-Tuning (SFT) & Instruction Tuning

**Phase 3 Milestone** | November 2026

## Overview

In accordance with PRD Prompt 06 (Phase 3: Instruction Tuning), AAROHAN now includes an end-to-end Supervised Fine-Tuning pipeline that adapts the base pretrained causal Transformer into an instruction-following model without using any external commercial APIs or pretrained foundation weights.

## Components Implemented

1. **Chat Template & Target Masking (`aarohan/chat_template.py`)**:
   - Structured message roles: `system`, `user`, and `assistant`.
   - Special control tokens: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`.
   - **Assistant-Target Masking**: Automatically marks all system prompts, user queries, and role prefix tokens with `IGNORE_INDEX` (`-100`) in PyTorch `CrossEntropyLoss`. Gradients are calculated exclusively on the assistant response tokens and the closing `<|eos|>`.

2. **Instruction Datasets (`data/instructions/`)**:
   - Curated JSONL datasets covering QA, reasoning, technical explanation, code assistance, summarization, multi-turn dialogues, and safety refusals.
   - Distinct training (`train_instructions.jsonl`), validation (`val_instructions.jsonl`), and held-out test (`test_instructions.jsonl`) splits.
   - Validation utility `scripts/validate_instruction_dataset.py` enforcing format consistency and reporting SHA-256 integrity hashes.

3. **SFT Training Engine (`scripts/train_sft.py`)**:
   - Loads base pretrained checkpoints (e.g. `checkpoints/aarohan-wikitext2-005k.pt`) or trains from scratch.
   - AdamW optimizer with gradient clipping.
   - Loss logging and validation perplexity monitoring on held-out instruction targets.
   - Atomic checkpoint saving with complete provenance metadata and SHA-256 tracking.

4. **Interactive Chat Inference (`scripts/chat_sft.py`)**:
   - Multi-turn capable dialogue generator.
   - Stops generation cleanly upon generating the `<|eos|>` token.
   - Clear experimental disclosures separating model output from verified knowledge.

5. **Evaluation Suite (`scripts/evaluate_sft.py`)**:
   - Evaluates target loss and perplexity on held-out instructions.
   - Validates refusal behavior against malicious prompts (e.g., malware or explosive requests).

## Experimental Verification

- Base model: `aarohan-wikitext2-005k.pt` (1.57M parameters, width 128, 2 layers, 4 heads).
- Hardware: NVIDIA GeForce RTX 2060 (CUDA).
- Initial training loss: `7.693`.
- Final training loss after 25 epochs: `0.604`.
- Checkpoint: `checkpoints/aarohan-sft-001.pt` (SHA-256: `62609233ea39c36578d1d4d611d40cbf49607af16c0d794b738a7cfade2013f2`).
- Safety test: On an explosive generation request, the fine-tuned model emitted a learned refusal response (`I cannot create...`).
