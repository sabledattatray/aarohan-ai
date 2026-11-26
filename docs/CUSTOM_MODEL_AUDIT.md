# AAROHAN AI — Custom Neural Model Technical Audit

**Architect & Developer:** Datta Sable  
**Project:** AAROHAN AI  
**Audit Scope:** Custom Neural Transformer (`AAROHANLM`), Tokenizer, Trainer, Checkpoint Registry, and Multi-Area Evaluator  
**Date:** November 2026  

---

## 1. Executive Summary

This document presents a comprehensive, transparent technical audit of the project-owned custom neural language model (`AAROHANLM`) developed for AAROHAN AI.

### Core Architecture Classification
- **Architecture Family:** Decoder-only Causal Transformer
- **Norm Type:** Root Mean Square Layer Normalization (RMSNorm) with learnable scale
- **Positional Encoding:** Rotary Position Embeddings (RoPE) applied to Query and Key projections
- **Attention Mechanism:** Causal Multi-Head Scaled Dot-Product Attention (`F.scaled_dot_product_attention`) with KV-caching support
- **Feedforward Network:** SwiGLU (Swish-Gated Linear Unit) with hidden dimension $4 \times d_{\text{model}}$
- **Weight Tying:** Output language projection head ties weights with token input embedding table
- **Parameter Count:**
  - Small WikiText-2 / SFT Config: **1,573,504 parameters** ($\approx 1.57\text{M}$)
  - Medium Toy Smoke Config: **5,387,328 parameters** ($\approx 5.39\text{M}$)
- **Target Hardware:** RTX 2060 6GB (CUDA FP16 / FP32) and CPU fallback

---

## 2. Subsystem Audit Breakdown

### A. Tokenizer & Vocabulary
- **Implementation:** `aarohan/tokenizer.py` (Byte-Level BPE via HuggingFace `tokenizers` engine)
- **Vocabulary Size:** Configurable from 64 to 8,192 tokens.
- **Special Tokens:** `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|separator|>`
- **Security & Integrity:** Enforces SHA-256 hash pinning. Model checkpoints store `tokenizer_sha256` to reject checkpoints loaded against mismatched vocabularies.

### B. Positional Encoding (RoPE)
- **Implementation:** `aarohan/model.py` (`CausalSelfAttention._rotate_half` and frequency register buffer `inv_freq`).
- **Characteristics:** RoPE provides relative positional awareness and permits length extrapolation up to configured `context_length`.
- **Validation:** Head dimension must be even (`(width // heads) % 2 == 0`). Validated in `AAROHANConfig.__post_init__`.

### C. Normalization & Feedforward
- **RMSNorm:** Eliminates mean centering overhead of LayerNorm, yielding faster compute on RTX 2060 hardware while preserving gradient scale stability.
- **SwiGLU:** Implements $F_{\text{silu}}(x W_{\text{gate}}) \times (x W_{\text{value}}) W_{\text{output}}$. Provides superior expressive capacity compared to standard ReLU/GELU feedforward networks.

### D. Loss Calculation & Numerical Stability
- **Criterion:** Cross-Entropy with reduction across sequence tokens (`F.cross_entropy`).
- **Gradient Clipping:** Strict norm clipping (`torch.nn.utils.clip_grad_norm_(parameters, 1.0)`) prevents exploding gradients.
- **Numerical Stability:** Guardrails assert loss is finite (`torch.isfinite(loss)`), failing gracefully if NaNs or infinities arise.

---

## 3. Checkpoint Inventory & Validation Metrics

The custom model family has been evaluated across multiple training iterations recorded in `checkpoints/registry.json`:

| Checkpoint Name | Vocab | Context | Layers | Heads | Width | Steps | Val Loss | Val Perplexity | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `toy-corpus-smoke.pt` | 265 | 32 | 2 | 2 | 64 | 100 | 3.1688 | 23.78 | Pipeline integration test |
| `toy-corpus-resumed.pt` | 265 | 32 | 2 | 2 | 64 | 120 | 3.0187 | 20.47 | Resumption validation |
| `aarohan-001-toy-smoke.pt`| 265 | 64 | 4 | 6 | 288 | 20 | 3.1865 | 24.20 | Wider architecture test |
| `aarohan-wikitext2-smoke.pt` | 8,192 | 128 | 2 | 4 | 128 | 100 | 7.3186 | 1508.02 | WikiText-2 baseline |
| `aarohan-wikitext2-001.pt` | 8,192 | 128 | 2 | 4 | 128 | 1,000 | 6.3392 | 566.35 | WikiText-2 intermediate |
| `aarohan-wikitext2-005k.pt`| 8,192 | 128 | 2 | 4 | 128 | 5,000 | 5.4949 | 243.45 | WikiText-2 converged |
| `aarohan-sft-001.pt` | 8,192 | 128 | 2 | 4 | 128 | Epoch 15 | 2.1400 | 8.50 | Supervised instruction tuned |
| `aarohan-release-candidate.pt` | 8,192 | 128 | 2 | 4 | 128 | Replay SFT | 2.1270 | 8.39 | Certified continually tuned |

---

## 4. Honest Capability Assessment & Operational Boundary

> [!IMPORTANT]
> **Production Boundary Policy:**
> - `AAROHANLM` (~1.57M parameters) is an experimental neural model engineered to prove project-owned Transformer mechanics, custom BPE tokenization, loss convergence, and continual learning algorithms.
> - `AAROHANLM` **cannot and must not** be claimed as independently intelligent or capable of replacing general-purpose foundation models.
> - In AAROHAN AI, open-domain reasoning, complex code generation, and multi-turn world knowledge are orchestrated by the **Foundation Model** (`Qwen/Qwen2.5-0.5B-Instruct`), running locally on workstation hardware.
> - The system reports exact model source attribution on every API response (`knowledge_source: "model_weights"` or `"custom_engine"`), maintaining 100% technical transparency.
