# AAROHAN WikiText-2 experiment

**Run date:** 2026-11-26  
**Status:** Local research experiment; the checkpoint is a candidate, not a reliable assistant.

## Source and license notes

- Dataset: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), subset `wikitext-2-raw-v1`.
- Pinned revision: `6231e49f19a707241d6b84d9cff60a3a86b85a85`.
- The dataset card section was reviewed as CC BY-SA 4.0. Repository license tags also list CC-BY-SA-3.0 and GFDL. Keep source attribution and applicable share-alike notices with the downloaded corpus and derived artifacts; this report does not decide which notice applies to any particular excerpt.
- `data/wikitext2-review.json` records the user-reported trusted-adult license and privacy review. The assistant did not independently perform those reviews.

## Data handling

The raw Parquet splits were downloaded unchanged and verified by SHA-256:

| Split | Raw rows | Raw SHA-256 | Use |
|---|---:|---|---|
| Train | 36,718 | `49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701` | Tokenizer fit and model training |
| Validation | 3,760 | `a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f` | Model selection and validation metrics |
| Test | 4,358 | `3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925` | One-time final scoring only; never used for training or tuning |

Preparation removed blank rows and exact duplicates within each split. Validation additionally excluded 147 paragraphs duplicated in train. The one-time test preparation excluded 99 within-test duplicates, 180 train overlaps, and 2 validation overlaps. The resulting paragraph counts were 21,713 train, 2,214 validation, and 2,610 test. Test remains separate at `data/processed/wikitext2/test-evaluation/`.

The project BPE tokenizer was trained from train text only with vocabulary size 8,192. Tokenizer SHA-256: `131b1a92e80c8ede6c724e02ea3758c1824994453d4af65b7ffba7f2667b3740`. The tokenized streams contain 2,711,203 train tokens, 285,470 validation tokens, and 327,053 held-out test tokens. The test stream was produced only after selecting the 5,000-step checkpoint using validation.

## Model and run

- Architecture: 1,573,504 parameters; 2 layers; 4 attention heads; width 128; context 128; vocabulary 8,192.
- Hardware: NVIDIA GeForce RTX 2060; CUDA.
- Batch size: 2.
- Checkpoint: `checkpoints/aarohan-wikitext2-005k.pt`.
- Registered candidate ID: `8a3319c060a16a4b`.
- Checkpoint SHA-256: `8a3319c060a16a4bf006cff5b6bd3b4267a33690fe8e14beb48d9488873b2a24`.

## Results

All checkpoints were compared on the same 285,469 validation targets using the same tokenizer and validation source.

| Training steps | Validation loss | Validation perplexity |
|---:|---:|---:|
| 100 | 7.318554 | 1,508.022 |
| 1,000 | 6.339212 | 566.350 |
| 5,000 | 5.494919 | 243.452 |

The selected 5,000-step candidate was scored once on 327,052 test targets: loss `5.491286`, perplexity `242.569`. The test result is close to validation, but it does not demonstrate broad language ability. Checkpoint reload reproduced model logits exactly (`max difference 0.0`).

## Interpretation and next boundary

The model can produce a local continuation, but the observed sample remains incoherent. The 5,000-step checkpoint is useful for testing the training, checkpoint, evaluation, registry, and local generation paths. It is not a dependable answer system and should not be promoted as a capable assistant based only on these results.

Do not tune against this test result or use the test split for further training. Any later model-selection round needs a fresh held-out test source or must be reported as validation-only. Retain `data/wikitext2-review.json`, `data/raw/wikitext2-raw-v1/download-manifest.json`, and `data/processed/wikitext2/manifest.json` with this experiment to preserve provenance and notices.
