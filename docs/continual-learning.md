# AAROHAN Continual Learning & Catastrophic Forgetting Prevention

**Phase 6 Deliverable** | November 2026

## Overview

In accordance with PRD Section 11 and Prompt 11, AAROHAN implements a controlled continual learning loop. Model learning (gradient updates to neural network parameters) is strictly separated from memory storage (database records), protecting the model from catastrophic forgetting and untested degradation.

## Architecture

1. **Experience Replay Buffer (`aarohan/continual.py:ReplayBuffer`)**:
   - Preserves historical instruction and pretraining datasets.
   - Blends new candidate examples with historical replay data at a configurable ratio (default: 50%).
   - Balances domain representation so that fine-tuning on new concepts reinforces existing language and reasoning patterns.

2. **Regression Gate (`RegressionGate`)**:
   - Before any candidate checkpoint is certified or staged, it is benchmarked against the certified baseline model on historical held-out test data.
   - Computes percentage regression:
     $$\Delta_{\text{reg}} = \frac{\mathcal{L}_{\text{candidate}} - \mathcal{L}_{\text{baseline}}}{\mathcal{L}_{\text{baseline}}} \times 100\%$$
   - Candidates showing regression exceeding the maximum allowed threshold (default: 8.0%) are automatically **rejected**.

3. **Rollback Manager (`RollbackManager`)**:
   - Maintains deployment history and current active model pointers in `checkpoints/deployment_registry.json`.
   - Provides instantaneous rollback to previous verified model checkpoints if regression anomalies are detected during staging or inference.

## CLI Usage

```powershell
# 1. Train continual learning candidate with replay
python scripts/continual_cli.py train-candidate `
    --base-checkpoint checkpoints/aarohan-sft-001.pt `
    --tokenizer artifacts/tokenizers/wikitext2-bpe/tokenizer.json `
    --new-data data/instructions/test_instructions.jsonl `
    --replay-data data/instructions/train_instructions.jsonl `
    --output checkpoints/candidates/continual-candidate-01.pt `
    --epochs 5 --replay-ratio 0.5

# 2. Evaluate regression gate on historical benchmark
python scripts/continual_cli.py eval-gate `
    --base-checkpoint checkpoints/aarohan-sft-001.pt `
    --candidate-checkpoint checkpoints/candidates/continual-candidate-01.pt `
    --tokenizer artifacts/tokenizers/wikitext2-bpe/tokenizer.json `
    --benchmark-file data/instructions/val_instructions.jsonl `
    --max-regression 8.0

# 3. Promote/Deploy certified model
python scripts/continual_cli.py deploy `
    --checkpoint checkpoints/candidates/continual-candidate-01.pt `
    --note "Passed regression gates"

# 4. Rollback in case of anomaly
python scripts/continual_cli.py rollback --reason "Performance regression detected"
```
