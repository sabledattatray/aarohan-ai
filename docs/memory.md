# AAROHAN Memory Architecture (Phase 4)

AAROHAN implements a persistent six-layer cognitive memory architecture independent of model parameters, surviving model replacement and application restarts as mandated in PRD Section 9 and Prompt 07.

## Memory Layers

1. **Layer 1 – Working Memory (`working`)**:
   - Ephemeral active conversation buffer and reasoning scratchpad.
2. **Layer 2 – Episodic Memory (`episodic`)**:
   - Interaction history, user query/response events, problem-solving attempts, and user corrections.
3. **Layer 3 – Semantic Memory (`semantic`)**:
   - General facts, conceptual definitions, architecture specifications, and stable domain knowledge.
4. **Layer 4 – Procedural Memory (`procedural`)**:
   - Reusable step-by-step procedures, data-cleaning pipelines, and training recipes.
5. **Layer 5 – Knowledge Graph (`knowledge_entities`, `knowledge_relations`)**:
   - Explicit structured concepts and semantic relationships (e.g., `SelfAttention -> part_of -> Transformer`).
   - Supports graph traversal, directional querying, and multi-hop path finding.
6. **Layer 6 – Learning Memory (`learning`)**:
   - Training benchmarks, loss metrics, knowledge gaps, evaluation failures, and regression tracking.

## Retrieval Modes

- **Keyword Search (`keyword`)**: Fast SQLite FTS5 full-text indexing with Porter/unicode61 tokenization and BM25 ranking.
- **Vector Search (`vector`)**: Deterministic 128-dimensional dense embeddings with cosine similarity matching. 100% offline and reproducible.
- **Hybrid Search (`hybrid`)**: Convex combination of BM25 keyword score and cosine similarity vector ranking.

## CLI Usage

```powershell
# Add memory across layers
python scripts/memory_cli.py add --content "RoPE encodes relative positions using complex 2D rotations." --type semantic --source "architecture.md" --verified-by "reviewer"

# Search using vector similarity or hybrid mode
python scripts/memory_cli.py search "rotary token positions" --mode vector
python scripts/memory_cli.py search "transformer architecture" --mode hybrid

# Knowledge Graph (Layer 5)
python scripts/memory_cli.py entity-add "SwiGLU" --type "activation" --description "Gated feed-forward layer"
python scripts/memory_cli.py entity-add "Transformer" --type "architecture"
python scripts/memory_cli.py relation-add "SwiGLU" "Transformer" --relation "part_of"
python scripts/memory_cli.py graph-query "Transformer"
python scripts/memory_cli.py graph-path "SwiGLU" "Transformer"

# Run memory consolidation
python scripts/memory_cli.py consolidate

# Check contradiction before adding
python scripts/memory_cli.py check-contradiction "RoPE does not encode relative token positions"
```

## Quality and Verification Safeguards

- **No Silent Elevation**: Generated claims remain `unverified` by default. Consolidation never promotes unverified records to verified facts without reviewer attribution.
- **Sensitive Data Blocked**: The storage layer strictly rejects `privacy_classification="sensitive"` until hardware encryption and access controls are implemented.
- **Contradiction Detection**: Flags polarity mismatches between new assertions and high-confidence existing records.
