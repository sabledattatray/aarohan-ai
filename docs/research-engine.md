# AAROHAN Knowledge Acquisition & Bounded Research Engine

**Phase 5 Deliverable** | November 2026

## Overview

In accordance with PRD Prompts 08, 09, and 10, AAROHAN implements an autonomous research engine and knowledge-gap detection system that operates with strict safety bounds:
- **Untrusted Content Handling**: Source documents are treated strictly as untrusted data; instructions inside ingested documents are never executed as system commands.
- **Strict Execution Budgets**: Enforces document scan limits (`max_budget_docs`), storage limits, and depth budgets.
- **Source Attribution & Unverified Default**: Extracted facts are recorded with exact document provenance and remain `unverified` by default until human review.

## Architecture

1. **Learning Task Queue (`aarohan/research.py`)**:
   - Manages the `learning_tasks` table in persistent SQLite storage.
   - Supports task statuses: `pending`, `approved`, `in_progress`, `completed`, `rejected`, and `failed`.
   - Tracks research objectives, budget limits, artifact paths, and execution timestamps.

2. **Knowledge Gap Detector (`KnowledgeGapDetector`)**:
   - Analyzes co-occurring entities across memory chunks that lack connecting edges in the Knowledge Graph.
   - Computes gap priority:
     $$\text{priority} = \min(0.9, 0.4 + 0.1 \times N_{\text{co-occurrences}})$$
   - Automatically proposes prioritized research tasks to resolve missing conceptual relationships.

3. **Bounded Research Engine (`BoundedResearchEngine`)**:
   - Scans approved documents matching query keywords.
   - Extracts candidate semantic facts and discovers capitalized entity names.
   - Runs real-time contradiction detection against stored high-confidence memories.
   - Emits structured JSON research artifacts into `artifacts/research/`.

## CLI Usage

```powershell
# 1. Propose or detect learning gaps from memory
python scripts/research_cli.py gap-detect

# 2. Manually create an approved research task
python scripts/research_cli.py task-create --title "Research RoPE Mechanics" --query "rotary embeddings attention" --priority 0.85

# 3. List active or pending research tasks
python scripts/research_cli.py task-list --status pending

# 4. Execute bounded research against approved documentation
python scripts/research_cli.py research-run --task-id <TASK_ID> --docs docs/architecture.md docs/environment.md
```
