# AAROHAN Autonomous Learning Scheduler, Multi-Area Evaluation & Safety

**Phase 7 Deliverable** | November 2026

## Overview

In accordance with PRD Section 13, Section 18, and Prompts 12, 13, and 14, AAROHAN introduces:
- **Multi-Area Evaluation Engine (`aarohan/evaluator.py`)**: Independent, deterministic grading across 6 capability domains (Language Modeling, Mathematics, Programming, Safety Refusals, Instruction Following, and Memory Consistency).
- **Autonomous Learning Scheduler (`aarohan/scheduler.py`)**: End-to-end orchestration that scans the queue, checks available resources, executes bounded research, trains continual learning candidates with replay, enforces regression gates, and archives cycle execution artifacts.
- **Safety and Governance (`aarohan/safety.py`)**: Real-time prompt injection detection, PII redactor (emails, phone numbers, API keys), and tamper-evident audit logging.

## Multi-Area Capability Domains

| Domain | Evaluation Method | Success Criteria |
| :--- | :--- | :--- |
| **Mathematics** | Deterministic arithmetic & logical displacement prompts | Presence of correct numerical outcomes |
| **Programming** | Functional Python syntax & logic tests | Clean algorithmic implementation keywords |
| **Safety** | Hostile adversarial prompts (malware, explosives, surveillance) | Explicit refusal emission; zero policy violations |
| **Instruction Following** | Prompt adherence & structured response tests | Format conformance |
| **Language Modeling** | Held-out instruction loss & perplexity | Finite cross-entropy target loss |

## Autonomous Learning Cycle Flow

```text
[CHECK HARDWARE HEADROOM]
          |
[SELECT PRIORITY TASK FROM QUEUE]
          |
[BOUNDED RESEARCH ON APPROVED SOURCES]
          |
[FORMULATE CANDIDATE INSTRUCTIONS]
          |
[TRAIN WITH BLENDED EXPERIENCE REPLAY]
          |
[EVALUATE REGRESSION GATES AGAINST BASELINE]
          |
[RECORD AUDIT LOG & STAGE ARTIFACT]
          |
[DEPLOYMENT PENDING EXPLICIT APPROVAL]
```
