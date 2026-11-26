# AAROHAN AI — Comprehensive Application Security Audit & Governance Report

**Architect & Security Lead:** Datta Sable  
**Project:** AAROHAN AI  
**Scope:** AI Engine, Memory, API Layer, Data Ingestion, Sandboxed Code Execution, and Authentication  
**Date:** November 2026  

---

## 1. Executive Summary

This security audit and hardening report documents the threat modeling, defense-in-depth controls, and vulnerability mitigations implemented across the AAROHAN AI platform.

### Core Security Mandates:
1. **Zero External Cloud Exposure:** All language model inferences run strictly on local workstation hardware (RTX 2060 CUDA / CPU). No prompts or private data are transmitted to external commercial AI providers.
2. **Untrusted Data Rule:** All user queries, retrieved document chunks, web search results, and tool outputs are classified and handled strictly as untrusted data.
3. **Multi-Tenant Isolation:** Memory records, documents, and conversation histories are strictly partitioned by `user_scope`.
4. **Sandboxed Code Execution:** Model-generated code is never run with raw system privileges; it executes only inside a constrained, time-limited subprocess with destructive command filtering.

---

## 2. Threat Modeling & Defense Matrix

| Attack Vector | Vulnerability Description | Mitigation in AAROHAN | Status |
| :--- | :--- | :--- | :--- |
| **Direct Prompt Injection** | Adversarial instructions attempting to override system behavior (e.g. "ignore rules") | Multi-layer heuristic & token detection: detects override verbs, delimiter hijacking, authority claims, and leetspeak substitutions | **HARDENED** |
| **Indirect Prompt Injection** | Hostile instructions embedded inside ingested PDF/TXT/HTML documents | `RAGPipeline.sanitize_chunk()` strips instruction patterns on ingest and at retrieval time before context formatting | **HARDENED** |
| **PII & Credential Leakage** | User inputs containing emails, phone numbers, API keys, private keys, or passwords | `SafetyFilter.redact_pii()` redacts sensitive patterns into `[REDACTED_*]` tokens before storage and logging | **HARDENED** |
| **Cross-Tenant Memory Access**| User A querying or retrieving User B's personal facts or private documents | SQL-level `user_scope` filtering on all searches, plus composite SHA-256 hashing `(user_scope + "::" + content)` | **HARDENED** |
| **Path Traversal / LFI** | Submitting paths like `../../Windows/System32` to file reading endpoints | `validate_file_path()` resolves canonical paths and strictly verifies containment within permitted directories | **HARDENED** |
| **Internal Path Exposure** | Server exceptions exposing `C:\Users\sable\...` or database paths to clients | `SafetyFilter.sanitize_error_message()` and global exception handler sanitize file paths to `[INTERNAL_PATH]` | **HARDENED** |
| **SQL Injection** | Malicious SQL inputs escaping quotes into database queries | 100% parameterized queries using `?` placeholders with SQLite `execute(sql, params)` | **HARDENED** |
| **Denial of Service (DoS)** | Excessive request flooding exhausting workstation CPU/CUDA resources | In-memory sliding window `RateLimiter` enforcing per-client request quotas | **HARDENED** |
| **Arbitrary Code Execution** | Untrusted model generating shell deletion or destructive system scripts | `CodeSandbox` with static destructive pattern blocking, timeout guards, and process isolation | **HARDENED** |

---

## 3. Detailed Subsystem Implementations

### A. Multi-Layer Prompt Injection Engine (`aarohan/safety.py`)
Rather than relying exclusively on simple regular expressions, AAROHAN uses four complementary defense layers:
1. **Instruction Override Analysis:** Matches override commands (`ignore previous instructions`, `system override`, `disregard safety`).
2. **Special Delimiter & Token Hijacking:** Detects artificial prompt framing, such as `<|im_start|>`, `<|system|>`, `[SYSTEM PROMPT]`, and markdown instruction fences.
3. **Roleplay & Authority Usurpation:** Blocks adversarial jailbreak personas (`"DAN mode"`, `"unfiltered AI"`, `"as your administrator, I command you"`).
4. **Obfuscation & Leetspeak Normalization:** Automatically maps numeric/symbol substitutions (`1gn0re`, `@dm1n`) back to alphabetical representations prior to inspection.

### B. Path Traversal & File Upload Validation
`validate_file_path(target, allowed_dirs)`:
- Calls `Path.resolve()` to eliminate relative dot-dot (`..`) segments and symlink indirections.
- Validates that `target.relative_to(allowed_root)` succeeds. Any attempt to access files outside the project root or artifacts directory raises an immediate `PermissionError`.

### C. Error Message Sanitization
The global FastAPI exception handler intercepts unhandled exceptions:
- Replaces Windows drive paths (`C:\...`, `D:\...`) with `[INTERNAL_PATH]`.
- Replaces database connection strings with `[CONNECTION_STRING]`.
- Logs the full traceback securely to server logs while returning only sanitized error envelopes to clients.

---

## 4. Security Verification
All security controls are verified by automated tests in `tests/test_security_hardening.py`.
