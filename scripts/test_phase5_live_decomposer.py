"""Test script for Phase 5 live verification:
- Single rule evaluation (LEVEL-A, LEVEL-B, ON-HOLD, LEVEL-D)
- Multi-question task decomposition
- Arithmetic fast-path
- Identity queries
- Scenario handling without false status hijacking
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000/api/chat"

def query_chat(q: str):
    data = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

test_cases = [
    {
        "name": "Identity Resolution",
        "query": "What is my name?",
        "expected_sub": "Datta Sable",
        "expected_src": "aarohan_identity_resolver",
        "expected_invoked": False,
    },
    {
        "name": "Rule LEVEL-A ACTIVE",
        "query": "What is the contact deadline for LEVEL-A ACTIVE?",
        "expected_sub": "12 hours",
        "expected_src": "aarohan_learning_engine",
        "expected_invoked": False,
    },
    {
        "name": "Rule LEVEL-A ON-HOLD",
        "query": "What is the policy for LEVEL-A ON-HOLD?",
        "expected_sub": "suspended",
        "expected_src": "aarohan_learning_engine",
        "expected_invoked": False,
    },
    {
        "name": "Rule LEVEL-D Undefined",
        "query": "What is the contact deadline for LEVEL-D?",
        "expected_sub": "is not defined in namespace",
        "expected_src": "aarohan_learning_engine",
        "expected_invoked": False,
    },
    {
        "name": "Math Verifier Fast-Path",
        "query": "What is 25 * 4?",
        "expected_sub": "100",
        "expected_src": "aarohan_math_verifier",
        "expected_invoked": False,
    },
    {
        "name": "Compound Multi-Task (Identity + Math + Rule)",
        "query": "1. What is my name?\n2. What is 144 / 12?\n3. What is the contact deadline for LEVEL-B?",
        "expected_sub": "Multi-Task Decomposed",
        "expected_src": "aarohan_multi_task_decomposer",
        "expected_invoked": False,
    },
]

print("=== Running Phase 5 Live Server Acceptance Tests ===")
passed = 0
for tc in test_cases:
    res = query_chat(tc["query"])
    resp_text = res.get("response", "")
    resp_src = res.get("response_source", "")
    invoked = res.get("model_invoked")

    sub_ok = tc["expected_sub"].lower() in resp_text.lower()
    src_ok = resp_src == tc["expected_src"]
    inv_ok = invoked == tc["expected_invoked"]

    status = "PASS" if (sub_ok and src_ok and inv_ok) else "FAIL"
    if status == "PASS":
        passed += 1
    print(f"[{status}] {tc['name']}")
    print(f"       Source: {resp_src} (Expected: {tc['expected_src']})")
    print(f"       Model Invoked: {invoked} (Expected: {tc['expected_invoked']})")
    print(f"       Sub-string Match: {sub_ok}")
    if status == "FAIL":
        print(f"       Response snippet: {resp_text[:150]}...")

print(f"\nSummary: {passed}/{len(test_cases)} tests passed.")
