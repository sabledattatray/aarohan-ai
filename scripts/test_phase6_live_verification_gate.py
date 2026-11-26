"""Phase 6 Live Server Acceptance Test for Output Verification Gate."""

import urllib.request
import json

URL = "http://127.0.0.1:8000/api/chat"

test_cases = [
    {
        "name": "Identity Verification",
        "query": "Who is Datta Sable?",
        "expected_status": "VERIFIED",
        "expected_truthful": True,
        "expected_sub": "dattasable.com",
    },
    {
        "name": "Rule Verification LEVEL-A",
        "query": "What is the contact deadline for LEVEL-A ACTIVE?",
        "expected_status": "VERIFIED",
        "expected_truthful": True,
        "expected_sub": "12 hours",
    },
    {
        "name": "Math Verification",
        "query": "What is 50 * 2?",
        "expected_status": "VERIFIED",
        "expected_truthful": True,
        "expected_sub": "100",
    },
    {
        "name": "Compound Multi-Task Verification",
        "query": "1. What is my name?\n2. What is 10 + 15?",
        "expected_status": "VERIFIED",
        "expected_truthful": True,
        "expected_sub": "25",
    },
]

print("=== Running Phase 6 Output Verification Gate Live Server Tests ===")
passed = 0
for tc in test_cases:
    data = json.dumps({"query": tc["query"]}).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        
    v = res.get("verification", {})
    status_ok = v.get("status") == tc["expected_status"]
    truth_ok = v.get("is_truthful") == tc["expected_truthful"]
    sub_ok = tc["expected_sub"].lower() in res.get("response", "").lower()
    
    ok = status_ok and truth_ok and sub_ok
    if ok:
        passed += 1
    
    badge = "PASS" if ok else "FAIL"
    print(f"[{badge}] {tc['name']}")
    print(f"       Verification Status: {v.get('status')} (Expected: {tc['expected_status']})")
    print(f"       Is Truthful: {v.get('is_truthful')} (Expected: {tc['expected_truthful']})")
    print(f"       Grounding Score: {v.get('grounding_score')}")
    print(f"       Contradictions: {v.get('contradictions_detected')}")
    print(f"       Sub-match: {sub_ok}")

print(f"\nSummary: {passed}/{len(test_cases)} tests passed.")
