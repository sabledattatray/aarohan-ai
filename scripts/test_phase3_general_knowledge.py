"""Test script to evaluate current general knowledge and reasoning handling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from aarohan.api import create_app

def main():
    app = create_app()
    client = TestClient(app)

    questions = [
        "What is the capital of India?",
        "What is the capital of Japan?",
        "Who invented the telephone?",
        "What is the chemical formula of water?",
        "How many planets are in our solar system?",
        "Explain SQL joins with examples.",
        "Write a Python program to calculate factorial.",
        "Explain the difference between AI and machine learning.",
    ]

    for q in questions:
        print(f"\n==========================================")
        print(f"QUERY: {q}")
        res = client.post("/api/chat", json={"query": q, "engine": "foundation", "max_new_tokens": 200})
        if res.status_code != 200:
            print(f"FAILED: {res.status_code} - {res.text}")
            continue
        data = res.json()
        print(f"RESPONSE SOURCE: {data.get('response_source')}")
        print(f"INTENT: {data.get('intent')}")
        print(f"MODEL INVOKED: {data.get('model_invoked')}")
        print(f"VERIFICATION STATUS: {data.get('verification', {}).get('status')}")
        retrieved = data.get("retrieved_memory", [])
        print(f"RETRIEVED MEMORY COUNT: {len(retrieved)}")
        if retrieved:
            for r in retrieved[:2]:
                print(f"  - [{r.get('source')}] {r.get('content')[:80]}...")
        print(f"RESPONSE PREVIEW:\n{data.get('response', '')[:300]}")

if __name__ == "__main__":
    main()
