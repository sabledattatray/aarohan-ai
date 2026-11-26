"""Phase 7 Live Web Search Acceptance Tests.

Verifies that AAROHAN's /api/chat endpoint:
1. Routes web-search queries to aarohan_web_search (not the neural model)
2. Returns live DuckDuckGo results with real URLs
3. Does NOT invoke the neural model for web search queries
4. Does NOT break existing identity / rule routing
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

API_BASE = "http://127.0.0.1:8000"
TOKEN = "aarohan_researcher_token"


def _post_chat(query: str) -> dict:
    payload = json.dumps({"query": query, "engine": "foundation"}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _assert(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    mark = "✅" if condition else "❌"
    print(f"  {mark} [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


def wait_for_server(max_retries: int = 20, delay: float = 2.0) -> None:
    print("⏳ Waiting for server to be ready …")
    for i in range(max_retries):
        try:
            with urllib.request.urlopen(f"{API_BASE}/api/health", timeout=5) as r:
                data = json.loads(r.read().decode())
                if data.get("status") == "healthy":
                    print(f"  ✅ Server healthy after {i * delay:.0f}s\n")
                    return
        except Exception:
            pass
        time.sleep(delay)
    print("  ❌ Server never became healthy — aborting")
    sys.exit(1)


test_cases = [
    {
        "id": "T7-01",
        "desc": "Self-search via 'search me on web'",
        "query": "search me on web",
        "expect_source": "aarohan_web_search",
        "expect_model_invoked": False,
        "expect_in_response": ["Live Web Search", "Datta Sable"],
    },
    {
        "id": "T7-02",
        "desc": "Self-search via 'search about me online'",
        "query": "search about me online",
        "expect_source": "aarohan_web_search",
        "expect_model_invoked": False,
        "expect_in_response": ["Live Web Search"],
    },
    {
        "id": "T7-03",
        "desc": "Explicit topic search — Power BI",
        "query": "search the web for Power BI dashboard tutorial",
        "expect_source": "aarohan_web_search",
        "expect_model_invoked": False,
        "expect_in_response": ["Live Web Search"],
    },
    {
        "id": "T7-04",
        "desc": "Non-search identity query must NOT hit web search",
        "query": "What is my name?",
        "expect_source": "aarohan_identity_resolver",
        "expect_model_invoked": False,
        "expect_in_response": ["Datta Sable"],
    },
    {
        "id": "T7-05",
        "desc": "Rule query must NOT hit web search",
        "query": "What is the contact deadline for LEVEL-A?",
        "expect_source": "aarohan_learning_engine",
        "expect_model_invoked": False,
        "expect_in_response": ["12"],
    },
]


def main() -> None:
    wait_for_server()
    total = len(test_cases)
    passed = 0
    failed = 0

    for tc in test_cases:
        print(f"🔍 {tc['id']}: {tc['desc']}")
        print(f"   Query: \"{tc['query']}\"")
        try:
            resp = _post_chat(tc["query"])
            source = resp.get("response_source", "")
            model_inv = resp.get("model_invoked", True)
            response_text = resp.get("response", "")

            source_ok = source == tc["expect_source"]
            model_ok = model_inv == tc["expect_model_invoked"]
            content_ok = all(term in response_text for term in tc.get("expect_in_response", []))

            _assert(
                f"response_source == '{tc['expect_source']}'",
                source_ok,
                f"got '{source}'",
            )
            _assert(
                f"model_invoked == {tc['expect_model_invoked']}",
                model_ok,
                f"got {model_inv}",
            )
            _assert(
                "response contains expected terms",
                content_ok,
                f"missing from: {response_text[:200]}",
            )

            print(f"   Source: {source} | Model invoked: {model_inv}")
            print(f"   Response snippet: {response_text[:160].strip()!r}\n")
            passed += 1
        except Exception as e:
            print(f"   ❌ Exception: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Phase 7 Live Web Search Results: {passed}/{total} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("✅ All Phase 7 acceptance tests PASSED")


if __name__ == "__main__":
    main()
