"""Live test for user memory save and recall."""

import json
import time
import urllib.request
import sys

API_BASE = "http://127.0.0.1:8000"
TOKEN = "aarohan_researcher_token"

def wait_for_server():
    for i in range(20):
        try:
            with urllib.request.urlopen(f"{API_BASE}/api/health", timeout=2) as r:
                data = json.loads(r.read().decode())
                if data.get("status") == "healthy":
                    print("Server is healthy.")
                    return
        except Exception:
            pass
        time.sleep(1)
    print("Server failed to respond.")
    sys.exit(1)

def chat(query: str):
    payload = json.dumps({"query": query, "engine": "foundation"}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def main():
    wait_for_server()

    # 1. Test saving personal fact
    save_q = "This Is Datta Sable, keep in your memory that i am Stays at Badlapur East"
    print(f"\n--- Testing Memory Save: '{save_q}' ---")
    res1 = chat(save_q)
    print("Response Source:", res1.get("response_source"))
    print("Model Invoked:", res1.get("model_invoked"))
    print("Response Text:\n", res1.get("response"))
    assert res1.get("response_source") == "aarohan_memory_engine", f"Expected aarohan_memory_engine, got {res1.get('response_source')}"
    assert "Badlapur East" in res1.get("response"), "Expected Badlapur East in confirmation"

    # 2. Test recalling location via identity/location query
    recall_q1 = "Where do I stay?"
    print(f"\n--- Testing Recall Query 1: '{recall_q1}' ---")
    res2 = chat(recall_q1)
    print("Response Source:", res2.get("response_source"))
    print("Model Invoked:", res2.get("model_invoked"))
    print("Response Text:\n", res2.get("response"))
    assert "Badlapur East" in res2.get("response"), f"Expected Badlapur East, got: {res2.get('response')}"

    # 3. Test recalling location query 2
    recall_q2 = "what is my location"
    print(f"\n--- Testing Recall Query 2: '{recall_q2}' ---")
    res3 = chat(recall_q2)
    print("Response Source:", res3.get("response_source"))
    print("Model Invoked:", res3.get("model_invoked"))
    print("Response Text:\n", res3.get("response"))
    assert "Badlapur East" in res3.get("response"), f"Expected Badlapur East, got: {res3.get('response')}"

    print("\n✅ ALL MEMORY SAVE & RECALL ACCEPTANCE TESTS PASSED!")

if __name__ == "__main__":
    main()
