import json
import urllib.request

test_queries = [
    "What is the contact deadline for LEVEL-A account?",
    "What is the deadline for LEVEL-B?",
    "What is the deadline for LEVEL-C?",
    "What action should be taken for a LEVEL-A account that is ON-HOLD?",
    "What is the deadline for LEVEL-D?",
    "Account is LEVEL-A and status is PENDING_REVIEW. What is the deadline?",
]

print("=================== POST /api/chat ===================")
for q in test_queries:
    payload = json.dumps({"query": q, "engine": "foundation"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        mr = res.get("model_response", {})
        print(f"QUERY: {q}")
        print(f"CHECKPOINT: {mr.get('checkpoint')}")
        print(f"TOKENS: {mr.get('generated_tokens')}")
        first_lines = res.get("response", "").splitlines()[:5]
        print("RESPONSE SNIPPET: " + " | ".join(first_lines))
        print("------------------------------------------------------")

print("\n=================== POST /api/chat/stream ===================")
for q in test_queries:
    payload = json.dumps({"query": q, "engine": "foundation"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        tokens = []
        done_data = None
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                event = json.loads(line_str[6:])
                if event.get("type") == "token":
                    tokens.append(event.get("token", ""))
                elif event.get("type") == "done":
                    done_data = event
        full_stream_text = "".join(tokens)
        print(f"STREAM QUERY: {q}")
        print(f"STREAM DONE: {done_data}")
        first_lines = full_stream_text.splitlines()[:5]
        print("STREAM RESPONSE SNIPPET: " + " | ".join(first_lines))
        print("------------------------------------------------------")
