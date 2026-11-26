import json
import urllib.request

test_cases = [
    # 1. Identity edge cases
    ("Tell me everything you know about me, Datta Sable.", "identity_resolver", False),
    ("Who is talking to you?", "identity_resolver", False),
    ("Who am I chatting with?", "identity_resolver", False),
    ("What model are you running?", "identity_resolver", False),
    ("Datta Sable's Website is https://dattasable.com/", "identity_resolver", False),
    ("my website?", "identity_resolver", False),
    ("what is my professional background", "identity_resolver", False),
    # 2. Deterministic rule query
    ("What is the contact deadline for LEVEL-A account?", "learning_engine", False),
    # 3. Model query
    ("Write a creative two-line poem about space.", "foundation_engine", True),
]

print("=================== PHASE 4: POST /api/chat VERIFICATION ===================")
for q, expected_source, expected_invoked in test_cases:
    payload = json.dumps({"query": q, "engine": "foundation", "max_new_tokens": 50}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode())
        source = res.get("response_source")
        invoked = res.get("model_invoked")
        mr = res.get("model_response", {})
        tokens = mr.get("generated_tokens")
        ckpt = mr.get("checkpoint")
        text = res.get("response", "")

        src_ok = (source == expected_source)
        inv_ok = (invoked == expected_invoked)

        print(f"QUERY: {q}")
        print(f"  SOURCE: {source} (Expected: {expected_source}) -> {'PASS' if src_ok else 'FAIL'}")
        print(f"  INVOKED: {invoked} (Expected: {expected_invoked}) -> {'PASS' if inv_ok else 'FAIL'}")
        print(f"  CHECKPOINT: {ckpt} | TOKENS: {tokens}")
        first_line = text.splitlines()[0] if text else ""
        print(f"  SNIPPET: {first_line[:100]}")
        print("-------------------------------------------------------------------------")

print("\n=================== PHASE 4: POST /api/chat/stream VERIFICATION ===================")
stream_cases = [
    ("Tell me everything you know about me, Datta Sable.", "identity_resolver", False),
    ("What is the contact deadline for LEVEL-A account?", "learning_engine", False),
    ("Write a creative two-line poem about space.", "foundation_engine", True),
]

for q, expected_source, expected_invoked in stream_cases:
    payload = json.dumps({"query": q, "engine": "foundation", "max_new_tokens": 50}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        tokens = []
        init_event = None
        done_event = None
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                event = json.loads(line_str[6:])
                if event.get("type") == "context":
                    init_event = event
                elif event.get("type") == "token":
                    tokens.append(event.get("token", ""))
                elif event.get("type") == "done":
                    done_event = event

        full_text = "".join(tokens)
        source = done_event.get("response_source") if done_event else None
        invoked = done_event.get("model_invoked") if done_event else None
        tokens_count = done_event.get("generated_tokens") if done_event else None
        ckpt = done_event.get("checkpoint") if done_event else None

        src_ok = (source == expected_source)
        inv_ok = (invoked == expected_invoked)

        print(f"STREAM QUERY: {q}")
        print(f"  SOURCE: {source} (Expected: {expected_source}) -> {'PASS' if src_ok else 'FAIL'}")
        print(f"  INVOKED: {invoked} (Expected: {expected_invoked}) -> {'PASS' if inv_ok else 'FAIL'}")
        print(f"  CHECKPOINT: {ckpt} | TOKENS: {tokens_count}")
        first_line = full_text.splitlines()[0] if full_text else ""
        print(f"  SNIPPET: {first_line[:100]}")
        print("-------------------------------------------------------------------------")
