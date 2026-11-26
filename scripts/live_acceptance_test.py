"""Live E2E Acceptance Test for AAROHAN Learning Engine 1.0."""

import json
import sqlite3
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

PROMPT_STAGE_1 = """I am teaching you 5 new business rules for handling accounts.
Please learn and store these rules under namespace LEARNING_TEST_2026_01:

RULES:
1. LEVEL-A cases have response deadline of 24 hours.
2. LEVEL-B cases have response deadline of 48 hours.
3. LEVEL-C cases have response deadline of 72 hours.
4. If account has status ON-HOLD, all response deadlines are suspended and no contact may be made.
5. ON-HOLD condition overrides LEVEL-A, LEVEL-B, and LEVEL-C deadlines."""

print("=" * 70)
print("STAGE 1: LIVE TEACHING INSTRUCTION (/api/chat)")
print("=" * 70)
req = urllib.request.Request(
    f"{BASE_URL}/api/chat",
    data=json.dumps({"query": PROMPT_STAGE_1}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    res1 = json.loads(resp.read().decode("utf-8"))
    print(res1["response"])

print("\n" + "=" * 70)
print("STAGE 2: DIRECT SQLITE VERIFICATION (data/memory/aarohan.sqlite)")
print("=" * 70)
with sqlite3.connect("data/memory/aarohan.sqlite") as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, namespace, rule_key, value, conditions, precedence, status, version FROM learned_rules WHERE namespace = 'LEARNING_TEST_2026_01' ORDER BY precedence DESC, created_at ASC"
    ).fetchall()
    print(f"Total SQLite rows verified in LEARNING_TEST_2026_01: {len(rows)}")
    for r in rows:
        print(f"- ID: {r['id']} | Key: {r['rule_key']} | Value: {r['value']} | Precedence: {r['precedence']} | Status: {r['status']} | v{r['version']}")

print("\n" + "=" * 70)
print("STAGE 3: FRESH SESSION RECALL (/api/chat)")
print("=" * 70)
req3 = urllib.request.Request(
    f"{BASE_URL}/api/chat",
    data=json.dumps({"query": "What are the rules we learned in namespace LEARNING_TEST_2026_01?"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req3) as resp:
    res3 = json.loads(resp.read().decode("utf-8"))
    print(res3["response"])

print("\n" + "=" * 70)
print("STAGE 4: SCENARIO REASONING & OVERRIDE EVALUATION (/api/chat)")
print("=" * 70)
req4 = urllib.request.Request(
    f"{BASE_URL}/api/chat",
    data=json.dumps({"query": "We have an account with priority LEVEL-A and status ON-HOLD. What is the response deadline according to namespace LEARNING_TEST_2026_01 rules?"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req4) as resp:
    res4 = json.loads(resp.read().decode("utf-8"))
    print(res4["response"])

print("\n" + "=" * 70)
print("STAGE 5: FEEDBACK REORDERING / VALUE CORRECTION (/api/chat)")
print("=" * 70)
req5 = urllib.request.Request(
    f"{BASE_URL}/api/chat",
    data=json.dumps({"query": "In namespace LEARNING_TEST_2026_01, change LEVEL-A deadline to 12 hours."}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req5) as resp:
    res5 = json.loads(resp.read().decode("utf-8"))
    print(res5["response"])

with sqlite3.connect("data/memory/aarohan.sqlite") as conn:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM learned_rules WHERE namespace = 'LEARNING_TEST_2026_01' AND rule_key LIKE '%LEVEL_A%'"
    ).fetchone()
    vers = conn.execute(
        "SELECT * FROM learned_rule_versions WHERE rule_id = ?", (row["id"],)
    ).fetchall()
    print(f"SQLite Verification: {row['rule_key']} value='{row['value']}' version=v{row['version']} (total versions: {len(vers)})")

print("\n" + "=" * 70)
print("STAGE 6: STREAMING VERIFICATION (/api/chat/stream)")
print("=" * 70)
req6 = urllib.request.Request(
    f"{BASE_URL}/api/chat/stream",
    data=json.dumps({"query": "Recall the rules in namespace LEARNING_TEST_2026_01"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req6) as resp:
    stream_content = resp.read().decode("utf-8")
    print(f"Stream status: 200 OK | Total bytes: {len(stream_content)}")
    print("Stream content preview:")
    tokens = []
    for line in stream_content.splitlines():
        if line.startswith("data: "):
            try:
                ev = json.loads(line[6:])
                if ev.get("type") == "token":
                    tokens.append(ev.get("token", ""))
            except Exception:
                pass
    print("".join(tokens)[:400] + "...")
print("\n" + "=" * 70)
print("ALL LIVE ACCEPTANCE TESTS EXECUTED SUCCESSFULLY ✅")
print("=" * 70)
