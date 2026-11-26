"""
Tests for Conversation Session Grouping, Multi-Turn History, and Session Restoration.
Ensures conversations are properly grouped by session_id, turns are preserved chronologically,
and individual sessions can be retrieved and deleted cleanly.
"""

from fastapi.testclient import TestClient

from aarohan.api import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer aarohan_admin_token"}


def test_conversation_grouping_and_restoration():
    session_a = "session_test_grp_aaa"
    session_b = "session_test_grp_bbb"

    # Turn 1 in Session A
    resp1 = client.post(
        "/api/chat",
        json={"query": "Hello AAROHAN, my favorite programming language is Python", "session_id": session_a},
        headers=AUTH_HEADERS,
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["session_id"] == session_a

    # Turn 2 in Session A (continuation)
    resp2 = client.post(
        "/api/chat",
        json={"query": "Can you explain list comprehensions in Python?", "session_id": session_a},
        headers=AUTH_HEADERS,
    )
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == session_a

    # Turn 1 in Session B (separate conversation)
    resp3 = client.post(
        "/api/chat",
        json={"query": "What is the capital of France?", "session_id": session_b},
        headers=AUTH_HEADERS,
    )
    assert resp3.status_code == 200
    assert resp3.json()["session_id"] == session_b

    # 1. Fetch grouped conversations
    convs_resp = client.get("/api/conversations?limit=30", headers=AUTH_HEADERS)
    assert convs_resp.status_code == 200
    convs = convs_resp.json().get("conversations", [])

    # Find session A and session B in the list
    session_a_entry = next((c for c in convs if c.get("session_id") == session_a), None)
    session_b_entry = next((c for c in convs if c.get("session_id") == session_b), None)

    assert session_a_entry is not None, f"Session {session_a} not found in grouped conversations"
    assert session_b_entry is not None, f"Session {session_b} not found in grouped conversations"

    # Verify session A grouped 2 turns
    assert session_a_entry["turns_count"] >= 2
    assert len(session_a_entry["turns"]) >= 2
    assert "Python" in session_a_entry["title"]

    # Verify session B has 1 turn
    assert session_b_entry["turns_count"] >= 1
    assert "France" in session_b_entry["title"]

    # 2. Fetch specific session details via GET /api/conversations/{session_id}
    detail_resp = client.get(f"/api/conversations/{session_a}", headers=AUTH_HEADERS)
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["session_id"] == session_a
    assert detail["turns_count"] >= 2
    assert len(detail["turns"]) >= 2
    # Verify chronological order of turns
    assert "Python" in detail["turns"][0]["user"]
    assert "list comprehensions" in detail["turns"][1]["user"]

    # 3. Delete session B via DELETE /api/conversations/{session_id}
    del_resp = client.delete(f"/api/conversations/{session_b}", headers=AUTH_HEADERS)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_count"] >= 1

    # Verify session B is deleted
    del_check = client.get(f"/api/conversations/{session_b}", headers=AUTH_HEADERS)
    assert del_check.status_code == 404

    # Cleanup session A
    client.delete(f"/api/conversations/{session_a}", headers=AUTH_HEADERS)
