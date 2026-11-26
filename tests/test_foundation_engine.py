"""Tests for AAROHAN Foundation Model Engine and Multi-Engine Chat Routing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from aarohan.api import create_app
from aarohan.foundation_engine import FoundationEngine
from aarohan.memory import MemoryStore
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import sha256_file, train_bpe


@pytest.fixture
def multi_engine_env(tmp_path: Path):
    db_path = tmp_path / "test_foundation.sqlite"
    store = MemoryStore(db_path)
    store.add("AAROHAN platform integrates local open-weights foundation models.", memory_type="semantic", source="test")

    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("AAROHAN is an autonomous AI assistant.\n", encoding="utf-8")
    tok_dir = tmp_path / "tokenizer"
    train_bpe(corpus_file, tok_dir, vocab_size=265)
    tok_path = tok_dir / "tokenizer.json"

    cfg = AAROHANConfig(vocab_size=265, context_length=32, layers=1, heads=2, width=16)
    model = AAROHANLM(cfg)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "model_state": model.state_dict(),
            "tokenizer_sha256": sha256_file(tok_path),
            "training_note": "Test scratch checkpoint",
        },
        ckpt_path,
    )
    reg_path = tmp_path / "deployment_registry.json"

    app = create_app(
        database_path=db_path,
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        registry_path=reg_path,
    )
    client = TestClient(app)
    return {"client": client, "store": store}


def test_foundation_engine_direct_generation():
    engine = FoundationEngine.get_instance()
    res = engine.generate(
        prompt="Reply with exactly the word 'AAROHAN_SUCCESS'",
        max_new_tokens=15,
        temperature=0.1,
    )
    assert "response" in res
    assert len(res["response"]) > 0
    assert res["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_foundation_chat_endpoints(multi_engine_env):
    client = multi_engine_env["client"]

    # 1. Foundation Engine Chat
    res = client.post(
        "/api/chat",
        json={"query": "Hello, what is your name?", "engine": "foundation", "max_new_tokens": 30},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "foundation"
    assert "model_response" in data
    assert len(data["model_response"]["text"]) > 0

    # 2. Scratch Engine Chat
    res_scratch = client.post(
        "/api/chat",
        json={"query": "Hello", "engine": "scratch", "max_new_tokens": 10},
    )
    assert res_scratch.status_code == 200
    assert res_scratch.json()["engine"] == "scratch"

    # 3. Foundation Streaming Chat
    res_stream = client.post(
        "/api/chat/stream",
        json={"query": "What is 2+2?", "engine": "foundation", "max_new_tokens": 20},
    )
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]
    assert "data: " in res_stream.text
