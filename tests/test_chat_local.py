import torch

from aarohan.memory import MemoryStore
from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file, train_bpe
from scripts.chat_local import run_query


def test_local_query_shows_retrieval_and_model_output_separately(tmp_path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text("Causal attention lets each token inspect earlier tokens.\n" * 5, encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_bpe(corpus, tokenizer_dir, vocab_size=300)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    config = AAROHANConfig(vocab_size=tokenizer.get_vocab_size(), context_length=32,
                           layers=1, heads=2, width=16)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({
        "config": config.__dict__,
        "model_state": AAROHANLM(config).state_dict(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "training_note": "unit test fixture",
    }, checkpoint_path)

    database = tmp_path / "memory.sqlite"
    MemoryStore(database).add("Causal attention uses prior tokens.", source="model notes")
    result = run_query("causal attention", checkpoint_path, tokenizer_path, database,
                       max_new_tokens=2)
    retrieved = result["retrieved_memory"]
    assert isinstance(retrieved, list)
    assert retrieved[0]["source"] == "model notes"
    assert retrieved[0]["verification_status"] == "unverified"
    model_output = result["model_output"]
    assert isinstance(model_output, dict)
    assert model_output["checkpoint"] == str(checkpoint_path.resolve())
    assert "not inserted into the model prompt" in str(result["note"])

