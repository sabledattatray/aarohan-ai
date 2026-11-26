import torch

from aarohan.model import AAROHANLM, AAROHANConfig
from aarohan.tokenizer import load_tokenizer, sha256_file, train_bpe
from scripts.generate_local import generate


def test_local_generation_loads_matching_checkpoint_and_tokenizer(tmp_path) -> None:
    train_file = tmp_path / "train.txt"
    train_file.write_text("A little robot crosses the garden.\n" * 8, encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_bpe(train_file, tokenizer_dir, vocab_size=300)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    vocab_size = load_tokenizer(tokenizer_path).get_vocab_size()
    config = AAROHANConfig(vocab_size=vocab_size, context_length=32, layers=1, heads=2, width=16)
    model = AAROHANLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({
        "config": model.config_dict(),
        "model_state": model.state_dict(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "training_note": "tiny toy corpus test fixture",
    }, checkpoint_path)

    result = generate(checkpoint_path, tokenizer_path, "A little", max_new_tokens=3)
    assert result["generated_token_count"] == 3
    assert result["prompt"] == "A little"
    assert "tiny AI-generated toy corpus" in str(result["warning"])


def test_local_generation_rejects_tokenizer_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("not a tokenizer", encoding="utf-8")
    torch.save({"tokenizer_sha256": "wrong"}, checkpoint_path)
    try:
        generate(checkpoint_path, tokenizer_path, "test", max_new_tokens=1)
    except ValueError as error:
        assert "Tokenizer hash does not match" in str(error)
    else:
        raise AssertionError("Expected a tokenizer/checkpoint mismatch to be rejected")

