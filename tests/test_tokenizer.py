import json

import pytest

from aarohan.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_bpe


def test_bpe_round_trip_unicode_and_special_tokens(tmp_path) -> None:
    train_path = tmp_path / "train.txt"
    train_path.write_text("Hello, world!\nनमस्ते दुनिया\nπ = 3.14\n", encoding="utf-8")
    output = tmp_path / "tokenizer"
    metadata = train_bpe(train_path, output, vocab_size=300)
    tokenizer = load_tokenizer(output / "tokenizer.json")

    text = "Hello, world! नमस्ते π"
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded.ids) == text
    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) is not None
        special_encoding = tokenizer.encode(token)
        assert special_encoding.tokens == [token]
        assert tokenizer.decode(special_encoding.ids, skip_special_tokens=False) == token
    saved_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_file_sha256"] == saved_metadata["training_file_sha256"]
    assert saved_metadata["validation_data_used_for_training"] is False


def test_bpe_refuses_empty_training_data(tmp_path) -> None:
    train_path = tmp_path / "empty.txt"
    train_path.write_text("# comment only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Training corpus is empty"):
        train_bpe(train_path, tmp_path / "tokenizer")

