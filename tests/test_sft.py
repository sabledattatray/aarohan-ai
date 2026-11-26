import json

from aarohan.chat_template import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    IGNORE_INDEX,
    SYSTEM_TOKEN,
    USER_TOKEN,
    format_dialogue,
    prepare_dialogue_tokens_with_mask,
    validate_dialogue,
)
from aarohan.tokenizer import load_tokenizer, train_bpe
from scripts.chat_sft import generate_chat_response
from scripts.train_sft import train_sft


def test_dialogue_validation_and_formatting() -> None:
    valid_msgs = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User question."},
        {"role": "assistant", "content": "Assistant answer."},
    ]
    validate_dialogue(valid_msgs)
    formatted = format_dialogue(valid_msgs)
    assert SYSTEM_TOKEN in formatted
    assert USER_TOKEN in formatted
    assert ASSISTANT_TOKEN in formatted
    assert EOS_TOKEN in formatted


def test_target_masking_ignores_prompt_tokens(tmp_path) -> None:
    # Train tiny tokenizer
    corpus_file = tmp_path / "train.txt"
    corpus_file.write_text("System prompt.\nUser question.\nAssistant answer.\n" * 10, encoding="utf-8")
    tok_dir = tmp_path / "tokenizer"
    train_bpe(corpus_file, tok_dir, vocab_size=280)
    tok_path = tok_dir / "tokenizer.json"
    tokenizer = load_tokenizer(tok_path)

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User question."},
        {"role": "assistant", "content": "Assistant answer."},
    ]
    tokens, labels, attention_mask = prepare_dialogue_tokens_with_mask(
        tokenizer, messages, context_length=64
    )

    assert tokens.shape == (64,)
    assert labels.shape == (64,)
    assert attention_mask.shape == (64,)

    # Count how many tokens are masked with IGNORE_INDEX
    masked_count = (labels == IGNORE_INDEX).sum().item()
    unmasked_count = (labels != IGNORE_INDEX).sum().item()

    assert unmasked_count > 0, "There must be unmasked target tokens for assistant response"
    assert masked_count > 0, "System and user tokens must be masked with IGNORE_INDEX"


def test_sft_training_reduces_loss_and_saves_checkpoint(tmp_path) -> None:
    # 1. Prepare small tokenizer
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "AAROHAN is an AI assistant.\nWhat is a model?\nA model learns from data.\n" * 20,
        encoding="utf-8",
    )
    tok_dir = tmp_path / "tok"
    train_bpe(corpus_file, tok_dir, vocab_size=280)
    tok_path = tok_dir / "tokenizer.json"

    # 2. Prepare tiny instruction files
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    rec1 = {
        "id": "t1",
        "category": "qa",
        "messages": [
            {"role": "system", "content": "You are AAROHAN."},
            {"role": "user", "content": "What is a model?"},
            {"role": "assistant", "content": "A model learns from data."},
        ],
    }
    train_file.write_text(json.dumps(rec1) + "\n", encoding="utf-8")
    val_file.write_text(json.dumps(rec1) + "\n", encoding="utf-8")

    out_ckpt = tmp_path / "sft.pt"
    result = train_sft(
        train_file=train_file,
        val_file=val_file,
        tokenizer_path=tok_path,
        output_checkpoint=out_ckpt,
        epochs=10,
        batch_size=1,
        lr=1e-3,
        device_name="cpu",
        context_length=32,
    )

    assert result["status"] == "completed"
    assert out_ckpt.is_file()
    assert result["final_train_loss"] < result["initial_train_loss"]

    # 3. Test chat response from the trained SFT checkpoint
    chat_res = generate_chat_response(
        checkpoint_path=out_ckpt,
        tokenizer_path=tok_path,
        messages=[
            {"role": "system", "content": "You are AAROHAN."},
            {"role": "user", "content": "What is a model?"},
        ],
        max_new_tokens=10,
        device_name="cpu",
    )
    assert "response" in chat_res
    assert chat_res["generated_tokens"] >= 0
