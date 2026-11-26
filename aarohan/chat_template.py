"""Chat template formatting and target masking for Supervised Fine-Tuning (SFT)."""

from __future__ import annotations

from typing import Any
import torch

from aarohan.tokenizer import SPECIAL_TOKENS

SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
IGNORE_INDEX = -100


def validate_dialogue(messages: list[dict[str, str]]) -> None:
    if not messages:
        raise ValueError("Dialogue cannot be empty")
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Must be one of {valid_roles}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be non-empty string")


def format_dialogue(messages: list[dict[str, str]]) -> str:
    """Format structured messages into a standardized chat prompt string."""
    validate_dialogue(messages)
    formatted_parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            formatted_parts.append(f"{SYSTEM_TOKEN}\n{content}\n")
        elif role == "user":
            formatted_parts.append(f"{USER_TOKEN}\n{content}\n")
        elif role == "assistant":
            formatted_parts.append(f"{ASSISTANT_TOKEN}\n{content}{EOS_TOKEN}\n")
    return "".join(formatted_parts)


def prepare_dialogue_tokens_with_mask(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    context_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize dialogue and produce input_ids, labels (target-masked), and attention_mask.
    
    Tokens corresponding to system messages, user queries, and role headers are masked
    with IGNORE_INDEX (-100) so that loss is computed ONLY on assistant responses and <|eos|>.
    """
    validate_dialogue(messages)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN) if tokenizer.token_to_id(PAD_TOKEN) is not None else 0

    all_tokens: list[int] = []
    all_labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if role == "system":
            header_str = f"{SYSTEM_TOKEN}\n{content}\n"
            t_ids = tokenizer.encode(header_str, add_special_tokens=False).ids
            all_tokens.extend(t_ids)
            all_labels.extend([IGNORE_INDEX] * len(t_ids))
        elif role == "user":
            header_str = f"{USER_TOKEN}\n{content}\n"
            t_ids = tokenizer.encode(header_str, add_special_tokens=False).ids
            all_tokens.extend(t_ids)
            all_labels.extend([IGNORE_INDEX] * len(t_ids))
        elif role == "assistant":
            prefix_str = f"{ASSISTANT_TOKEN}\n"
            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False).ids
            all_tokens.extend(prefix_ids)
            all_labels.extend([IGNORE_INDEX] * len(prefix_ids))

            content_ids = tokenizer.encode(content, add_special_tokens=False).ids
            # Assistant content is unmasked (the model must learn to predict this)
            all_tokens.extend(content_ids)
            all_labels.extend(content_ids)

            # Include EOS token as target so model learns when to stop
            if eos_id is not None:
                all_tokens.append(eos_id)
                all_labels.append(eos_id)

    seq_len = len(all_tokens)
    if seq_len > context_length:
        all_tokens = all_tokens[:context_length]
        all_labels = all_labels[:context_length]
        attention_mask = [1] * context_length
    else:
        padding_needed = context_length - seq_len
        attention_mask = [1] * seq_len + [0] * padding_needed
        all_tokens = all_tokens + [pad_id] * padding_needed
        all_labels = all_labels + [IGNORE_INDEX] * padding_needed

    return (
        torch.tensor(all_tokens, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )
