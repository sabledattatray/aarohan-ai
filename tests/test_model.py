import torch

from aarohan.model import AAROHANLM, AAROHANConfig


def test_forward_backward_and_parameter_count() -> None:
    model = AAROHANLM(AAROHANConfig(vocab_size=32, context_length=16, layers=2, heads=2, width=32))
    tokens = torch.randint(0, 32, (2, 12))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 12, 32)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.parameter_count() == sum(parameter.numel() for parameter in model.parameters())


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(17)
    model = AAROHANLM(AAROHANConfig(vocab_size=32, context_length=16, layers=2, heads=2, width=32))
    model.eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad():
        first_logits, _ = model(first)
        changed_logits, _ = model(changed_future)
    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2], rtol=0, atol=1e-6)


def test_generation_respects_context_and_restores_training_state() -> None:
    model = AAROHANLM(AAROHANConfig(vocab_size=32, context_length=8, layers=1, heads=2, width=16))
    model.train()
    result = model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, top_k=4, top_p=0.9)
    assert result.shape == (1, 5)
    assert model.training


def test_same_seed_gives_same_initial_parameters() -> None:
    config = AAROHANConfig(vocab_size=16, context_length=8, layers=1, heads=2, width=16)
    torch.manual_seed(2026)
    first = AAROHANLM(config)
    torch.manual_seed(2026)
    second = AAROHANLM(config)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right)

