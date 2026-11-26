"""Small, original decoder-only Transformer used for AAROHAN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AAROHANConfig:
    vocab_size: int = 128
    context_length: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.context_length < 1:
            raise ValueError("vocab_size must be >= 2 and context_length >= 1")
        if self.layers < 1 or self.heads < 1 or self.width < 1:
            raise ValueError("layers, heads, and width must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (self.width // self.heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


def _rotate_half(x: Tensor) -> Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: AAROHANConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.dropout = config.dropout
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        positions = torch.arange(length, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
        q = q * angles.cos().to(dtype=q.dtype) + _rotate_half(q) * angles.sin().to(dtype=q.dtype)
        k = k * angles.cos().to(dtype=k.dtype) + _rotate_half(k) * angles.sin().to(dtype=k.dtype)

        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        hidden = 4 * width
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.output(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: AAROHANConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = RMSNorm(config.width)
        self.feedforward = SwiGLU(config.width, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class AAROHANLM(nn.Module):
    """Causal language model with freshly initialized, project-owned weights."""

    def __init__(self, config: AAROHANConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.final_norm = RMSNorm(config.width)
        self.language_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # Weight tying reduces parameters and is initialized by the embedding initializer.
        self.language_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds configured context_length")
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative and temperature positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.config.context_length :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                    filtered = torch.full_like(logits, -torch.inf)
                    logits = filtered.scatter(1, sorted_indices, sorted_logits)
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                token_ids = torch.cat((token_ids, next_token), dim=1)
        finally:
            self.train(was_training)
        return token_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)

