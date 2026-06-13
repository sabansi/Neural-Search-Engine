"""Transformer text encoder.

A bi-encoder built directly from plain PyTorch primitives
(nn.Linear / nn.Embedding / nn.LayerNorm) — the attention, transformer block
and pooling are implemented here rather than imported from a model library.

Architecture (bi-encoder with shared weights for queries and documents):

    input_ids
        │  token embedding + learned positional embedding
        ▼
    N × TransformerBlock          pre-LayerNorm, manual multi-head attention
        ▼
    final LayerNorm
        ▼
    mean pooling over non-padding tokens
        ▼
    Linear(d_model → proj_dim, no bias)
        ▼
    L2 normalisation               → unit-norm sentence embedding
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 1536
    max_len: int = 320
    dropout: float = 0.1
    proj_dim: int = 256
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention, written out explicitly."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Project to queries/keys/values and split into heads: (B, H, T, d_head)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)
        # Padding positions must never receive attention (as keys).
        key_mask = attention_mask[:, None, None, :] == 0  # (B, 1, 1, T)
        scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        ctx = attn @ v  # (B, H, T, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block (more stable than post-LN for randomly-initialised training)."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.proj_dim, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Contextual token representations, (B, T, d_model). Used by the MLM head."""
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.cfg.max_len}")
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_ln(x)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Unit-norm sentence embeddings, (B, proj_dim)."""
        h = self.hidden_states(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLMHead(nn.Module):
    """BERT-style masked-language-modelling head with the decoder tied to the
    token embedding matrix (saves ~11.5M parameters and trains better)."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        d = encoder.cfg.d_model
        self.dense = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(encoder.cfg.vocab_size))
        # Wrapped in a tuple so the shared embedding is NOT registered as a
        # submodule — otherwise its weights would show up twice in the optimizer.
        self._tied_embedding = (encoder.token_emb,)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.ln(F.gelu(self.dense(hidden)))
        return h @ self._tied_embedding[0].weight.t() + self.bias


def save_encoder(path, encoder: Encoder, extra: dict | None = None) -> None:
    payload = {"config": encoder.cfg.to_dict(), "model_state": encoder.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_encoder(path, map_location="cpu") -> Encoder:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    encoder = Encoder(cfg)
    state = payload.get("model_state") or payload["encoder_state"]
    encoder.load_state_dict(state)
    return encoder
