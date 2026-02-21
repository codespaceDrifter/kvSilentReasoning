"""Pure attention transformer - no FFW/MLP layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # no output projection (O) - pure QKV only
        self.dropout = nn.Dropout(dropout)

        # causal mask: True = masked (can't attend)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T, 3*C) -> 3 * (B, T, C)
        qkv = self.qkv(x)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # (B, n_heads, T, head_dim) -> (B, T, C)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out


class PureAttentionBlock(nn.Module):
    """Transformer block with attention only, no MLP."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm, no MLP
        x = x + self.attn(self.ln(x))
        return x


class PureAttentionGPT(nn.Module):
    """GPT with attention only, no FFW/MLP layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 72,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PureAttentionBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        positions = torch.arange(T, device=input_ids.device)

        # (B, T, d_model)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        return logits

    def compute_loss(self, input_ids: torch.Tensor, equals_id: int = 15) -> torch.Tensor:
        """Loss on tokens after first '=' per example, excluding padding."""
        # (B, T, vocab_size)
        logits = self.forward(input_ids)

        # shift for next-token prediction
        # (B, T-1, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        # (B, T-1)
        shift_labels = input_ids[:, 1:].contiguous()

        # per-example: find first '=' position
        # (B,)
        eq_pos = (input_ids == equals_id).float().argmax(dim=1)

        # (B, T-1) True where position >= eq_pos (predict tokens after '=')
        positions = torch.arange(shift_labels.size(1), device=input_ids.device).unsqueeze(0)
        loss_mask = positions >= eq_pos.unsqueeze(1)

        # exclude padding targets
        loss_mask = loss_mask & (shift_labels != self.pad_id)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        flat_mask = loss_mask.reshape(-1).float()
        return (loss * flat_mask).sum() / flat_mask.sum()
