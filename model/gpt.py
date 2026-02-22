"""Simple GPT-style decoder-only transformer for FSM tracing."""

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
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # causal mask: (1, 1, seq, seq) - True means MASKED (can't attend)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T, 3*C) -> 3 * (B, T, C)
        qkv = self.qkv(x)
        # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        out = att @ v
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, 4*d_model) -> (B, T, d_model)
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
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
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
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
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        # (T,) positions
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
        """
        Compute next-token loss on tokens AFTER the first '=' in each example,
        excluding padding positions.

        Args:
            input_ids: (B, T) token indices
            equals_id: token id for '=' — loss starts after this token per example

        Returns:
            loss: scalar
        """
        # (B, T, vocab_size)
        logits = self.forward(input_ids)

        # shift: predict position i+1 from position i
        # (B, T-1, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        # (B, T-1)
        shift_labels = input_ids[:, 1:].contiguous()

        # per-example mask: find first '=' in each row of input_ids
        # (B, T) True where token == equals_id
        eq_mask = input_ids == equals_id
        # first True position per row (if no '=' found, argmax returns 0 — won't happen in valid data)
        # (B,)
        eq_pos = eq_mask.float().argmax(dim=1)

        # (B, T-1) position grid
        positions = torch.arange(shift_labels.size(1), device=input_ids.device).unsqueeze(0)
        # (B, T-1) True where we want loss: positions >= eq_pos (i.e. predict tokens after '=')
        loss_mask = positions >= eq_pos.unsqueeze(1)

        # also exclude padding targets from loss
        # (B, T-1) True where target is NOT pad
        pad_mask = shift_labels != self.pad_id
        # (B, T-1) final mask: after '=' AND not padding
        loss_mask = loss_mask & pad_mask

        # flatten
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        # (B*(T-1),)
        flat_mask = loss_mask.reshape(-1).float()
        return (loss * flat_mask).sum() / flat_mask.sum()
