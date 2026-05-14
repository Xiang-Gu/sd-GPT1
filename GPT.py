import math

import torch
from torch import nn
import torch.nn.functional as F


def build_sinusoidal_positional_embeddings(
    max_seq_len, d_model, device=None, dtype=None
):
    positions = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.empty((max_seq_len, d_model), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe


class PositionalEmbeddings:
    @staticmethod
    def forward(x):
        """
        Return sinusoidal positional embeddings for x.

        x.shape = (B, L, d_model)
        return.shape = (L, d_model), broadcastable over the batch dimension.
        """
        _, L, d_model = x.shape
        return build_sinusoidal_positional_embeddings(
            L, d_model, device=x.device, dtype=x.dtype
        )


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Normalize each token vector over the last dimension."""
        return self.layer_norm(x)


class MHA(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.d_model = d_model
        self.d_qkv = d_model // h
        self.dropout = dropout

        self.QKV_linear = nn.Linear(d_model, 3 * d_model)
        self.O_linear = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Causal multi-head self-attention.

        x.shape = (B, L, d_model)
        return.shape = (B, L, d_model)
        """
        qkv = self.QKV_linear(x)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)

        Q, K, V = self.split_heads(Q, K, V)
        out = self.attn(Q, K, V)
        out = self.combine_heads(out)
        out = self.O_linear(out)
        return self.out_dropout(out)

    def attn(self, Q, K, V):
        # Q, K, V shape = (B, h, L, d_qkv)
        score = Q @ K.transpose(-1, -2) / math.sqrt(self.d_qkv)

        L = score.shape[-1]
        causal_mask = torch.triu(
            torch.ones((L, L), device=score.device, dtype=torch.bool), diagonal=1
        )
        score = score.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(score, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return attn @ V

    def split_heads(self, Q, K, V):
        assert Q.shape == K.shape == V.shape
        B, L, d_model = Q.shape
        assert d_model == self.d_model

        Q = Q.reshape(B, L, self.h, self.d_qkv).permute(0, 2, 1, 3)
        K = K.reshape(B, L, self.h, self.d_qkv).permute(0, 2, 1, 3)
        V = V.reshape(B, L, self.h, self.d_qkv).permute(0, 2, 1, 3)
        return Q, K, V

    def combine_heads(self, out):
        B, h, L, d_qkv = out.shape
        assert h == self.h and d_qkv == self.d_qkv
        return out.permute(0, 2, 1, 3).reshape(B, L, self.d_model)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Position-wise feed-forward network."""
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return self.dropout(x)


class AttnFFNBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.MHA = MHA(d_model, h, dropout=dropout)
        self.FFN = FFN(d_model, d_ff, dropout=dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        x = x + self.MHA(self.ln1(x))
        x = x + self.FFN(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        N,
        d_model,
        d_ff,
        h,
        max_seq_len=None,
        dropout=0.1,
        weight_tying=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.embeds = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [AttnFFNBlock(d_model, h, d_ff, dropout=dropout) for _ in range(N)]
        )
        self.final_linear = nn.Linear(d_model, vocab_size, bias=False)
        if weight_tying:
            self.final_linear.weight = self.embeds.weight
        if max_seq_len is None:
            self.register_buffer("positional_embeddings", None, persistent=False)
        else:
            self.register_buffer(
                "positional_embeddings",
                build_sinusoidal_positional_embeddings(max_seq_len, d_model),
                persistent=False,
            )

    @staticmethod
    def from_config(config):
        return GPT(
            vocab_size=config["vocab_size"],
            N=config["N"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            h=config["h"],
            max_seq_len=config["L"],
            dropout=config["dropout"],
            weight_tying=config["weight_tying"],
        )

    def forward(self, tokens):
        """
        tokens.shape = (B, L)
        return.shape = (B, L, vocab_size)
        """
        x = self.embeds(tokens)
        L = x.shape[1]
        if self.positional_embeddings is None:
            pos = PositionalEmbeddings.forward(x)
        else:
            pos = self.positional_embeddings[:L, :].to(dtype=x.dtype)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        return self.final_linear(x)

    @torch.no_grad()
    def sample_next_token_id(self, tokens, temperature=0.8, top_k=50):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        logits = self.forward(tokens)
        logits = logits[:, -1, :] / temperature

        top_k = min(top_k, logits.shape[-1])
        values, indices = torch.topk(logits, top_k, dim=-1)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=indices, src=values)

        probs = torch.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
