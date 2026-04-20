"""
Compound token embedding modules.

Copied from Moonbeam-MIDI-Foundation-Model/src/llama_recipes/transformers_minimal/
src/transformers/models/llama/modeling_llama.py (lines 1295-1348).
No external Moonbeam imports.
"""

import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    """Standard learnable word embedding lookup table."""

    def __init__(self, vocab_size: int, dim: int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.embedding(inp)


class Fundamental_Music_Embedding(nn.Module):
    """
    Fourier/Frequency Music Embedding (FME).

    Encodes continuous-valued musical attributes (onset, duration, pitch, velocity)
    using sinusoidal position encoding with a learnable bias and a linear projection.

    Args:
        dim:   Output embedding dimension.
        base:  RoPE-style base frequency (different per attribute).
    """

    def __init__(self, dim: int, base: float, padding_idx=None, device=None):
        super().__init__()
        self.d_model = dim
        self.base = base

        translation_bias = torch.rand((1, self.d_model), dtype=torch.float32).to(device)
        self.translation_bias = nn.Parameter(translation_bias, requires_grad=True)

        i = torch.arange(self.d_model)
        angle_rates = 1.0 / torch.pow(self.base, (2 * (i // 2)) / self.d_model)
        self.angles = angle_rates[None, ...]  # (1, dim)

        self.linear_fme = nn.Linear(self.d_model, self.d_model)

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: (batch, seq_len)  — integer token values
        assert inp.dim() == 2

        device_type = inp.device.type
        if device_type == "mps":
            device_type = "cpu"
        data_type = self.linear_fme.weight.dtype

        inp = inp[..., None]  # (batch, seq_len, 1)
        angle_rads = inp * self.angles.to(inp.device, dtype=data_type)  # (batch, seq_len, dim)

        # sin on even indices, cos on odd
        angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, :, 0::2])
        angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

        pos_encoding = angle_rads.to(data_type)
        pos_encoding = pos_encoding + self.translation_bias.to(data_type)
        out = self.linear_fme(pos_encoding)
        return out  # (batch, seq_len, dim)


EMBEDDING_METHODS = {
    "WE": WordEmbedding,
    "FME": Fundamental_Music_Embedding,
}
