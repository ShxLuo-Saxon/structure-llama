"""
scripts/model/lora.py

Low-Rank Adaptation (LoRA) module for StructureLlama.

LoraLinear wraps a frozen nn.Linear with trainable rank-r adapters:
    output = base(x) + (alpha/r) * x @ A^T @ B^T

where A is initialised with Kaiming uniform and B is initialised to zero,
so the LoRA contribution starts at zero and grows during training.

Reference: Hu et al. 2022 -- LoRA: Low-Rank Adaptation of Large Language Models
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoraLinear(nn.Module):
    """
    Wraps a frozen nn.Linear with LoRA adapters (rank r, scaling alpha/r).

    The base weight is NOT copied -- the original Linear object is stored as-is
    (already frozen by apply_lora before injection).

    Parameters
    ----------
    base : nn.Linear
        The frozen linear layer to adapt.
    r : int
        LoRA rank (number of adapter dimensions).
    alpha : int
        LoRA scaling hyperparameter. Effective scale = alpha / r.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base    = base
        self.scaling = alpha / r
        in_f  = base.in_features
        out_f = base.out_features
        # A: (r, in_f)   -- projection down
        # B: (out_f, r)  -- projection up, zero-init so LoRA starts at zero
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base(x) + scaling * x @ A^T @ B^T
        return self.base(x) + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.lora_A.shape[0]}, scaling={self.scaling:.3f}")
