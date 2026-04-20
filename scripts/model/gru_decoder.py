"""
GRU-based output decoder for compound token generation.

Copied from Moonbeam-MIDI-Foundation-Model/src/llama_recipes/transformers_minimal/
src/transformers/models/llama/modeling_llama.py (lines 2478-2501).
No external Moonbeam imports.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Tuple


class OutputGRU(nn.Module):
    """
    Multi-layer GRU decoder used to autoregressively generate the 7 language tokens
    (sos_out + 6 compound attributes) for each music token position.

    During training (teacher-forcing):
        x      : (B*T, L-1, hidden_size)  — embedded target tokens (shifted right)
        hidden : (num_layers, B*T, hidden_size) — initial hidden = Llama summary projection
        → output: (B*T, L-1, hidden_size), new_hidden

    During inference (step-by-step):
        x      : (B, 1, hidden_size)   — single embedded token
        hidden : (num_layers, B, hidden_size)
        → output: (B, 1, hidden_size), new_hidden
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        self.num_hidden_layers = num_hidden_layers

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (batch, seq_len, hidden_size)
            hidden: (num_layers, batch, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
            hidden: (num_layers, batch, hidden_size)
        """
        output, hidden = self.gru(x, hidden)
        logits = self.fc_out(output)
        return logits, hidden


def make_decoder_config(decoder_dict: dict) -> SimpleNamespace:
    """Convert the decoder sub-dict from the model config JSON into a SimpleNamespace."""
    cfg = SimpleNamespace(**decoder_dict)
    return cfg
