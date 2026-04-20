# coding=utf-8
"""
Self-contained LLaMA backbone for StructureLlama.

Adapted from:
  Moonbeam-MIDI-Foundation-Model/src/llama_recipes/transformers_minimal/
  src/transformers/models/llama/modeling_llama.py
.
"""

import math
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from scripts.model.embeddings import EMBEDDING_METHODS

# ---------------------------------------------------------------------------
# Activation functions (subset of transformers ACT2FN)
# ---------------------------------------------------------------------------
ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}

# ---------------------------------------------------------------------------
# Minimal logger shim
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


class _Logger:
    """Thin wrapper that adds warning_once (suppresses duplicate warnings)."""

    def __init__(self):
        self._warned: set = set()

    def warning_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            warnings.warn(msg, stacklevel=3)

    def warning(self, msg: str) -> None:
        warnings.warn(msg, stacklevel=2)


logger = _Logger()

# ---------------------------------------------------------------------------
# Flash-attention availability (always False — we use SDPA only)
# ---------------------------------------------------------------------------


def is_flash_attn_2_available() -> bool:
    return False


def is_flash_attn_greater_or_equal_2_10() -> bool:
    return False


# ---------------------------------------------------------------------------
# Minimal config class
# ---------------------------------------------------------------------------


class LlamaConfig:
    """
    Minimal config container.  Initialised from a dict (from JSON).

    All keys in the dict become attributes on the object, so existing
    code like ``config.hidden_size`` continues to work without change.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Normalise: JSON uses 'attn_implementation', code uses '_attn_implementation'
        if not hasattr(self, "_attn_implementation"):
            self._attn_implementation = getattr(self, "attn_implementation", "sdpa")
        # Defaults for optional flags
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = True
        if not hasattr(self, "use_cache"):
            self.use_cache = False
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = None

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        return cls(**d)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None


# ---------------------------------------------------------------------------
# Utility functions (verbatim from source)
# ---------------------------------------------------------------------------


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Rotary embedding classes (verbatim from source)
# ---------------------------------------------------------------------------


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (batch, seq_len)
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )  # (batch, dim/2, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (batch, 1, len)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling."""

    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling."""

    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = super().forward(x, position_ids)
        return cos, sin


# ---------------------------------------------------------------------------
# LlamaRMSNorm (verbatim from source)
# ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """LlamaRMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# LlamaMLP (verbatim from source)
# ---------------------------------------------------------------------------


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# ---------------------------------------------------------------------------
# LlamaAttention (multi-rope, used as base for LlamaSdpaAttention)
# ---------------------------------------------------------------------------


class LlamaAttention(nn.Module):
    """Multi-headed attention with per-attribute RoPE (verbatim from source)."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta_onset = config.rope_theta_onset
        self.rope_theta_dur = config.rope_theta_dur
        self.rope_theta_octave = config.rope_theta_octave
        self.rope_theta_pitch = config.rope_theta_pitch
        self.rope_theta_velocity = config.rope_theta_velocity
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb_onset = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_onset,
            )
            self.rotary_emb_dur = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_dur,
            )
            self.rotary_emb_octave = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_octave,
            )
            self.rotary_emb_pitch = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_pitch,
            )
            self.rotary_emb_velocity = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_velocity,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta_onset,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta_onset,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = torch.cat(
                [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            key_states = torch.cat(
                [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            value_states = torch.cat(
                [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb_onset(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(
                [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# LlamaSdpaAttention (verbatim from source — this is the variant we use)
# ---------------------------------------------------------------------------


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention using torch.nn.functional.scaled_dot_product_attention.
    Inherits weights from LlamaAttention unchanged; only the forward pass is
    adapted to use the SDPA API with per-attribute multi-RoPE.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        additional_token_map: Optional[Dict] = None,
        additional_tokens_pos_map: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but "
                "`torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers '
                'version v5.0.0 onwards. This warning can be removed using the argument '
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # SOS / EOS position fixup — replace negative token IDs with neutral positions
        where_sos = (position_ids[..., 0] == self.config.sos_token).unsqueeze(-1)
        where_eos = (position_ids[..., 0] == self.config.eos_token).unsqueeze(-1)
        position_ids_sos = torch.where(
            where_sos, torch.tensor([0 for _ in range(6)]).to(hidden_states.device), position_ids
        )
        position_ids_eos = torch.where(
            where_eos,
            torch.tensor([2 ** 15 for _ in range(6)]).to(hidden_states.device),
            position_ids_sos,
        )
        position_ids = position_ids_eos

        # Handle additional (non-SOS/EOS) token positions
        if additional_token_map is not None and additional_tokens_pos_map is not None:
            for token_id in additional_token_map:
                where_new_token = (position_ids[..., 0] == token_id).unsqueeze(-1)
                if str(token_id) in additional_tokens_pos_map:
                    position_ids = torch.where(
                        where_new_token,
                        torch.tensor(additional_tokens_pos_map[str(token_id)]).to(hidden_states.device),
                        position_ids,
                    )
                else:
                    position_ids = torch.where(
                        where_new_token,
                        torch.tensor([0 for _ in range(6)]).to(hidden_states.device),
                        position_ids,
                    )

        # Per-attribute RoPE — each head group gets its own rotation
        cos_onset, sin_onset = self.rotary_emb_onset(value_states, position_ids[:, :, 0])
        cos_dur, sin_dur = self.rotary_emb_dur(value_states, position_ids[:, :, 1])
        cos_octave, sin_octave = self.rotary_emb_octave(value_states, position_ids[:, :, 2])
        cos_pitch, sin_pitch = self.rotary_emb_pitch(value_states, position_ids[:, :, 3])
        cos_velocity, sin_velocity = self.rotary_emb_velocity(value_states, position_ids[:, :, 5])

        query_states_split = query_states.view(bsz, 6, -1, q_len, self.head_dim)
        key_states_split = key_states.view(bsz, 6, -1, q_len, self.head_dim)

        query_states_onset, key_states_onset = apply_rotary_pos_emb(
            query_states_split[:, 0], key_states_split[:, 0], cos_onset, sin_onset
        )
        query_states_dur, key_states_dur = apply_rotary_pos_emb(
            query_states_split[:, 1], key_states_split[:, 1], cos_dur, sin_dur
        )
        query_states_octave, key_states_octave = apply_rotary_pos_emb(
            query_states_split[:, 2], key_states_split[:, 2], cos_octave, sin_octave
        )
        query_states_pitch, key_states_pitch = apply_rotary_pos_emb(
            query_states_split[:, 3], key_states_split[:, 3], cos_pitch, sin_pitch
        )
        # instrument head group reuses onset rotation (same as source)
        query_states_instr, key_states_instr = apply_rotary_pos_emb(
            query_states_split[:, 4], key_states_split[:, 4], cos_onset, sin_onset
        )
        query_states_velocity, key_states_velocity = apply_rotary_pos_emb(
            query_states_split[:, 5], key_states_split[:, 5], cos_velocity, sin_velocity
        )

        query_states = torch.cat(
            (
                query_states_onset,
                query_states_dur,
                query_states_octave,
                query_states_pitch,
                query_states_instr,
                query_states_velocity,
            ),
            dim=1,
        )
        key_states = torch.cat(
            (
                key_states_onset,
                key_states_dur,
                key_states_octave,
                key_states_pitch,
                key_states_instr,
                key_states_velocity,
            ),
            dim=1,
        )

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ---------------------------------------------------------------------------
# Attention class registry
# ---------------------------------------------------------------------------

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "sdpa": LlamaSdpaAttention,
}

# ---------------------------------------------------------------------------
# LlamaDecoderLayer (verbatim from source)
# ---------------------------------------------------------------------------


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        additional_token_map: Optional[Dict] = None,
        additional_tokens_pos_map: Optional[Dict] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: input of shape ``(batch, seq_len, embed_dim)``
            attention_mask: optional mask
            position_ids: optional position indices
            past_key_value: optional KV cache
            output_attentions: whether to return attention weights
            use_cache: whether to return present KV cache
            cache_position: indices for static KV cache update
            additional_token_map: mapping of extra token IDs to embedding indices
            additional_tokens_pos_map: mapping of extra token IDs to 6-d position vectors
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            additional_token_map=additional_token_map,
            additional_tokens_pos_map=additional_tokens_pos_map,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ---------------------------------------------------------------------------
# LlamaModel — backbone adapted for StructureLlama
# ---------------------------------------------------------------------------


class LlamaModel(nn.Module):
    """
    Transformer decoder backbone for StructureLlama.

    Differences vs the original Moonbeam LlamaModel:
    - Inherits from nn.Module directly (no PreTrainedModel dependency).
    - ``embed_tokens`` is a method, not an nn.Embedding layer.
    - ``supplementary_embedding`` is a unified table covering ALL special
      tokens (SOS=0, EOS=1, SOC=2, EOC=3, then metadata values, then
      structure tokens), sized automatically from config attributes
      ``num_metadata_values`` and ``num_structure_tokens``.
    - No separate ``supplementary_embedding_metadata`` table.
    - ``additional_token_map`` covers all non-SOS/EOS special tokens;
      each entry maps token_id -> embedding_index in the unified table.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.sos_token = config.sos_token
        self.eos_token = config.eos_token

        # --- Music attribute embeddings (6 compound token dimensions) ---
        emb_size = config.hidden_size // 6
        self.onset_embedding = EMBEDDING_METHODS[config.onset_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.onset_embedding.items() if k != "method"}
        )
        self.dur_embedding = EMBEDDING_METHODS[config.dur_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.dur_embedding.items() if k != "method"}
        )
        self.octave_embedding = EMBEDDING_METHODS[config.octave_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.octave_embedding.items() if k != "method"}
        )
        self.pitch_embedding = EMBEDDING_METHODS[config.pitch_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.pitch_embedding.items() if k != "method"}
        )
        self.instrument_embedding = EMBEDDING_METHODS[config.instrument_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.instrument_embedding.items() if k != "method"}
        )
        self.velocity_embedding = EMBEDDING_METHODS[config.velocity_embedding["method"]](
            dim=emb_size, **{k: v for k, v in config.velocity_embedding.items() if k != "method"}
        )

        # --- Unified supplementary embedding ---
        # Index layout:
        #   0        : SOS
        #   1        : EOS
        #   2        : SOC
        #   3        : EOC
        #   4 ..     : metadata values (num_metadata_values entries)
        #   4+M ..   : structure tokens (num_structure_tokens entries)
        num_special = (
            4
            + getattr(config, "num_metadata_values", 0)
            + getattr(config, "num_structure_tokens", 56)
        )
        self.supplementary_embedding = nn.Embedding(num_special, config.hidden_size)

        # 3-layer MLP applied after embedding fusion
        self.supplementary_MLP = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        # --- Transformer decoder layers ---
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_tokens(self, input_ids: torch.Tensor, additional_token_map: Optional[Dict] = None):
        """
        Embed a batch of compound token sequences.

        Args:
            input_ids: ``(batch, seq_len, 6)`` integer tensor
                       (onset, duration, octave, pitch_class, instrument, velocity)
            additional_token_map: optional ``{token_id: embedding_index}`` mapping
                covering ALL non-SOS/EOS special tokens (SOC, EOC, metadata,
                structure tokens).  Each ``embedding_index`` indexes into the
                unified ``supplementary_embedding`` table.

        Returns:
            ``(batch, seq_len, hidden_size)`` float tensor
        """
        # Pre-compute SOS / EOS embeddings and expand to batch
        sos = (
            self.supplementary_embedding(torch.tensor(0).to(input_ids.device))[None, None, ...]
            .expand(input_ids.size(0), -1, -1)
        )
        eos = (
            self.supplementary_embedding(torch.tensor(1).to(input_ids.device))[None, None, ...]
            .expand(input_ids.size(0), -1, -1)
        )

        # Detect SOS / EOS positions
        where_sos = (input_ids[:, :, 0] == self.sos_token).unsqueeze(-1)
        where_eos = (input_ids[:, :, 0] == self.eos_token).unsqueeze(-1)

        # Pre-compute masks and embeddings for all additional tokens
        if additional_token_map is not None:
            where_new_tokens_dict: Dict[int, torch.Tensor] = {}
            for token_id in additional_token_map.keys():
                where_new_tokens_dict[token_id] = (input_ids[:, :, 0] == token_id).unsqueeze(-1)

            new_token_embeddings: Dict[int, torch.Tensor] = {}
            for token_id, embed_idx in additional_token_map.items():
                new_token_embeddings[token_id] = (
                    self.supplementary_embedding(torch.tensor(embed_idx).to(input_ids.device))[None, None, ...]
                    .expand(input_ids.size(0), -1, -1)
                )

        # Replace special token IDs with 0 so FME indexing doesn't fail
        input_ids_tmp = torch.where(
            (where_sos | where_eos),
            torch.tensor([0 for _ in range(6)]).to(input_ids),
            input_ids,
        )
        if additional_token_map is not None:
            for token_id in additional_token_map:
                input_ids_tmp = torch.where(
                    where_new_tokens_dict[token_id],
                    torch.tensor([0 for _ in range(6)]).to(input_ids),
                    input_ids_tmp,
                )

        # Embed each of the 6 compound attributes
        onsets = self.onset_embedding(input_ids_tmp[..., 0])
        durs = self.dur_embedding(input_ids_tmp[..., 1])
        octaves = self.octave_embedding(input_ids_tmp[..., 2])
        pitch_classes = self.pitch_embedding(input_ids_tmp[..., 3])
        instruments = self.instrument_embedding(input_ids_tmp[..., 4])
        velocities = self.velocity_embedding(input_ids_tmp[..., 5])
        out_fme = torch.cat([onsets, durs, octaves, pitch_classes, instruments, velocities], dim=-1)

        # Overwrite SOS / EOS positions
        out_fme_sos = torch.where(where_sos, sos, out_fme)
        out_fme_sos_eos = torch.where(where_eos, eos, out_fme_sos)

        # Overwrite additional special token positions
        out_final = out_fme_sos_eos
        if additional_token_map is not None:
            for token_id in additional_token_map:
                out_final = torch.where(
                    where_new_tokens_dict[token_id],
                    new_token_embeddings[token_id],
                    out_final,
                )

        # Non-linear projection over all positions
        out_final = self.supplementary_MLP(out_final)
        return out_final

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        additional_token_map: Optional[Dict[int, int]] = None,
        additional_tokens_pos_map: Optional[Dict[str, List[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, "
                "and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, additional_token_map=additional_token_map)

        # KV-cache handling — keep legacy tuple format support
        return_legacy_cache = False
        if use_cache and past_key_values is not None and not hasattr(past_key_values, "update"):
            return_legacy_cache = True
            # Wrap legacy tuple cache into a simple dynamic cache
            past_key_values = _DynamicCacheCompat.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    additional_token_map,
                    additional_tokens_pos_map,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    additional_token_map=additional_token_map,
                    additional_tokens_pos_map=additional_tokens_pos_map,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # ------------------------------------------------------------------
    # Causal mask (verbatim from source)
    # ------------------------------------------------------------------

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if attention_mask is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            target_length = past_seen_tokens + sequence_length

            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)

            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            return causal_mask

        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, None, :]  # batch, 1, len
            attention_mask_rep = attention_mask.expand(-1, attention_mask.shape[2], -1)  # batch, len, len
            block_mask = attention_mask_rep == attention_mask_rep.transpose(1, 2)
            seq_len = attention_mask.shape[2]
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask.device)
            )
            attention_mask = block_mask & causal_mask  # batch, len, len
            attention_mask = attention_mask.unsqueeze(1)  # batch, 1, len, len
            return attention_mask

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        # kept for API compatibility; embed_tokens is a method here
        pass

    def add_supplementary_embedding(self, num_tokens: int, embedding_name: str, hidden_size: int) -> None:
        """
        Dynamically attach an additional nn.Embedding to the model.
        Needed for compatibility with checkpoint-loading code.
        """
        new_embedding = nn.Embedding(num_tokens, hidden_size)
        new_embedding.requires_grad_(True)
        setattr(self, embedding_name, new_embedding)


# ---------------------------------------------------------------------------
# Minimal dynamic KV-cache shim (used only when past_key_values is a legacy
# tuple; avoids importing transformers.cache_utils)
# ---------------------------------------------------------------------------


class _DynamicCacheCompat:
    """
    Minimal KV-cache wrapper that understands the legacy tuple format and
    exposes the ``update`` / ``get_seq_length`` / ``to_legacy_cache`` API
    expected by the decoder layers.
    """

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple]) -> "_DynamicCacheCompat":
        cache = cls()
        if past_key_values is not None:
            for k, v in past_key_values:
                cache.key_cache.append(k)
                cache.value_cache.append(v)
        return cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(zip(self.key_cache, self.value_cache))
