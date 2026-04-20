"""
StructureLlama: Moonbeam 839M backbone + GRU decoder for compound MIDI generation
with similarity-bucket conditioning and prototype injection.

Architecture:
    Backbone: Llama (frozen, loaded from Moonbeam checkpoint).
    Decoder:  per-token OutputGRU over 6 compound attributes.
    LoRA:     rank-8 adapters on Q/V projections of all attention layers.

Conditioning (M1/M3 training modes):
    sim:high/mid/low token (after SOC): requests the structural similarity level
        of the generated section relative to the context.
    PROTO tokens (at bar boundaries): embeddings overridden with proto_proj applied
        to the mean-pooled Llama hidden states of the context section. Anchors
        generation to the context's musical character independently of the bucket label.

Sequence format (M1/M3):
    [SOS] [ctx_notes] [SOC] [sim:high/mid/low] [PROTO] [bar1_tgt] [PROTO] [bar2_tgt] ... [EOC]

Sequence format (M2, unconditional baseline):
    [SOS] [ctx_notes] [SOC] [tgt_notes] [EOC]

No external Moonbeam imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from scripts.model.llama_backbone import LlamaModel, LlamaConfig
from scripts.model.gru_decoder import OutputGRU
from scripts.model.lora import LoraLinear

# Compound token ID for prototype anchor tokens (must match dataset.PROTO_TOKEN)
PROTO_TOKEN_ID = -8


# ──────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class StructureLlamaOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None               # projected Llama hidden states
    generation_logits: Optional[torch.Tensor] = None    # GRU output logits (inference)
    generation_hidden_state: Optional[torch.Tensor] = None  # GRU hidden state (inference)


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class StructureLlama(nn.Module):
    """
    StructureLlama: Moonbeam-style Llama + GRU compound token generation model.

    Sequence format (M1/M3):
        [SOS] [ctx_notes] [SOC] [sim:high/mid/low] [PROTO] [bar1_tgt] [PROTO] [bar2_tgt] ... [EOC]

    Sequence format (M2):
        [SOS] [ctx_notes] [SOC] [tgt_notes] [EOC]

    PROTO embedding injection:
        Before each Llama forward pass, PROTO token embeddings are overridden with
        proto_proj(mean_pool(Llama_hidden(ctx_tokens))). This is done by calling embed_tokens
        separately to get the full embedding matrix, patching PROTO positions, then
        passing inputs_embeds to the backbone (bypassing its embed_tokens step).

    Loss is computed only over music token positions (where condition_mask == False).
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config

        # ── Llama backbone ──────────────────────────────────
        llama_cfg = LlamaConfig(**vars(config))
        if hasattr(llama_cfg, "decoder"):
            del llama_cfg.__dict__["decoder"]
        self.model = LlamaModel(llama_cfg)

        # ── GRU decoder ─────────────────────────────────────
        decoder_cfg = LlamaConfig(**vars(config.decoder))
        self.decoder = OutputGRU(decoder_cfg)

        # ── Projections ─────────────────────────────────────
        decoder_hidden = config.decoder.hidden_size
        self.summary_projection = nn.Linear(config.hidden_size, decoder_hidden, bias=False)
        self.decoder_embedding = nn.Embedding(config.decode_vocab_size, decoder_hidden)
        self.lm_head = nn.Linear(decoder_hidden, config.decode_vocab_size, bias=False)

        # ── Prototype projection ─────────────────────────────
        # Maps mean-pooled ctx Llama hidden states -> prototype vector injected at PROTO positions.
        # Randomly initialised (not in Moonbeam checkpoint); trained from scratch.
        self.proto_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.loss_func = CrossEntropyLoss()

    # ──────────────────────────────────────────────────────────
    # LoRA
    # ──────────────────────────────────────────────────────────

    def apply_lora(self, r: int = 8, alpha: int = 16,
                   use_proto: bool = True) -> None:
        """
        Freeze the entire backbone and inject LoRA adapters on Q/V projections.

        Trainable after this call:
          - model.supplementary_embedding (sim-bucket + other special token embeddings)
          - proto_proj (only when use_proto=True; frozen otherwise)
          - LoRA adapters (lora_A, lora_B) on every attention layer's q_proj and v_proj

        All other parameters remain frozen.

        Call AFTER loading the pretrained checkpoint and BEFORE moving to device.
        """
        # Step 1: freeze everything
        for param in self.parameters():
            param.requires_grad_(False)

        # Step 2: inject LoRA on Q/V in every Llama layer
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.q_proj = LoraLinear(attn.q_proj, r, alpha)
            attn.v_proj = LoraLinear(attn.v_proj, r, alpha)
            # LoraLinear.lora_A / lora_B are new nn.Parameters -> requires_grad=True by default

        # Step 3: unfreeze small trainable components
        self.model.supplementary_embedding.weight.requires_grad_(True)
        if use_proto:
            for param in self.proto_proj.parameters():
                param.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        proto_status = "trainable" if use_proto else "frozen (no PROTO)"
        print(f"LoRA applied (r={r}, alpha={alpha}): "
              f"trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M total "
              f"({100*trainable/total:.2f}%) -- proto_proj: {proto_status}")

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                              # (B, T, 6)
        labels: Optional[torch.Tensor] = None,               # (B, T, 7)
        condition_mask: Optional[torch.Tensor] = None,       # (B, T) bool
        additional_token_map: Optional[Dict[int, int]] = None,
        sample_weights: Optional[torch.Tensor] = None,       # (B,) per-sample loss weights
        # ── Inference-only ──────────────────────────────────
        decoded_language_tokens: Optional[torch.Tensor] = None,
        decoded_hidden_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> StructureLlamaOutput:
        """
        Training: provide input_ids (+ labels, condition_mask).
        Inference: provide decoded_language_tokens + decoded_hidden_state.
        sample_weights: optional (B,) tensor; when provided, per-sample loss weighting
            is applied.
        """
        if input_ids is not None and decoded_hidden_state is None:
            return self._training_forward(
                input_ids, labels, condition_mask, additional_token_map,
                attention_mask, sample_weights
            )
        if decoded_hidden_state is not None:
            return self._inference_forward(decoded_language_tokens, decoded_hidden_state)
        raise ValueError(
            "Provide input_ids (training) or decoded_hidden_state (inference)."
        )

    # ──────────────────────────────────────────────────────────
    # Position IDs
    # ──────────────────────────────────────────────────────────

    def _build_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build position_ids for multi-attribute RoPE from input_ids.

        For music tokens (input_ids[..., 0] > structure_token_offset):
            position_ids == input_ids  (raw compound values drive each RoPE axis)

        For structure tokens (input_ids[..., 0] <= structure_token_offset, i.e. <= -10):
            col 0 (onset RoPE): set to input_ids[..., 1] (bar_onset_tick)
            col 1 (dur RoPE):   set to 0

        For SOC/EOC/sec/PROTO/PAD tokens (token_id < pad_token AND NOT struct):
            all 6 position axes set to 0 (no temporal position)
        """
        position_ids = input_ids.clone()   # (B, T, 6)

        struct_mask = input_ids[..., 0] <= self.config.structure_token_offset  # (B, T)

        position_ids[..., 0] = torch.where(
            struct_mask,
            input_ids[..., 1],
            input_ids[..., 0],
        )
        position_ids[..., 1] = torch.where(
            struct_mask,
            torch.zeros_like(input_ids[..., 1]),
            input_ids[..., 1],
        )

        # SOC, EOC, sim-bucket, PROTO, PAD -- zero all RoPE axes
        non_music_mask = (input_ids[..., 0] < self.config.pad_token) & ~struct_mask
        position_ids = torch.where(
            non_music_mask.unsqueeze(-1).expand_as(position_ids),
            torch.zeros_like(position_ids),
            position_ids,
        )

        return position_ids

    # ──────────────────────────────────────────────────────────
    # Prototype helpers
    # ──────────────────────────────────────────────────────────

    def _get_ctx_proto_hidden(
        self,
        input_ids: torch.Tensor,                          # (B, T, 6)
        additional_token_map: Optional[Dict[int, int]],
    ) -> torch.Tensor:
        """
        Compute prototype vectors from Llama hidden states of the context region.

        For each sample, the context is input_ids[b, 1:soc_pos, :] (tokens between
        SOS and SOC). These are embedded, forwarded through Llama, mean-pooled, and
        projected through proto_proj.

        Returns: (B, hidden_size) prototype vectors.
        """
        B, T, _ = input_ids.shape
        soc_id = self.config.soc_token  # -4

        # SOC position per batch item
        soc_positions = (input_ids[..., 0] == soc_id).float().argmax(dim=1).long()  # (B,)
        max_ctx_len = int((soc_positions - 1).clamp(min=0).max().item())

        proto_vecs = torch.zeros(B, self.config.hidden_size, device=input_ids.device,
                                 dtype=input_ids.float().dtype)

        if max_ctx_len == 0:
            return proto_vecs

        # Build padded ctx batch (pad with PAD compound token)
        pad_row = torch.full((1, 6), self.config.pad_token,
                             dtype=input_ids.dtype, device=input_ids.device)
        ctx_ids_list = []
        ctx_mask_list = []
        for b in range(B):
            soc_pos = int(soc_positions[b].item())
            ctx = input_ids[b, 1:soc_pos, :]           # (L_ctx, 6)
            L = ctx.shape[0]
            if L == 0:
                ctx = pad_row.expand(max_ctx_len, -1)
                mask = torch.zeros(max_ctx_len, dtype=torch.long, device=input_ids.device)
            elif L < max_ctx_len:
                pad = pad_row.expand(max_ctx_len - L, -1)
                ctx = torch.cat([ctx, pad], dim=0)
                mask = torch.cat([
                    torch.ones(L, dtype=torch.long, device=input_ids.device),
                    torch.zeros(max_ctx_len - L, dtype=torch.long, device=input_ids.device),
                ])
            else:
                mask = torch.ones(max_ctx_len, dtype=torch.long, device=input_ids.device)
            ctx_ids_list.append(ctx)
            ctx_mask_list.append(mask)

        ctx_ids_batch = torch.stack(ctx_ids_list)   # (B, max_ctx_len, 6)
        ctx_attn_mask = torch.stack(ctx_mask_list)  # (B, max_ctx_len)

        # Single Llama forward over ctx tokens
        ctx_embeds  = self.model.embed_tokens(ctx_ids_batch,
                                              additional_token_map=additional_token_map)
        ctx_pos_ids = self._build_position_ids(ctx_ids_batch)
        ctx_out     = self.model(inputs_embeds=ctx_embeds,
                                 attention_mask=ctx_attn_mask,
                                 position_ids=ctx_pos_ids)
        ctx_hidden  = ctx_out.last_hidden_state  # (B, max_ctx_len, hidden)

        # Masked mean-pool
        mask_f = ctx_attn_mask.float().unsqueeze(-1)          # (B, max_ctx_len, 1)
        proto  = (ctx_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return self.proto_proj(proto)  # (B, hidden_size)

    def _inject_prototype_vecs(
        self,
        input_ids: torch.Tensor,   # (B, T, 6)
        all_embeds: torch.Tensor,  # (B, T, hidden) -- modified
        proto_vecs: torch.Tensor,  # (B, hidden)
    ) -> torch.Tensor:
        """Override PROTO token positions in all_embeds with pre-computed proto_vecs."""
        proto_mask = (input_ids[..., 0] == PROTO_TOKEN_ID)  # (B, T)
        if not proto_mask.any():
            return all_embeds
        for b in range(input_ids.shape[0]):
            positions = proto_mask[b].nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                all_embeds[b, positions, :] = proto_vecs[b]
        return all_embeds

    def compute_proto_vec(
        self,
        ctx_ids: torch.Tensor,
        additional_token_map: Optional[Dict[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute prototype vector from context section Llama hidden states.

        ctx_ids: (1, L_ctx, 6) -- context note tokens only (no SOS, no SOC)
        Returns: (1, hidden_size)

        Called once at inference start; result cached and passed to each encode() call.
        """
        ctx_embeds  = self.model.embed_tokens(ctx_ids,
                                              additional_token_map=additional_token_map)
        ctx_pos_ids = self._build_position_ids(ctx_ids)
        backbone_out = self.model(inputs_embeds=ctx_embeds, position_ids=ctx_pos_ids)
        hidden = backbone_out.last_hidden_state  # (1, L_ctx, hidden)
        proto  = hidden.mean(dim=1)              # (1, hidden)
        return self.proto_proj(proto)            # (1, hidden)

    # ──────────────────────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────────────────────

    def _training_forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        additional_token_map: Optional[Dict[int, int]],
        attention_mask: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor] = None,
    ) -> StructureLlamaOutput:
        position_ids = self._build_position_ids(input_ids)

        # Step 1: compute prototype from ctx hidden states.
        # Only when PROTO tokens exist in batch (M1/M3); skipped for M2.
        has_proto = (input_ids[..., 0] == PROTO_TOKEN_ID).any()
        if has_proto:
            proto_vecs = self._get_ctx_proto_hidden(input_ids, additional_token_map)
        else:
            proto_vecs = None

        # Step 2: embed full sequence
        all_embeds = self.model.embed_tokens(
            input_ids, additional_token_map=additional_token_map
        )  # (B, T, hidden)

        # Step 3: inject prototype at PROTO positions (no-op if proto_vecs is None)
        if proto_vecs is not None:
            all_embeds = self._inject_prototype_vecs(input_ids, all_embeds, proto_vecs)

        # Step 4: Llama forward
        backbone_out = self.model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = backbone_out.last_hidden_state   # (B, T, hidden_size)

        # Step 5: project to decoder hidden size
        projected = self.summary_projection(hidden_states)   # (B, T, decoder_hidden)

        loss = None
        if labels is not None:
            loss = self._compute_loss(projected, labels, condition_mask, sample_weights)

        return StructureLlamaOutput(loss=loss, logits=projected)

    def _compute_loss(
        self,
        projected: torch.Tensor,
        labels: torch.Tensor,
        condition_mask: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GRU teacher-forcing loss masked at condition prefix and PROTO positions.

        sample_weights: optional (B,) float tensor.  When provided, per-sample loss
            weighting is applied: each sample's mean active-token loss is scaled by
            its weight before averaging across the batch.
        """
        B  = projected.shape[0]
        T  = projected.shape[1]
        dh = projected.shape[2]

        shift_logits = projected[:, :-1, :].contiguous()       # (B, T-1, dh)
        shift_labels = labels[:, 1:T, :].contiguous()          # (B, T-1, 7)

        flat_logits = shift_logits.reshape(-1, dh)              # (B*(T-1), dh)
        flat_labels = shift_labels.reshape(-1, 7)               # (B*(T-1), 7)

        flat_labels_in  = flat_labels[:, :-1]                   # (B*(T-1), 6)
        flat_labels_out = flat_labels[:, 1:]                    # (B*(T-1), 6)

        encoded_in = self.decoder_embedding(flat_labels_in.long())  # (B*(T-1), 6, dh)

        h0 = flat_logits.unsqueeze(0).expand(
            self.decoder.gru.num_layers, -1, -1
        ).contiguous()

        gru_out, _ = self.decoder(encoded_in, h0)              # (B*(T-1), 6, dh)
        logits_out = self.lm_head(gru_out).float()             # (B*(T-1), 6, vocab)

        logits_flat  = logits_out.reshape(-1, self.config.decode_vocab_size)
        targets_flat = flat_labels_out.reshape(-1).long()

        if sample_weights is not None:
            # Per-token CE (no reduction), then weight by sample
            per_token = CrossEntropyLoss(reduction="none")(
                logits_flat, targets_flat
            ).view(B, T - 1, 6)                                 # (B, T-1, 6)

            if condition_mask is not None:
                shift_cond = condition_mask[:, 1:T].contiguous()   # (B, T-1)
                active = (~shift_cond).unsqueeze(-1).expand(-1, -1, 6)  # (B, T-1, 6)
                per_token = per_token * active.float()
                n_active = active.float().sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
            else:
                n_active = torch.full(
                    (B,), (T - 1) * 6, device=per_token.device, dtype=per_token.dtype
                )

            per_sample = per_token.sum(dim=(1, 2)) / n_active        # (B,)
            w = sample_weights.to(per_sample.device).float()

            total_w = w.sum()
            if total_w < 1e-9:
                return torch.tensor(0.0, device=per_sample.device, requires_grad=True)
            return (per_sample * w).sum() / total_w

        # Unweighted path (original behaviour)
        if condition_mask is not None:
            shift_cond = condition_mask[:, 1:T].contiguous()         # (B, T-1)
            active     = (~shift_cond).reshape(-1)                   # (B*(T-1),)
            active_exp = active.unsqueeze(1).expand(-1, 6).reshape(-1)  # (B*(T-1)*6,)

            if active_exp.sum() == 0:
                return torch.tensor(0.0, device=logits_flat.device, requires_grad=True)

            return self.loss_func(logits_flat[active_exp], targets_flat[active_exp])

        return self.loss_func(logits_flat, targets_flat)

    # ──────────────────────────────────────────────────────────
    # Inference forward (single GRU step)
    # ──────────────────────────────────────────────────────────

    def _inference_forward(
        self,
        decoded_language_tokens: torch.Tensor,   # (B, L)
        decoded_hidden_state: torch.Tensor,      # (num_layers, B, dh)
    ) -> StructureLlamaOutput:
        encoded = self.decoder_embedding(decoded_language_tokens.long())  # (B, L, dh)
        gen_logits, gen_hidden = self.decoder(encoded, decoded_hidden_state)
        gen_logits = self.lm_head(gen_logits).float()

        return StructureLlamaOutput(
            generation_logits=gen_logits,
            generation_hidden_state=gen_hidden,
        )

    # ──────────────────────────────────────────────────────────
    # Encode-only (get Llama hidden states for inference)
    # ──────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,
        additional_token_map: Optional[Dict[int, int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        proto_vec: Optional[torch.Tensor] = None,   # (1, hidden) precomputed prototype
    ) -> torch.Tensor:
        """
        Encode input_ids through the Llama backbone and project.
        Returns projected hidden states (B, T, decoder_hidden_size).

        proto_vec: if provided, overrides all PROTO token embeddings in the sequence.
                   Compute once via model.compute_proto_vec(ctx_ids) before the generation
                   loop and pass the same tensor at every step.
        """
        position_ids = self._build_position_ids(input_ids)
        all_embeds   = self.model.embed_tokens(
            input_ids, additional_token_map=additional_token_map
        )  # (B, T, hidden)

        # Inject precomputed prototype at PROTO positions
        if proto_vec is not None:
            proto_mask = (input_ids[..., 0] == PROTO_TOKEN_ID)  # (B, T)
            if proto_mask.any():
                for b in range(input_ids.shape[0]):
                    positions = proto_mask[b].nonzero(as_tuple=True)[0]
                    if positions.numel() > 0:
                        all_embeds[b, positions, :] = proto_vec[b]

        backbone_out = self.model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.summary_projection(backbone_out.last_hidden_state)
