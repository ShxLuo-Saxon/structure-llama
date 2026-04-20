"""
Checkpoint loader for StructureLlama.

Maps weights from moonbeam_309M.pt (or moonbeam.pt) to our StructureLlama model.

Handles:
  - Direct-load keys: Llama backbone layers, GRU decoder, projections
  - Partial-load:  supplementary_embedding (checkpoint has 2 rows; ours has more)
  - Skip + re-init: decoder_embedding, lm_head (vocab size mismatch for 309M)
  - Skip + remove:  old injection layers (bar/beat/struct/chord/gru_condition)

No external Moonbeam imports.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Keys that no longer exist in our model (old injection layers)
# ──────────────────────────────────────────────────────────────
_REMOVED_KEYS: Set[str] = {
    "bar_embedding.weight",
    "beat_embedding.weight",
    "structure_embedding.weight",
    "chord_placeholder_embedding.weight",
    "gru_condition_layer.weight",
    "gru_condition_layer.bias",
    "model.supplementary_embedding_metadata.weight",
    "supplementary_embedding_metadata.weight",
}

# ──────────────────────────────────────────────────────────────
# Keys where the checkpoint shape may differ from ours
# (will be re-initialised if shapes don't match)
# ──────────────────────────────────────────────────────────────
_SHAPE_SENSITIVE_KEYS: Set[str] = {
    "decoder_embedding.weight",
    "lm_head.weight",
}

# ──────────────────────────────────────────────────────────────
# Keys where we do a PARTIAL copy (checkpoint rows ⊂ our rows)
# ──────────────────────────────────────────────────────────────
_PARTIAL_KEYS: Set[str] = {
    "model.supplementary_embedding.weight",
}


def load_moonbeam_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """
    Load a Moonbeam pretrained checkpoint into a StructureLlama model.

    Args:
        model:        StructureLlama instance (already instantiated from config).
        ckpt_path:    Path to moonbeam_309M.pt or moonbeam.pt.
        map_location: torch.load map_location argument.
        verbose:      Print a per-key summary.

    Returns:
        The model with weights loaded in-place.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    # Some checkpoints wrap weights in an outer dict
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state_dict = raw["model"]
        else:
            state_dict = raw
    else:
        state_dict = raw

    # Strip DataParallel "module." prefix if present
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    our_state = model.state_dict()
    filtered: dict = {}
    skipped_removed: list = []
    skipped_shape: list = []
    partial_loaded: list = []
    not_in_model: list = []

    for k, v in state_dict.items():
        # 1. Old injection layers — discard entirely
        if k in _REMOVED_KEYS or any(k.endswith(suffix) for suffix in _REMOVED_KEYS):
            skipped_removed.append(k)
            continue

        # 2. Keys not present in our model at all
        if k not in our_state:
            not_in_model.append(k)
            continue

        # 3. Partial-load keys (embed table may have grown)
        if k in _PARTIAL_KEYS:
            target = our_state[k].clone()
            rows_ckpt = v.shape[0]
            rows_ours = target.shape[0]
            copy_rows = min(rows_ckpt, rows_ours)
            target[:copy_rows] = v[:copy_rows].to(target.dtype)
            filtered[k] = target
            partial_loaded.append(f"{k}: ckpt={v.shape} -> ours={target.shape}, copied {copy_rows} rows")
            continue

        # 4. Shape-sensitive keys — re-init if shapes differ
        if k in _SHAPE_SENSITIVE_KEYS:
            if v.shape != our_state[k].shape:
                skipped_shape.append(
                    f"{k}: ckpt={tuple(v.shape)}, ours={tuple(our_state[k].shape)} -> re-init"
                )
                continue  # leave model's random init in place

        # 5. Direct copy (dtype cast if needed)
        if v.shape == our_state[k].shape:
            filtered[k] = v.to(our_state[k].dtype)
        else:
            skipped_shape.append(
                f"{k}: ckpt={tuple(v.shape)}, ours={tuple(our_state[k].shape)} -> re-init"
            )

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if verbose:
        n_total = len(state_dict)
        n_loaded = len(filtered)
        print(f"\n[checkpoint_loader] Loaded {n_loaded}/{n_total} keys from {ckpt_path}")
        if partial_loaded:
            print(f"  Partial loads ({len(partial_loaded)}):")
            for msg in partial_loaded:
                print(f"    {msg}")
        if skipped_shape:
            print(f"  Shape mismatch -> re-init ({len(skipped_shape)}):")
            for msg in skipped_shape:
                print(f"    {msg}")
        if skipped_removed:
            print(f"  Removed (old injection layers): {len(skipped_removed)} keys")
        if not_in_model:
            print(f"  Not in our model: {len(not_in_model)} keys")
            for k in not_in_model[:5]:
                print(f"    {k}")
        if missing:
            print(f"  Missing in checkpoint (random init): {len(missing)} keys")
            for k in list(missing)[:5]:
                print(f"    {k}")
        print()

    return model
