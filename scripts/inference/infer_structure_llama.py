r"""
Inference script for StructureLlama -- similarity-bucket conditioning.

For each test pair in pairs.csv:
  - Context section (normalized) is the structural prompt.
  - Prototype vector is computed once from context hidden states.
  - PROTO tokens are inserted at target bar boundaries (estimated from filename).
  - A sim-bucket token (sim:high/mid/low) is appended after [SOC].
  - The model autoregressively generates the target section.

Prompt format (M1 / M3):
    [SOS] [ctx notes] [SOC] [sim:high/mid/low] [PROTO] --> generate

Prompt format (M2 -- no bucket token, no PROTO):
    [SOS] [ctx notes] [SOC] --> generate

Generation is O(N^2) in output length -- use HPC for full test sets.

Usage:
    python scripts/inference/infer_structure_llama.py \
        --ckpt_path    /path/to/best.pt \
        --config_path  scripts/configs/model_config_structure_llama_839M.json \
        --sections_dir /path/to/data/sections \
        --pairs_csv    /path/to/data/sections/pairs.csv \
        --output_dir   outputs/m1 \
        --condition_mode m1 \
        --split test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.config import load_config
from scripts.model.structure_llama import StructureLlama, PROTO_TOKEN_ID
from scripts.model.tokenizer import MusicTokenizer
from scripts.utils.checkpoint_loader import load_moonbeam_checkpoint
from scripts.training.dataset import (
    SECTION_TOKEN_MAP, SIM_HIGH_TOKEN, SIM_MID_TOKEN, SIM_LOW_TOKEN,
    PROTO_TOKEN, _BUCKET_TOKEN, _OTHER_BUCKETS,
    _bars_from_filename, _parse_num_bars, _normalize_onsets,
)


# ──────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling. Returns (B, 1) sampled token indices."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum  = torch.cumsum(probs_sort, dim=-1)
    mask       = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True).clamp(min=1e-9))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    use_proto: bool = True,
) -> tuple[StructureLlama, MusicTokenizer]:
    """Load StructureLlama from checkpoint + config.

    Supports two checkpoint formats:
      - Moonbeam pretrained (.pt with raw state dict): loaded via load_moonbeam_checkpoint.
      - Training checkpoint (.pt with model_state_dict key): saved by train.py.

    For Run 9+ (LoRA) checkpoints: pass use_lora=True so apply_lora() is called before
    loading, ensuring the LoraLinear architecture is in place to receive the state dict.
    use_proto: when False, proto_proj stays frozen (ablation: no PROTO tokens).
    """
    config = load_config(config_path)
    model  = StructureLlama(config)

    if use_lora:
        model.apply_lora(r=lora_rank, alpha=lora_alpha, use_proto=use_proto)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Training checkpoint saved by train.py
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        print(f"Loaded training checkpoint (epoch {epoch}) from {ckpt_path}")
    else:
        # Raw Moonbeam pretrained checkpoint
        load_moonbeam_checkpoint(model, ckpt_path, verbose=True)

    model.to(device).eval()

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print("Model precision: BF16")

    tokenizer = MusicTokenizer.from_config(config, token_dict_path=None)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────

def build_prompt(
    tokenizer: MusicTokenizer,
    ctx_tokens: np.ndarray,        # (N, 6) normalized context section
    sim_bucket: str,               # "sim:high", "sim:mid", or "sim:low"
    condition_mode: str,           # "m1", "m2", or "m3"
    max_section_len: int = 509,
    use_proto: bool = True,
) -> torch.Tensor:
    """
    Build initial prompt tensor.

    For M1 (use_proto=True):  [SOS][ctx][SOC][sim:bucket][PROTO]
    For M1 (use_proto=False): [SOS][ctx][SOC][sim:bucket]
    For M2:                   [SOS][ctx][SOC]
    For M3: same as M1 but with wrong bucket label

    Returns (1, T_prompt, 6) long tensor.
    """
    ctx = _normalize_onsets(ctx_tokens)[:max_section_len]
    ctx_list = ctx.tolist()

    sos = tokenizer.sos_token_compound   # [-1]*6
    soc = tokenizer.soc_token_compound   # [-4]*6

    if condition_mode == "m2":
        prompt = [sos] + ctx_list + [soc]
    else:
        if condition_mode == "m3":
            bucket = _OTHER_BUCKETS[sim_bucket][0]   # first wrong bucket (deterministic)
        else:
            bucket = sim_bucket
        bucket_tok = _BUCKET_TOKEN[bucket]   # already [-6]*6 / [-7]*6 / [-9]*6
        if use_proto:
            proto  = [PROTO_TOKEN] * 6
            prompt = [sos] + ctx_list + [soc, bucket_tok, proto]
        else:
            prompt = [sos] + ctx_list + [soc, bucket_tok]

    return torch.tensor([prompt], dtype=torch.long)   # (1, T_prompt, 6)


def _estimate_bar_size(ctx_tokens: np.ndarray, ctx_bars_str: str) -> float:
    """
    Estimate bar size in onset ticks from context section.

    Uses context rather than target because we don't have target notes at inference time.
    Same-label pairs should have similar bar durations; diff-label is approximate.
    """
    num_bars = _parse_num_bars(ctx_bars_str)
    if num_bars <= 1 or len(ctx_tokens) == 0:
        return 800.0  # default: 4/4 at 120 BPM = 800 ticks/bar
    ctx_norm    = _normalize_onsets(ctx_tokens)
    max_onset   = float(ctx_norm[-1, 0])
    if max_onset == 0:
        return 800.0
    return max_onset / max(num_bars - 1, 1)


# ──────────────────────────────────────────────────────────────
# Core generation
# ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_one(
    model: StructureLlama,
    tokenizer: MusicTokenizer,
    prompt_ids: torch.Tensor,
    additional_token_map: Dict[int, int],
    proto_vec: Optional[torch.Tensor],     # (1, hidden) precomputed; None for M2
    bar_size_est: float,                   # estimated bar duration in onset ticks
    target_bars: int,                      # expected number of bars in target
    condition_mode: str,
    max_gen_len: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """
    Generate a single target section conditioned on prompt_ids.

    For M1/M3: inserts additional PROTO tokens at estimated bar boundaries
               during generation to maintain the prototype signal.
    For M2:    no PROTO tokens; proto_vec is None.

    Returns list of compound tokens with absolute onsets.
    """
    num_gru_layers = model.config.decoder.num_hidden_layers

    valid_sets: List[List[int]] = [
        list(tokenizer.timeshift_dict.values()),
        list(tokenizer.duration_dict.values()),
        list(tokenizer.octave_dict.values()),
        list(tokenizer.pitch_dict.values()),
        list(tokenizer.instrument_dict.values()),
        list(tokenizer.velocity_dict.values()),
    ]
    valid_tensors: List[torch.Tensor] = [
        torch.tensor(vs, dtype=torch.long, device=device) for vs in valid_sets
    ]

    eos_lang_idx = tokenizer.timeshift_dict[tokenizer.eos_timeshift]

    all_ids  = prompt_ids.to(device)
    projected = model.encode(all_ids, additional_token_map, proto_vec=proto_vec)

    gru_h = (
        projected[:, -1, :]
        .unsqueeze(0)
        .expand(num_gru_layers, -1, -1)
        .contiguous()
    )

    cumulative_onset  = 0
    generated: List[List[int]] = []

    # Bar boundary tracking for PROTO insertion (M1/M3 only)
    # Bar 0 PROTO is already in the prompt; track from bar 1 onwards
    next_bar_onset    = bar_size_est    # onset at which to insert next PROTO
    bars_inserted     = 1              # bar 0 PROTO already in prompt
    use_proto = (condition_mode != "m2") and (proto_vec is not None)

    for step in range(max_gen_len):

        # Insert PROTO at bar boundary before generating next note
        if use_proto and bars_inserted < target_bars and cumulative_onset >= next_bar_onset:
            proto_tok = torch.tensor(
                [[[PROTO_TOKEN] * 6]], dtype=torch.long, device=device
            )  # (1, 1, 6)
            all_ids   = torch.cat([all_ids, proto_tok], dim=1)
            projected = model.encode(all_ids, additional_token_map, proto_vec=proto_vec)
            gru_h = (
                projected[:, -1, :]
                .unsqueeze(0)
                .expand(num_gru_layers, -1, -1)
                .contiguous()
            )
            next_bar_onset   += bar_size_est
            bars_inserted    += 1

        # GRU: decode 6 attributes
        gru_in = torch.tensor(
            [[tokenizer.sos_out]], dtype=torch.long, device=device
        )
        lang_tokens: List[int] = []

        for attr_idx in range(6):
            out    = model._inference_forward(gru_in, gru_h)
            gru_h  = out.generation_hidden_state
            logits = out.generation_logits[:, -1, :].float()

            masked = torch.full_like(logits, float("-inf"))
            idx    = valid_tensors[attr_idx]
            masked.scatter_(1, idx.unsqueeze(0), logits[:, idx])

            if temperature > 0:
                probs = torch.softmax(masked / temperature, dim=-1)
                tok   = sample_top_p(probs, top_p)
            else:
                tok   = masked.argmax(dim=-1, keepdim=True)

            lang_tokens.append(tok.item())
            gru_in = tok

        # EOC check
        if lang_tokens[0] == eos_lang_idx:
            break

        lang_t   = torch.tensor([lang_tokens], dtype=torch.long, device=device)
        raw_vals = tokenizer.convert_from_language_tokens(lang_t)[0].tolist()

        if (int(raw_vals[2]) >= tokenizer.sos_octave or
                int(raw_vals[3]) >= tokenizer.sos_pitch_class or
                int(raw_vals[5]) >= tokenizer.sos_velocity):
            continue

        timeshift         = int(raw_vals[0])
        cumulative_onset += timeshift
        abs_compound = [
            cumulative_onset,
            int(raw_vals[1]),
            int(raw_vals[2]),
            int(raw_vals[3]),
            int(raw_vals[4]),
            int(raw_vals[5]),
        ]
        generated.append(abs_compound)

        new_tok = (
            torch.tensor(abs_compound, dtype=torch.long, device=device)
            .unsqueeze(0).unsqueeze(0)
        )
        all_ids   = torch.cat([all_ids, new_tok], dim=1)
        projected = model.encode(all_ids, additional_token_map, proto_vec=proto_vec)
        gru_h = (
            projected[:, -1, :]
            .unsqueeze(0)
            .expand(num_gru_layers, -1, -1)
            .contiguous()
        )

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{max_gen_len}")

    return generated


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructureLlama Run 6 inference -- prototype interleaving"
    )
    parser.add_argument("--ckpt_path",      required=True)
    parser.add_argument("--config_path",    required=True)
    parser.add_argument("--sections_dir",   required=True)
    parser.add_argument("--pairs_csv",      required=True)
    parser.add_argument("--output_dir",     default="outputs/generated")
    parser.add_argument("--condition_mode", default="m1", choices=["m1", "m2", "m3"])
    parser.add_argument("--split",          default="test")
    parser.add_argument("--max_gen_len",    type=int,   default=None)
    parser.add_argument("--len_multiplier", type=float, default=1.2)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_lora",   action="store_true",
                        help="Run 9: reconstruct LoRA architecture before loading checkpoint.")
    parser.add_argument("--lora_rank",  type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Mode  : {args.condition_mode}")

    model, tokenizer = load_model(
        args.ckpt_path, args.config_path, device,
        use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )

    df      = pd.read_csv(args.pairs_csv)
    test_df = df[df["split"] == args.split].reset_index(drop=True)
    print(f"Test pairs: {len(test_df)}")

    sections_dir = Path(args.sections_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for row_idx, row in test_df.iterrows():
        ctx_stem   = Path(row["context_file"]).stem
        tgt_stem   = Path(row["target_file"]).stem
        sim_bucket = row["sim_bucket"]
        pair_id    = f"{ctx_stem}_vs_{tgt_stem}"

        print(f"\n-- [{row_idx}] {pair_id}  ({sim_bucket}) --")

        ctx_path = sections_dir / row["context_file"]
        tgt_path = sections_dir / row["target_file"]

        if not ctx_path.exists():
            print(f"  SKIP: context not found: {ctx_path}")
            continue
        if not tgt_path.exists():
            print(f"  SKIP: target not found: {tgt_path}")
            continue

        ctx_tokens = np.load(str(ctx_path)).astype(np.int64)
        tgt_tokens = np.load(str(tgt_path)).astype(np.int64)
        tgt_norm   = _normalize_onsets(tgt_tokens)

        # Build prompt
        prompt_ids = build_prompt(
            tokenizer, ctx_tokens, sim_bucket, args.condition_mode
        ).to(device)

        # Compute prototype vector once (None for M2)
        proto_vec = None
        if args.condition_mode != "m2":
            ctx_norm_t = _normalize_onsets(ctx_tokens)
            max_ctx    = 509
            ctx_only   = torch.tensor(
                [ctx_norm_t[:max_ctx].tolist()], dtype=torch.long, device=device
            )  # (1, L_ctx, 6)
            with torch.inference_mode():
                proto_vec = model.compute_proto_vec(ctx_only, SECTION_TOKEN_MAP)
                # Match model dtype
                if next(model.parameters()).dtype == torch.bfloat16:
                    proto_vec = proto_vec.to(torch.bfloat16)

        # Bar size and target bar count for PROTO insertion during generation
        # pairs.csv has no bars columns -- parse from filenames
        ctx_bars_str = _bars_from_filename(row["context_file"])
        tgt_bars_str = _bars_from_filename(row["target_file"])
        bar_size_est = _estimate_bar_size(ctx_tokens, ctx_bars_str)
        target_bars  = _parse_num_bars(tgt_bars_str)

        gt_len     = len(tgt_tokens)
        piece_max  = (
            args.max_gen_len
            if args.max_gen_len is not None
            else max(1, int(gt_len * args.len_multiplier))
        )

        ctx_used = len(_normalize_onsets(ctx_tokens)[:509])
        print(f"  Prompt : {prompt_ids.shape[1]} tokens (ctx={ctx_used})")
        print(f"  GT len : {gt_len} notes  target_bars={target_bars}  bar_size~{bar_size_est:.0f}")
        print(f"  Budget : {piece_max} tokens")

        generated = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            additional_token_map=SECTION_TOKEN_MAP,
            proto_vec=proto_vec,
            bar_size_est=bar_size_est,
            target_bars=target_bars,
            condition_mode=args.condition_mode,
            max_gen_len=piece_max,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        stop_reason = "EOS" if len(generated) < piece_max else "budget"
        print(f"  Output : {len(generated)} tokens ({stop_reason})")

        if generated:
            gen_path = Path(args.output_dir) / f"{pair_id}_generated.mid"
            try:
                tokenizer.compound_to_midi(generated).save(str(gen_path))
                print(f"  Saved  : {gen_path}")
            except Exception as e:
                print(f"  WARNING: MIDI save failed for {pair_id}: {e} -- skipping")
        else:
            print(f"  WARNING: empty output for {pair_id}")

        gt_path = Path(args.output_dir) / f"{pair_id}_groundtruth.mid"
        tokenizer.compound_to_midi(tgt_norm.tolist()).save(str(gt_path))
        print(f"  GT     : {gt_path}")


if __name__ == "__main__":
    main()
