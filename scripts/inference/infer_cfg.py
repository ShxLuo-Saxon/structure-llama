r"""
scripts/inference/infer_cfg.py

Classifier-Free Guidance (CFG) inference.

Combines M1 (conditioned: bucket label + PROTO) and M2 (unconditional) logits
at each GRU attribute step:

    logit_cfg = logit_M2 + alpha * (logit_M1 - logit_M2)

alpha=1.0 recovers standard M1.  alpha>1 amplifies the conditioning signal.
Tests whether the M1/M2 logit divergence encodes bucket information that can be
sharpened at inference without any retraining.

Output naming:
  {ctx_stem}_cfg{alpha}_high_generated.mid
  {ctx_stem}_cfg{alpha}_low_generated.mid

Usage (HPC):
    python scripts/inference/infer_cfg.py \
        --ckpt_m1      $SCRATCH/runs/m1/best.pt \
        --ckpt_m2      $SCRATCH/runs/m2/best.pt \
        --config_path  scripts/configs/model_config_structure_llama_839M.json \
        --sections_dir $SCRATCH/data/sections \
        --pairs_csv    $SCRATCH/data/sections/pairs.csv \
        --output_dir   $SCRATCH/outputs/cfg \
        --split        test \
        --cfg_alphas   1.5 2.0 3.0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.structure_llama import StructureLlama, PROTO_TOKEN_ID
from scripts.model.tokenizer import MusicTokenizer
from scripts.training.dataset import (
    SECTION_TOKEN_MAP,
    _bars_from_filename, _parse_num_bars, _normalize_onsets,
)
from scripts.inference.infer_structure_llama import (
    sample_top_p, load_model, build_prompt, _estimate_bar_size,
)


# ──────────────────────────────────────────────────────────────
# CFG generation
# ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def _encode_and_init_gru(
    model: StructureLlama,
    all_ids: torch.Tensor,
    proto_vec: Optional[torch.Tensor],
    num_gru_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode all_ids through Llama, return (projected, gru_h_init)."""
    projected = model.encode(all_ids, SECTION_TOKEN_MAP, proto_vec=proto_vec)
    gru_h = (
        projected[:, -1, :]
        .unsqueeze(0)
        .expand(num_gru_layers, -1, -1)
        .contiguous()
    )
    return projected, gru_h


@torch.inference_mode()
def generate_cfg(
    model_m1: StructureLlama,
    model_m2: StructureLlama,
    tokenizer: MusicTokenizer,
    prompt_m1: torch.Tensor,       # (1, T1, 6) with bucket + PROTO
    prompt_m2: torch.Tensor,       # (1, T2, 6) no bucket, no PROTO
    proto_vec: torch.Tensor,       # (1, hidden)  precomputed
    alpha: float,
    bar_size_est: float,
    target_bars: int,
    max_gen_len: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """
    CFG generation: logit = logit_M2 + alpha*(logit_M1 - logit_M2).

    Both sequences grow identically with generated notes.
    M1 sequence additionally receives PROTO tokens at bar boundaries.
    """
    num_layers = model_m1.config.decoder.num_hidden_layers
    valid_sets = [
        list(tokenizer.timeshift_dict.values()),
        list(tokenizer.duration_dict.values()),
        list(tokenizer.octave_dict.values()),
        list(tokenizer.pitch_dict.values()),
        list(tokenizer.instrument_dict.values()),
        list(tokenizer.velocity_dict.values()),
    ]
    valid_tensors = [torch.tensor(vs, dtype=torch.long, device=device) for vs in valid_sets]
    eos_idx = tokenizer.timeshift_dict[tokenizer.eos_timeshift]

    ids_m1 = prompt_m1.to(device)
    ids_m2 = prompt_m2.to(device)

    cumulative_onset = 0
    next_bar_onset   = bar_size_est
    bars_inserted    = 1
    generated: List[List[int]] = []

    for step in range(max_gen_len):

        # --- PROTO insertion at bar boundaries (M1 only) ---
        if bars_inserted < target_bars and cumulative_onset >= next_bar_onset:
            proto_tok = torch.tensor(
                [[[PROTO_TOKEN_ID] * 6]], dtype=torch.long, device=device
            )
            ids_m1 = torch.cat([ids_m1, proto_tok], dim=1)
            next_bar_onset   += bar_size_est
            bars_inserted    += 1

        # --- Llama encode both sequences ---
        _, gru_h_m1 = _encode_and_init_gru(model_m1, ids_m1, proto_vec,  num_layers)
        _, gru_h_m2 = _encode_and_init_gru(model_m2, ids_m2, None,        num_layers)

        # --- GRU decode 6 attributes ---
        gru_in = torch.tensor([[tokenizer.sos_out]], dtype=torch.long, device=device)
        lang_tokens: List[int] = []

        for attr_idx in range(6):
            # M1 forward
            out_m1   = model_m1._inference_forward(gru_in, gru_h_m1)
            gru_h_m1 = out_m1.generation_hidden_state
            logit_m1 = out_m1.generation_logits[:, -1, :].float()

            # M2 forward
            out_m2   = model_m2._inference_forward(gru_in, gru_h_m2)
            gru_h_m2 = out_m2.generation_hidden_state
            logit_m2 = out_m2.generation_logits[:, -1, :].float()

            # CFG combination
            logit_cfg = logit_m2 + alpha * (logit_m1 - logit_m2)

            # Mask to valid tokens for this attribute
            masked = torch.full_like(logit_cfg, float("-inf"))
            idx    = valid_tensors[attr_idx]
            masked.scatter_(1, idx.unsqueeze(0), logit_cfg[:, idx])

            if temperature > 0:
                probs = torch.softmax(masked / temperature, dim=-1)
                tok   = sample_top_p(probs, top_p)
            else:
                tok = masked.argmax(dim=-1, keepdim=True)

            lang_tokens.append(tok.item())
            gru_in = tok

        if lang_tokens[0] == eos_idx:
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
            cumulative_onset, int(raw_vals[1]), int(raw_vals[2]),
            int(raw_vals[3]),  int(raw_vals[4]), int(raw_vals[5]),
        ]
        generated.append(abs_compound)

        new_tok = (
            torch.tensor(abs_compound, dtype=torch.long, device=device)
            .unsqueeze(0).unsqueeze(0)
        )
        ids_m1 = torch.cat([ids_m1, new_tok], dim=1)
        ids_m2 = torch.cat([ids_m2, new_tok], dim=1)

    return generated


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_m1",      required=True, help="M1 best.pt")
    parser.add_argument("--ckpt_m2",      required=True, help="M2 best.pt")
    parser.add_argument("--config_path",  required=True)
    parser.add_argument("--sections_dir", required=True)
    parser.add_argument("--pairs_csv",    required=True)
    parser.add_argument("--output_dir",   default="outputs/generated/cfg")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--cfg_alphas",   type=float, nargs="+", default=[1.5, 2.0, 3.0],
                        help="CFG alpha values to run (e.g. 1.5 2.0 3.0)")
    parser.add_argument("--buckets",     nargs="+", default=["high", "low"],
                        choices=["high", "mid", "low"],
                        help="Which bucket labels to generate for")
    parser.add_argument("--len_multiplier", type=float, default=1.2)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CFG alphas : {args.cfg_alphas}")
    print(f"Buckets    : {args.buckets}")
    print(f"Device     : {device}")

    # Load models (LoRA r=8, alpha=16)
    print("\nLoading M1 (LoRA r=8, use_proto=True) ...")
    model_m1, tokenizer = load_model(
        args.ckpt_m1, args.config_path, device,
        use_lora=True, lora_rank=8, lora_alpha=16, use_proto=True,
    )
    print("Loading M2 (LoRA r=8, use_proto=False) ...")
    model_m2, _ = load_model(
        args.ckpt_m2, args.config_path, device,
        use_lora=True, lora_rank=8, lora_alpha=16, use_proto=False,
    )

    pairs_df = pd.read_csv(args.pairs_csv)
    test_df  = pairs_df[pairs_df["split"] == args.split].copy()
    test_df["context_stem"] = test_df["context_file"].apply(lambda f: Path(f).stem)
    test_df["target_bars"]  = test_df["target_file"].apply(
        lambda f: _parse_num_bars(_bars_from_filename(f))
    )
    ctx_stats = test_df.groupby("context_stem").agg(
        mean_target_ntokens=("target_ntokens", "mean"),
        mean_target_bars   =("target_bars",    "mean"),
        context_file       =("context_file",   "first"),
    ).reset_index()

    sections_dir = Path(args.sections_dir)
    print(f"\nUnique contexts in {args.split} split: {len(ctx_stats)}")
    print(f"CFG outputs to generate: {len(args.cfg_alphas) * len(args.buckets) * len(ctx_stats)}")

    for ctx_idx, (_, row) in enumerate(ctx_stats.iterrows()):
        ctx_stem = row["context_stem"]
        ctx_path = sections_dir / row["context_file"]

        if not ctx_path.exists():
            print(f"SKIP (missing npy): {ctx_path}")
            continue

        ctx_tokens   = np.load(str(ctx_path)).astype(np.int64)
        ctx_bars_str = _bars_from_filename(row["context_file"])
        bar_size_est = _estimate_bar_size(ctx_tokens, ctx_bars_str)
        target_bars  = max(1, round(row["mean_target_bars"]))
        budget       = max(1, int(row["mean_target_ntokens"] * args.len_multiplier))

        ctx_norm_t = _normalize_onsets(ctx_tokens)
        ctx_only   = torch.tensor(
            [ctx_norm_t[:509].tolist()], dtype=torch.long, device=device
        )
        proto_vec = model_m1.compute_proto_vec(ctx_only, SECTION_TOKEN_MAP)
        if next(model_m1.parameters()).dtype == torch.bfloat16:
            proto_vec = proto_vec.to(torch.bfloat16)

        print(f"\n[{ctx_idx+1}/{len(ctx_stats)}] {ctx_stem}  "
              f"budget={budget}  bars={target_bars}  bar_size~{bar_size_est:.0f}")

        for alpha in args.cfg_alphas:
            alpha_tag = f"{alpha:.1f}".replace(".", "p")
            for bucket in args.buckets:
                sim_bucket = f"sim:{bucket}"
                out_path = out_dir / f"{ctx_stem}_cfg{alpha_tag}_{bucket}_generated.mid"
                if out_path.exists():
                    print(f"  [skip] {out_path.name}")
                    continue

                prompt_m1 = build_prompt(
                    tokenizer, ctx_tokens, sim_bucket, "m1", use_proto=True,
                ).to(device)
                prompt_m2 = build_prompt(
                    tokenizer, ctx_tokens, sim_bucket, "m2", use_proto=False,
                ).to(device)

                gen = generate_cfg(
                    model_m1, model_m2, tokenizer,
                    prompt_m1, prompt_m2, proto_vec,
                    alpha=alpha,
                    bar_size_est=bar_size_est,
                    target_bars=target_bars,
                    max_gen_len=budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
                print(f"  CFG α={alpha} {bucket}: {len(gen)} notes", end="  ")
                if gen:
                    try:
                        tokenizer.compound_to_midi(gen).save(str(out_path))
                        print(f"-> {out_path.name}")
                    except Exception as e:
                        print(f"MIDI save failed: {e}")
                else:
                    print("empty -- skipped")

    print("\nDone.")
    print(f"Output dir : {out_dir}")
    print(f"File count : {len(list(out_dir.glob('*.mid')))}")


if __name__ == "__main__":
    main()