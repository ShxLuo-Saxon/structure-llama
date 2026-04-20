r"""
scripts/inference/infer_triplet.py

Matched-triplet inference for Ctx-sim and GT-sim evaluation.

For each unique context section in the test split:
  - M1: generate one output per bucket label (sim:high, sim:mid, sim:low)
  - M2: generate one output unconditionally (baseline)
  - M3: generate one output per bucket label with wrong labels (ablation)

This gives perfectly matched triplets: same context × 3 bucket labels × free generation.
Downstream eval (eval_generation.py) computes compound_sim(context, generated) and
compound_sim(generated, GT_target), testing whether M1 follows the bucket label.

Generation budget: mean target_ntokens across all test pairs for each context × len_multiplier.
PROTO bar estimate: mean target_bars across all test pairs for each context.

Output naming:
  {ctx_stem}_m1_high_generated.mid
  {ctx_stem}_m1_mid_generated.mid
  {ctx_stem}_m1_low_generated.mid
  {ctx_stem}_m2_generated.mid         (M2 unconditional, confound check)
  {ctx_stem}_m3_high_generated.mid    (M3 wrong-label ablation)
  {ctx_stem}_m3_mid_generated.mid
  {ctx_stem}_m3_low_generated.mid

Usage (HPC, run once for M1+M2+M3 together):
    python scripts/inference/infer_triplet.py \
        --ckpt_m1      $SCRATCH/runs/m1/best.pt \
        --ckpt_m2      $SCRATCH/runs/m2/best.pt \
        --ckpt_m3      $SCRATCH/runs/m3/best.pt \
        --config_path  scripts/configs/model_config_structure_llama_839M.json \
        --sections_dir $SCRATCH/data/sections \
        --pairs_csv    $SCRATCH/data/sections/pairs.csv \
        --output_dir   $SCRATCH/outputs/triplet \
        --split        test
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.dataset import (
    SECTION_TOKEN_MAP,
    _bars_from_filename, _parse_num_bars, _normalize_onsets,
)

# Re-use helpers from the original inference script
from scripts.inference.infer_structure_llama import (
    sample_top_p, load_model, build_prompt,
    _estimate_bar_size, generate_one,
)

BUCKETS = ["sim:high", "sim:mid", "sim:low"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_m1",      required=True,
                        help="M1 best.pt (correctly conditioned)")
    parser.add_argument("--ckpt_m2",      required=True,
                        help="M2 best.pt (unconditional baseline)")
    parser.add_argument("--ckpt_m3",      required=True,
                        help="M3 best.pt (wrong-label ablation)")
    parser.add_argument("--config_path",  required=True)
    parser.add_argument("--sections_dir", required=True)
    parser.add_argument("--pairs_csv",    required=True)
    parser.add_argument("--output_dir",   default="outputs/generated/triplet")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--len_multiplier", type=float, default=1.2)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    parser.add_argument("--seed",         type=int,   default=42)
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

    # ── Load all three models ─────────────────────────────────────────────────
    # Checkpoints use LoRA (r=8, alpha=16) -- apply_lora() must be called
    # before load_state_dict so the architecture matches the saved keys.
    print("Loading M1 ...")
    model_m1, tokenizer = load_model(args.ckpt_m1, args.config_path, device,
                                     use_lora=True, lora_rank=8, lora_alpha=16)
    print("Loading M2 ...")
    model_m2, _         = load_model(args.ckpt_m2, args.config_path, device,
                                     use_lora=True, lora_rank=8, lora_alpha=16)
    print("Loading M3 ...")
    model_m3, _         = load_model(args.ckpt_m3, args.config_path, device,
                                     use_lora=True, lora_rank=8, lora_alpha=16)

    # ── Build per-context stats from pairs_csv ────────────────────────────────
    pairs_df   = pd.read_csv(args.pairs_csv)
    test_df    = pairs_df[pairs_df["split"] == args.split].copy()
    test_df["context_stem"] = test_df["context_file"].apply(lambda f: Path(f).stem)
    test_df["target_bars"]  = test_df["target_file"].apply(
        lambda f: _parse_num_bars(_bars_from_filename(f))
    )

    # Mean target length (notes) and mean target bars per context
    ctx_stats = test_df.groupby("context_stem").agg(
        mean_target_ntokens=("target_ntokens", "mean"),
        mean_target_bars   =("target_bars",    "mean"),
        context_file       =("context_file",   "first"),
    ).reset_index()

    print(f"\nUnique contexts in {args.split} split: {len(ctx_stats)}")
    sections_dir = Path(args.sections_dir)

    # ── Generate ──────────────────────────────────────────────────────────────
    for _, row in ctx_stats.iterrows():
        ctx_stem   = row["context_stem"]
        ctx_path   = sections_dir / row["context_file"]

        if not ctx_path.exists():
            print(f"SKIP (missing npy): {ctx_path}")
            continue

        ctx_tokens   = np.load(str(ctx_path)).astype(np.int64)
        ctx_bars_str = _bars_from_filename(row["context_file"])
        bar_size_est = _estimate_bar_size(ctx_tokens, ctx_bars_str)
        target_bars  = max(1, round(row["mean_target_bars"]))
        budget       = max(1, int(row["mean_target_ntokens"] * args.len_multiplier))

        print(f"\n== {ctx_stem}  budget={budget}  est_bars={target_bars} ==")

        # ── Prototype vector for M1 (shared across buckets) ───────────────────
        ctx_norm_t = _normalize_onsets(ctx_tokens)
        ctx_only   = torch.tensor(
            [ctx_norm_t[:509].tolist()], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            proto_vec = model_m1.compute_proto_vec(ctx_only, SECTION_TOKEN_MAP)
            if next(model_m1.parameters()).dtype == torch.bfloat16:
                proto_vec = proto_vec.to(torch.bfloat16)

        # ── M3 prototype (computed from M3's own encoder) ────────────────────
        with torch.inference_mode():
            proto_vec_m3 = model_m3.compute_proto_vec(ctx_only, SECTION_TOKEN_MAP)
            if next(model_m3.parameters()).dtype == torch.bfloat16:
                proto_vec_m3 = proto_vec_m3.to(torch.bfloat16)

        # ── M1 and M3: generate for each bucket ───────────────────────────────
        # M3 receives correct bucket labels at inference but was trained on
        # shuffled labels -- if it shows no ordering, bucket semantics are learned.
        for cmode, model_ref, proto in [("m1", model_m1, proto_vec),
                                        ("m3", model_m3, proto_vec_m3)]:
            for bucket in BUCKETS:
                bkt_label = bucket.split(":")[1]   # "high" / "mid" / "low"
                out_path  = out_dir / f"{ctx_stem}_{cmode}_{bkt_label}_generated.mid"
                if out_path.exists():
                    print(f"  [skip existing] {out_path.name}")
                    continue

                prompt_ids = build_prompt(
                    tokenizer, ctx_tokens, bucket, "m1"   # m1 format for both (bucket token injected)
                ).to(device)

                generated = generate_one(
                    model=model_ref,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    additional_token_map=SECTION_TOKEN_MAP,
                    proto_vec=proto,
                    bar_size_est=bar_size_est,
                    target_bars=target_bars,
                    condition_mode="m1",
                    max_gen_len=budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )

                print(f"  {cmode}/{bkt_label}: {len(generated)} notes", end="  ")
                if generated:
                    try:
                        tokenizer.compound_to_midi(generated).save(str(out_path))
                        print(f"-> {out_path.name}")
                    except Exception as e:
                        print(f"MIDI save failed: {e}")
                else:
                    print("empty output -- skipped")

        # ── M2: generate once (unconditional, no bucket label) ────────────────
        m2_path = out_dir / f"{ctx_stem}_m2_generated.mid"  # no bucket suffix needed
        if not m2_path.exists():
            prompt_ids_m2 = build_prompt(
                tokenizer, ctx_tokens, "sim:high", "m2"   # bucket arg ignored for m2
            ).to(device)

            generated_m2 = generate_one(
                model=model_m2,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids_m2,
                additional_token_map=SECTION_TOKEN_MAP,
                proto_vec=None,
                bar_size_est=bar_size_est,
                target_bars=target_bars,
                condition_mode="m2",
                max_gen_len=budget,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            print(f"  M2:       {len(generated_m2)} notes", end="  ")
            if generated_m2:
                try:
                    tokenizer.compound_to_midi(generated_m2).save(str(m2_path))
                    print(f"-> {m2_path.name}")
                except Exception as e:
                    print(f"MIDI save failed: {e}")
            else:
                print("empty output -- skipped")

    print("\nDone.")


if __name__ == "__main__":
    main()
