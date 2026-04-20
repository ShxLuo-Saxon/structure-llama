r"""
scripts/eval/eval_generation.py

Combined Ctx-sim and GT-sim evaluation for StructureLlama triplet outputs.

Two evaluation modes run in a single pass:

  1. Ctx-sim (GT-free):  compound_sim(context_tokens, generated_tokens)
     Tests whether the bucket label modulates similarity to the context section.

  2. GT-sim:             compound_sim(generated_tokens, GT_target_tokens)
     Tests whether the bucket label leads to style-appropriate output vs ground truth.

Triplet output naming convention:
  {ctx_stem}_m1_{high|mid|low}_generated.mid
  {ctx_stem}_m2_generated.mid
  {ctx_stem}_m3_{high|mid|low}_generated.mid

GT targets are loaded directly from sections_dir .npy files (no groundtruth.mid needed).

Optional truncation (applied to generated output before ALL compound_sim scoring):
  --truncate_to_ctx        truncate gen at context span (recommended -- removes drift tail)
  --truncate_fixed_ticks N truncate gen at N ticks (e.g. 1600 = 16s at 100 ticks/s)

Output files (with default --out_csv outputs/eval/results_tests.csv):
  outputs/eval/results_tests.csv      — Wilcoxon test results (Table 3)
  outputs/eval/results_ctx_sim.csv    — raw per-context ctx-sim scores (Table 2 / Figure 3a)
  outputs/eval/results_gt_sim.csv     — raw per-pair GT-sim scores (Table 2 / Figure 3b)
  outputs/eval/results_tests_cross_gen.csv — cross-gen divergence scores

Override with --ctx_csv and --gt_csv if needed.

Usage:
    python scripts/eval/eval_generation.py \
        --triplet_dir  outputs/generated/triplet \
        --sections_dir <SECTIONS_DIR> \
        --pairs_csv    <SECTIONS_DIR>/pairs.csv \
        --out_csv      outputs/eval/results_tests.csv \
        --truncate_to_ctx
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, skew as sp_skew

from scripts.eval.compound_metric import compound_sim  # noqa: E402


# ── MIDI -> compound tokens ────────────────────────────────────────────────────

def midi_to_compound(midi_path: Path) -> np.ndarray:
    """Load generated MIDI and return (N, 6) compound tokens.

    Generated MIDIs use ticks_per_beat=50 (from tokenizer.compound_to_midi with
    TIME_RESOLUTION=100).  We do NOT rescale to 480 tpb — keeping the native
    tick scale so that onset/duration values match the .npy reference files
    (which also use TIME_RESOLUTION=100).
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return np.zeros((0, 6), dtype=np.int64)

    notes: list[list[int]] = []
    for track in mid.tracks:
        abs_tick = 0
        active: dict[tuple[int, int], int] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    onset = active.pop(key)
                    duration = max(1, abs_tick - onset)
                    octave = msg.note // 12
                    pitch = msg.note % 12
                    velocity = min(getattr(msg, "velocity", 64), 127)
                    notes.append([onset, duration, octave, pitch, msg.channel % 16, velocity])

    if not notes:
        return np.zeros((0, 6), dtype=np.int64)
    return np.array(sorted(notes, key=lambda x: x[0]), dtype=np.int64)


# ── Wilcoxon helper ────────────────────────────────────────────────────────────

def wtest(a: list[float], b: list[float], label: str,
          alternative: str = "greater") -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 5:
        print(f"  {label}: n={n} too small -- skipped")
        return {}
    stat, p = wilcoxon(diffs, alternative=alternative)
    med       = float(np.median(diffs))
    frac_pos  = float(np.mean([d > 0 for d in diffs]))
    skewness  = float(sp_skew(diffs))
    sig = "**" if p < 0.05 else "  "
    print(f"  {sig}{label}: n={n}  med_diff={med:+.4f}  W={stat:.0f}  p={p:.4f}  "
          f"frac_pos={frac_pos:.2f}  skew={skewness:+.2f}{sig}")
    return {"label": label, "n": n, "median_diff": med, "W": stat, "p": p,
            "alternative": alternative, "frac_pos": frac_pos, "skew": skewness}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet_dir",  required=True,
                        help="Flat dir containing all triplet .mid files")
    parser.add_argument("--sections_dir", required=True,
                        help="Dir containing context and target .npy files")
    parser.add_argument("--pairs_csv",    required=True,
                        help="Path to pairs.csv")
    parser.add_argument("--out_csv",      default="outputs/eval/results_tests.csv",
                        help="Output path for Wilcoxon test results")
    parser.add_argument("--ctx_csv",      default=None,
                        help="Output path for raw ctx-sim scores "
                             "(default: results_ctx_sim.csv beside --out_csv)")
    parser.add_argument("--gt_csv",       default=None,
                        help="Output path for raw GT-sim scores "
                             "(default: results_gt_sim.csv beside --out_csv)")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--min_notes",    type=int, default=10)
    parser.add_argument("--truncate_to_ctx", action="store_true",
                        help="Truncate generated output to context span before ctx-sim. "
                             "Removes post-PROTO drift tail. Applied only to ctx-sim scoring.")
    parser.add_argument("--truncate_fixed_ticks", type=int, default=0,
                        help="Truncate generated output to this many ticks (e.g. 1600=16s at 100t/s). "
                             "Applied instead of --truncate_to_ctx when > 0.")
    args = parser.parse_args()

    triplet_dir = Path(args.triplet_dir)
    sec_dir     = Path(args.sections_dir)

    # ── Parse generated filenames ─────────────────────────────────────────────
    pat_m1m3 = re.compile(r"^(.+)_(m[13])_(high|mid|low)_generated\.mid$")
    pat_m2   = re.compile(r"^(.+)_(m2)_generated\.mid$")

    gen_files: dict[tuple[str, str, str], Path] = {}  # (ctx_stem, model, bucket) -> path
    for f in sorted(triplet_dir.glob("*_generated.mid")):
        m = pat_m1m3.match(f.name)
        if m:
            gen_files[(m.group(1), m.group(2), m.group(3))] = f
            continue
        m = pat_m2.match(f.name)
        if m:
            gen_files[(m.group(1), "m2", "none")] = f

    stems = sorted({k[0] for k in gen_files})
    print(f"Found {len(gen_files)} generated files, {len(stems)} unique contexts")

    # ── Load context .npy (for ctx-sim) ──────────────────────────────────────
    ctx_tokens: dict[str, np.ndarray] = {}
    for stem in stems:
        npy = sec_dir / f"{stem}.npy"
        if npy.exists():
            ctx_tokens[stem] = np.load(str(npy))
    print(f"Loaded {len(ctx_tokens)}/{len(stems)} context .npy files")

    # ── Load pairs_csv (for GT-based eval) ───────────────────────────────────
    pairs_df = pd.read_csv(args.pairs_csv)
    pairs_df = pairs_df[pairs_df["split"] == args.split].reset_index(drop=True)
    pairs_df["ctx_stem"] = pairs_df["context_file"].apply(lambda f: Path(f).stem)
    pairs_df["bucket"]   = pairs_df["sim_bucket"].str.replace("sim:", "", regex=False)
    print(f"Test pairs: {len(pairs_df)}")

    # ── Context-level anomaly filter ──────────────────────────────────────────
    # Exclude a context entirely if ANY of its 7 generated outputs is too short.
    # Per-file filtering (min_notes) is insufficient: selective collapse in one
    # condition (e.g. m3-mid collapses to 4 notes, m3-high/low are fine) biases
    # the paired tests because different conditions lose different data points.
    bad_stems: set[str] = set()
    for (ctx_stem, model, bucket), midi_path in gen_files.items():
        gen = midi_to_compound(midi_path)
        if len(gen) < args.min_notes:
            bad_stems.add(ctx_stem)
    if bad_stems:
        print(f"Excluded {len(bad_stems)} contexts (any output < {args.min_notes} notes): "
              f"{sorted(bad_stems)}")

    # ── Main scoring loop ─────────────────────────────────────────────────────
    ctx_sim_rows = []   # {ctx_stem, model, bucket, ctx_sim}
    gt_rows      = []   # {ctx_stem, model, bucket, target_stem, gt_sim}
    skipped_gen  = 0
    skipped_gt   = 0

    all_ctx_stems = sorted({k[0] for k in gen_files})

    for ctx_stem in all_ctx_stems:
        if ctx_stem not in ctx_tokens:
            continue
        if ctx_stem in bad_stems:
            skipped_gen += sum(1 for k in gen_files if k[0] == ctx_stem)
            continue
        ctx_tok = ctx_tokens[ctx_stem]

        ctx_span = int(ctx_tok[-1, 0] - ctx_tok[0, 0])  # ticks, same scale as generated

        # ctx-sim for all (model, bucket) combos for this context
        for (s, model, bucket), midi_path in gen_files.items():
            if s != ctx_stem:
                continue
            gen = midi_to_compound(midi_path)
            if len(gen) > 0 and (args.truncate_to_ctx or args.truncate_fixed_ticks > 0):
                gen_norm = gen.copy()
                gen_norm[:, 0] -= gen_norm[0, 0]
                cutoff = ctx_span if args.truncate_to_ctx else args.truncate_fixed_ticks
                gen = gen_norm[gen_norm[:, 0] <= cutoff]
            if len(gen) < args.min_notes:
                skipped_gen += 1
                continue
            score = compound_sim(ctx_tok, gen)
            ctx_sim_rows.append({
                "ctx_stem": ctx_stem, "model": model,
                "bucket": bucket, "ctx_sim": score,
            })

        # GT-based: iterate over test pairs for this context
        ctx_pairs = pairs_df[pairs_df["ctx_stem"] == ctx_stem]
        for _, pair_row in ctx_pairs.iterrows():
            bucket  = pair_row["bucket"]
            tgt_npy = sec_dir / pair_row["target_file"]
            if not tgt_npy.exists():
                skipped_gt += 1
                continue
            gt_tok = np.load(str(tgt_npy)).astype(np.int64)

            for model in ("m1", "m2", "m3"):
                if model == "m2":
                    key = (ctx_stem, "m2", "none")
                else:
                    key = (ctx_stem, model, bucket)
                if key not in gen_files:
                    continue
                gen = midi_to_compound(gen_files[key])
                if len(gen) > 0 and (args.truncate_to_ctx or args.truncate_fixed_ticks > 0):
                    gen_norm = gen.copy()
                    gen_norm[:, 0] -= gen_norm[0, 0]
                    cutoff = ctx_span if args.truncate_to_ctx else args.truncate_fixed_ticks
                    gen = gen_norm[gen_norm[:, 0] <= cutoff]
                if len(gen) < args.min_notes:
                    continue
                gt_rows.append({
                    "ctx_stem":    ctx_stem,
                    "model":       model,
                    "bucket":      bucket,
                    "target_stem": Path(pair_row["target_file"]).stem,
                    "gt_sim":      compound_sim(gen, gt_tok),
                })

    print(f"ctx-sim scores: {len(ctx_sim_rows)}  ({skipped_gen} skipped short/missing)")
    print(f"GT-based scores: {len(gt_rows)}  ({skipped_gt} GT .npy missing)")

    ctx_df = pd.DataFrame(ctx_sim_rows)
    gt_df  = pd.DataFrame(gt_rows)

    out_path  = Path(args.out_csv)
    ctx_path  = Path(args.ctx_csv) if args.ctx_csv else out_path.parent / "results_ctx_sim.csv"
    gt_path   = Path(args.gt_csv)  if args.gt_csv  else out_path.parent / "results_gt_sim.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_df.to_csv(ctx_path, index=False)
    gt_df.to_csv(gt_path, index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: CTX-SIM RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 1: CTX-SIM  compound_sim(context, generated)")
    print("="*60)

    print("\n=== Mean ctx_sim by model and bucket ===")
    pivot = (ctx_df[ctx_df["bucket"] != "none"]
             .groupby(["model", "bucket"])["ctx_sim"]
             .agg(["mean", "std", "count"]))
    print(pivot.to_string())
    m2_mean = ctx_df[ctx_df["model"] == "m2"]["ctx_sim"].mean()
    m2_n    = len(ctx_df[ctx_df["model"] == "m2"])
    print(f"\nM2 (unconditional, no bucket): mean={m2_mean:.4f}  n={m2_n}")

    def collect_paired_ctx(model: str, bkt_a: str, bkt_b: str):
        sub  = ctx_df[ctx_df["model"] == model]
        a_map = dict(zip(sub[sub["bucket"] == bkt_a]["ctx_stem"],
                         sub[sub["bucket"] == bkt_a]["ctx_sim"]))
        b_map = dict(zip(sub[sub["bucket"] == bkt_b]["ctx_stem"],
                         sub[sub["bucket"] == bkt_b]["ctx_sim"]))
        common = sorted(set(a_map) & set(b_map))
        return [a_map[s] for s in common], [b_map[s] for s in common]

    ctx_results = []
    print("\n=== H1_ctx: M1 bucket ordering (paired by context) ===")
    for bkt_a, bkt_b, label in [("high", "low",  "M1 high > low"),
                                  ("high", "mid",  "M1 high > mid"),
                                  ("mid",  "low",  "M1 mid  > low")]:
        va, vb = collect_paired_ctx("m1", bkt_a, bkt_b)
        r = wtest(va, vb, label, "greater")
        if r:
            ctx_results.append(r)

    print("\n=== M3 reversed ordering (theoretical: low > high) ===")
    for bkt_a, bkt_b, label in [("low",  "high", "M3 low  > high (reversed)"),
                                  ("low",  "mid",  "M3 low  > mid"),
                                  ("mid",  "high", "M3 mid  > high")]:
        va, vb = collect_paired_ctx("m3", bkt_a, bkt_b)
        r = wtest(va, vb, label, "greater")
        if r:
            ctx_results.append(r)

    print("\n=== M1 vs M2 overall ctx-sim (one-sided, paired by context) ===")
    m1_ctx_mean = ctx_df[ctx_df["model"] == "m1"].groupby("ctx_stem")["ctx_sim"].mean()
    m2_ctx      = ctx_df[ctx_df["model"] == "m2"].set_index("ctx_stem")["ctx_sim"]
    common      = sorted(set(m1_ctx_mean.index) & set(m2_ctx.index))
    r = wtest([float(m1_ctx_mean[s]) for s in common],
              [float(m2_ctx[s])      for s in common],
              "M1 > M2 (mean over buckets)", "greater")
    if r:
        ctx_results.append(r)

    print("\n=== M1 vs M3 overall ctx-sim (one-sided, paired by context) ===")
    m3_ctx_mean = ctx_df[ctx_df["model"] == "m3"].groupby("ctx_stem")["ctx_sim"].mean()
    common      = sorted(set(m1_ctx_mean.index) & set(m3_ctx_mean.index))
    r = wtest([float(m1_ctx_mean[s]) for s in common],
              [float(m3_ctx_mean[s]) for s in common],
              "M1 > M3 (mean over buckets)", "greater")
    if r:
        ctx_results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: GT-BASED RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 2: GT-BASED  compound_sim(generated, GT_target)")
    print("="*60)

    print("\n=== Mean gt_sim by model and bucket ===")
    pivot_gt = gt_df.groupby(["model", "bucket"])["gt_sim"].agg(["mean", "std", "count"])
    print(pivot_gt.to_string())

    print("\n=== Overall mean gt_sim by model ===")
    print(gt_df.groupby("model")["gt_sim"].mean().to_string())

    # H1_GT: M1 high vs low, paired by context (mean over GT targets per context-bucket)
    def collect_paired_gt(model: str, bkt_a: str, bkt_b: str):
        sub = gt_df[gt_df["model"] == model]
        # Average over multiple GT targets per (context, bucket)
        mean_a = sub[sub["bucket"] == bkt_a].groupby("ctx_stem")["gt_sim"].mean()
        mean_b = sub[sub["bucket"] == bkt_b].groupby("ctx_stem")["gt_sim"].mean()
        common = sorted(set(mean_a.index) & set(mean_b.index))
        return [float(mean_a[s]) for s in common], [float(mean_b[s]) for s in common]

    gt_results = []
    print("\n=== H1_GT: M1 bucket ordering vs GT (paired by context) ===")
    for bkt_a, bkt_b, label in [("high", "low",  "M1 high > low (GT)"),
                                  ("high", "mid",  "M1 high > mid (GT)"),
                                  ("mid",  "low",  "M1 mid  > low (GT)")]:
        va, vb = collect_paired_gt("m1", bkt_a, bkt_b)
        r = wtest(va, vb, label, "greater")
        if r:
            gt_results.append(r)

    print("\n=== M1 vs M2 overall GT-sim (one-sided, paired by ctx-bucket) ===")
    # Pair at pair level: same (ctx_stem, bucket, target_stem) for M1 vs M2
    m1_gt = gt_df[gt_df["model"] == "m1"].set_index(["ctx_stem", "bucket", "target_stem"])["gt_sim"]
    m2_gt = gt_df[gt_df["model"] == "m2"].set_index(["ctx_stem", "bucket", "target_stem"])["gt_sim"]
    common_idx = sorted(set(m1_gt.index) & set(m2_gt.index))
    r = wtest([float(m1_gt[i]) for i in common_idx],
              [float(m2_gt[i]) for i in common_idx],
              "M1 > M2 (per pair)", "greater")
    if r:
        gt_results.append(r)

    print("\n=== M1 vs M3 overall GT-sim (one-sided, paired by ctx-bucket) ===")
    m3_gt = gt_df[gt_df["model"] == "m3"].set_index(["ctx_stem", "bucket", "target_stem"])["gt_sim"]
    common_idx = sorted(set(m1_gt.index) & set(m3_gt.index))
    r = wtest([float(m1_gt[i]) for i in common_idx],
              [float(m3_gt[i]) for i in common_idx],
              "M1 > M3 (per pair)", "greater")
    if r:
        gt_results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: CROSS-GENERATION DIVERGENCE
    # compound_sim(gen_high_i, gen_low_i) for each context i
    #
    # Tests whether the bucket label drives the model to generate *different*
    # outputs for the same context.  High cross_gen_sim -> model ignores label
    # (gen_high ≈ gen_low).  Low cross_gen_sim -> label successfully diverges
    # generation.  Compare M1 vs M3: if M1 cross_gen_sim < M3, correct label
    # produces more divergence than wrong label.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PART 3: CROSS-GEN DIVERGENCE  compound_sim(gen_high, gen_low)")
    print("="*60)

    cross_rows = []
    for ctx_stem in all_ctx_stems:
        if ctx_stem in bad_stems:
            continue
        for model in ("m1", "m3"):
            high_key = (ctx_stem, model, "high")
            low_key  = (ctx_stem, model, "low")
            mid_key  = (ctx_stem, model, "mid")
            if high_key not in gen_files or low_key not in gen_files:
                continue
            gen_high = midi_to_compound(gen_files[high_key])
            gen_low  = midi_to_compound(gen_files[low_key])
            if len(gen_high) < args.min_notes or len(gen_low) < args.min_notes:
                continue
            sim_hl = compound_sim(gen_high, gen_low)
            row = {"ctx_stem": ctx_stem, "model": model,
                   "cross_sim_high_low": sim_hl}
            # also high vs mid, mid vs low if mid exists
            if mid_key in gen_files:
                gen_mid = midi_to_compound(gen_files[mid_key])
                if len(gen_mid) >= args.min_notes:
                    row["cross_sim_high_mid"] = compound_sim(gen_high, gen_mid)
                    row["cross_sim_mid_low"]  = compound_sim(gen_mid,  gen_low)
            cross_rows.append(row)

    cross_df = pd.DataFrame(cross_rows)

    print("\n=== Mean cross_gen_sim(high, low) by model ===")
    print("  Lower = bucket label drives generation further apart")
    for model in ("m1", "m3"):
        sub = cross_df[cross_df["model"] == model]["cross_sim_high_low"]
        if len(sub) == 0:
            continue
        print(f"  {model.upper()}: mean={sub.mean():.4f}  median={sub.median():.4f}"
              f"  std={sub.std():.4f}  n={len(sub)}")

    print("\n=== Distribution of cross_gen_sim(high, low) — M1 ===")
    m1_cross = cross_df[cross_df["model"] == "m1"]["cross_sim_high_low"].dropna()
    if len(m1_cross) > 0:
        pcts = np.percentile(m1_cross, [10, 25, 50, 75, 90])
        print(f"  p10={pcts[0]:.3f}  p25={pcts[1]:.3f}  p50={pcts[2]:.3f}"
              f"  p75={pcts[3]:.3f}  p90={pcts[4]:.3f}")
        frac_low = float(np.mean(m1_cross < 0.7))
        print(f"  Fraction of contexts with cross_sim < 0.7 (meaningful divergence): {frac_low:.2f}")

    cross_results = []
    print("\n=== M1 vs M3 cross_gen_sim(high, low): does correct label diverge more? ===")
    m1_map = cross_df[cross_df["model"] == "m1"].set_index("ctx_stem")["cross_sim_high_low"]
    m3_map = cross_df[cross_df["model"] == "m3"].set_index("ctx_stem")["cross_sim_high_low"]
    common = sorted(set(m1_map.index) & set(m3_map.index))
    if common:
        # M1 < M3 means M1 (correct label) diverges generation more than M3 (wrong label)
        r = wtest([float(m3_map[s]) for s in common],
                  [float(m1_map[s]) for s in common],
                  "M3 cross_sim > M1 cross_sim (M1 diverges more)", "greater")
        if r:
            cross_results.append(r)

    cross_path = out_path.with_name(out_path.stem + "_cross_gen.csv")
    cross_df.to_csv(cross_path, index=False)

    # ── Save test results ─────────────────────────────────────────────────────
    all_results = (
        [{"section": "ctx_sim",   **r} for r in ctx_results] +
        [{"section": "gt",        **r} for r in gt_results]  +
        [{"section": "cross_gen", **r} for r in cross_results]
    )
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(str(out_path), index=False)
    print(f"\nAll test results saved to {out_path}")
    print(f"Raw ctx_sim scores:   {ctx_path}")
    print(f"Raw GT scores:        {gt_path}")
    print(f"Cross-gen sim scores: {cross_path}")


if __name__ == "__main__":
    main()
