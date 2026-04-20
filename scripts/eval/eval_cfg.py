r"""
scripts/eval/eval_cfg.py

Evaluate CFG outputs from infer_cfg.py.

Three tests, same design as eval_generation.py:
  1. Ctx-sim:   compound_sim(ctx, gen_high) > compound_sim(ctx, gen_low)
  2. GT-sim:    compound_sim(GT,  gen_high) > compound_sim(GT,  gen_low)
                GT target identified from pairs_csv (sim_bucket column).
  3. Cross-gen: compound_sim(gen_high, gen_low) per context --
                lower = bucket label drives outputs further apart.

For each CFG alpha, all three tests are run and compared to the M1 baseline
outputs in the triplet inference directory.

Expected file names in cfg_dir:
  {ctx_stem}_cfg{alpha_tag}_high_generated.mid   e.g. _cfg2p0_high_
  {ctx_stem}_cfg{alpha_tag}_low_generated.mid

Usage:
    python scripts/eval/eval_cfg.py \
        --cfg_dir      outputs/generated/cfg \
        --baseline_dir outputs/generated/triplet \
        --sections_dir <SECTIONS_DIR> \
        --pairs_csv    <SECTIONS_DIR>/pairs.csv \
        --split        test \
        --truncate_to_ctx
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import skew as sp_skew

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval.eval_generation import midi_to_compound, compound_sim, wtest


def _normalize_onsets(tokens: np.ndarray) -> np.ndarray:
    if len(tokens) == 0:
        return tokens
    t = tokens.copy()
    t[:, 0] -= t[0, 0]
    return t


MIN_NOTES = 10


def load_gen(path: Path, cutoff_ticks: int = 0) -> np.ndarray | None:
    """Load generated MIDI. If cutoff_ticks > 0, truncate to that span."""
    if not path.exists():
        return None
    arr = midi_to_compound(path)
    if cutoff_ticks > 0 and len(arr) > 0:
        arr = _normalize_onsets(arr)
        arr = arr[arr[:, 0] <= cutoff_ticks]
    if len(arr) < MIN_NOTES:
        return None
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir",       required=True)
    parser.add_argument("--baseline_dir",  required=True)
    parser.add_argument("--sections_dir",  required=True)
    parser.add_argument("--pairs_csv",     required=True)
    parser.add_argument("--split",         default="test")
    parser.add_argument("--out_prefix",    default=None,
                        help="Prefix for output CSVs (default: inside cfg_dir)")
    parser.add_argument("--truncate_to_ctx", action="store_true",
                        help="Truncate generated output to context span before scoring.")
    args = parser.parse_args()

    cfg_dir      = Path(args.cfg_dir)
    baseline_dir = Path(args.baseline_dir)
    sections_dir = Path(args.sections_dir)
    out_prefix   = args.out_prefix or str(cfg_dir / "eval_cfg")

    # Detect which alpha tags exist in the output directory
    cfg_tags = sorted(set(
        f.name.split("_cfg")[1].split("_")[0]
        for f in cfg_dir.glob("*_cfg*_high_generated.mid")
    ))
    print(f"CFG alpha tags found: {cfg_tags}")

    # Methods: baseline M1 + all CFG alphas
    methods = (
        [("M1_baseline", "baseline")]
        + [(f"CFG_a{t}", f"cfg{t}") for t in cfg_tags]
    )

    # Build per-context info from pairs_csv
    pairs_df = pd.read_csv(args.pairs_csv)
    test_df  = pairs_df[pairs_df["split"] == args.split].copy()
    test_df["context_stem"] = test_df["context_file"].apply(lambda f: Path(f).stem)

    # GT targets per (context_stem, sim_bucket)
    gt_map: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for _, row in test_df.iterrows():
        tgt_path = sections_dir / row["target_file"]
        if tgt_path.exists():
            tgt = np.load(str(tgt_path)).astype(np.int64)
            tgt = _normalize_onsets(tgt)
            if len(tgt) >= MIN_NOTES:
                gt_map[(row["context_stem"], row["sim_bucket"])].append(tgt)

    ctx_stats = test_df.groupby("context_stem").agg(
        context_file=("context_file", "first"),
    ).reset_index()
    print(f"Unique contexts: {len(ctx_stats)}")

    ctx_sims_by_method:   dict[str, dict[str, list[float]]] = {m: {"high": [], "low": []} for m, _ in methods}
    gt_sims_by_method:    dict[str, dict[str, list[float]]] = {m: {"high": [], "low": []} for m, _ in methods}
    cross_sims_by_method: dict[str, list[float]] = {m: [] for m, _ in methods}

    n_skipped = 0
    for _, row in ctx_stats.iterrows():
        ctx_stem = row["context_stem"]
        ctx_path = sections_dir / row["context_file"]
        if not ctx_path.exists():
            n_skipped += 1
            continue

        ctx_npy  = _normalize_onsets(np.load(str(ctx_path)).astype(np.int64))
        ctx_span = int(ctx_npy[-1, 0]) if args.truncate_to_ctx else 0

        for method_label, file_tag in methods:
            if method_label == "M1_baseline":
                gen_dir = baseline_dir
                hi_name = f"{ctx_stem}_m1_high_generated.mid"
                lo_name = f"{ctx_stem}_m1_low_generated.mid"
            else:
                gen_dir = cfg_dir
                hi_name = f"{ctx_stem}_{file_tag}_high_generated.mid"
                lo_name = f"{ctx_stem}_{file_tag}_low_generated.mid"

            gen_high = load_gen(gen_dir / hi_name, cutoff_ticks=ctx_span)
            gen_low  = load_gen(gen_dir / lo_name, cutoff_ticks=ctx_span)

            if gen_high is None or gen_low is None:
                continue

            ctx_sims_by_method[method_label]["high"].append(compound_sim(ctx_npy, gen_high))
            ctx_sims_by_method[method_label]["low"].append(compound_sim(ctx_npy, gen_low))

            for bkt, gen_arr, bkt_key in [
                ("high", gen_high, "sim:high"),
                ("low",  gen_low,  "sim:low"),
            ]:
                gts = gt_map.get((ctx_stem, bkt_key), [])
                if gts:
                    gt_s = float(np.mean([compound_sim(gt, gen_arr) for gt in gts]))
                    gt_sims_by_method[method_label][bkt].append(gt_s)

            cross_sims_by_method[method_label].append(compound_sim(gen_high, gen_low))

    print(f"Contexts skipped (missing npy): {n_skipped}")

    sep = "=" * 65
    print(f"\n{sep}")
    print("PART 1: CTX-SIM  compound_sim(context, generated)")
    print(sep)

    print(f"\n{'Method':<18}  {'high':>8}  {'low':>8}  {'n':>5}")
    print("-" * 45)
    for method_label, _ in methods:
        hi = ctx_sims_by_method[method_label]["high"]
        lo = ctx_sims_by_method[method_label]["low"]
        n  = min(len(hi), len(lo))
        if n == 0:
            continue
        print(f"  {method_label:<16}  {np.mean(hi):8.4f}  {np.mean(lo):8.4f}  {n:5d}")

    print(f"\n=== H1_ctx: gen_high ctx-sim > gen_low ctx-sim ===")
    test_results = []
    for method_label, _ in methods:
        hi = ctx_sims_by_method[method_label]["high"]
        lo = ctx_sims_by_method[method_label]["low"]
        n  = min(len(hi), len(lo))
        if n < 5:
            continue
        r = wtest(hi[:n], lo[:n], f"{method_label} ctx high>low")
        if r:
            r["method"] = method_label
            r["test"]   = "H1_ctx"
            test_results.append(r)

    print(f"\n{sep}")
    print("PART 2: GT-SIM  compound_sim(GT_target, generated)")
    print(sep)

    print(f"\n{'Method':<18}  {'high':>8}  {'low':>8}  {'n_hi':>6}  {'n_lo':>6}")
    print("-" * 55)
    for method_label, _ in methods:
        hi = gt_sims_by_method[method_label]["high"]
        lo = gt_sims_by_method[method_label]["low"]
        print(f"  {method_label:<16}  "
              f"{np.mean(hi) if hi else 0.0:8.4f}  "
              f"{np.mean(lo) if lo else 0.0:8.4f}  "
              f"{len(hi):6d}  {len(lo):6d}")

    print(f"\n=== H1_GT: gen_high GT-sim > gen_low GT-sim ===")
    for method_label, _ in methods:
        hi = gt_sims_by_method[method_label]["high"]
        lo = gt_sims_by_method[method_label]["low"]
        n  = min(len(hi), len(lo))
        if n < 5:
            continue
        r = wtest(hi[:n], lo[:n], f"{method_label} GT  high>low")
        if r:
            r["method"] = method_label
            r["test"]   = "H1_GT"
            test_results.append(r)

    print(f"\n{sep}")
    print("PART 3: CROSS-GEN DIVERGENCE  compound_sim(gen_high, gen_low)")
    print("Lower = high and low outputs are more distinct")
    print(sep)

    print(f"\n{'Method':<18}  {'mean':>8}  {'median':>8}  {'<0.7':>6}  {'n':>5}")
    print("-" * 55)
    cross_by_method = {}
    for method_label, _ in methods:
        sims = cross_sims_by_method[method_label]
        if not sims:
            continue
        cross_by_method[method_label] = sims
        frac_low = np.mean([s < 0.7 for s in sims])
        print(f"  {method_label:<16}  {np.mean(sims):8.4f}  {np.median(sims):8.4f}"
              f"  {frac_low:6.2f}  {len(sims):5d}")

    if "M1_baseline" in cross_by_method:
        baseline_cross = cross_by_method["M1_baseline"]
        print(f"\n=== M1_baseline vs CFG: does CFG increase divergence? ===")
        for method_label, _ in methods:
            if method_label == "M1_baseline":
                continue
            sims = cross_by_method.get(method_label, [])
            n = min(len(baseline_cross), len(sims))
            if n < 5:
                continue
            r = wtest(baseline_cross[:n], sims[:n],
                      f"baseline>  {method_label} (method diverges more)")
            if r:
                r["method"] = method_label
                r["test"]   = "cross_gen_vs_baseline"
                test_results.append(r)

    if test_results:
        out_tests = f"{out_prefix}_tests.csv"
        pd.DataFrame(test_results).to_csv(out_tests, index=False)
        print(f"\nTest results saved: {out_tests}")

    print(f"\n{sep}")
    print("SUMMARY TABLE")
    print(sep)
    print(f"{'Method':<18}  {'ctx_hi':>7}  {'ctx_lo':>7}  "
          f"{'gt_hi':>7}  {'gt_lo':>7}  {'cross':>7}  {'<0.7':>5}")
    print("-" * 70)
    for method_label, _ in methods:
        ctx_hi = ctx_sims_by_method[method_label]["high"]
        ctx_lo = ctx_sims_by_method[method_label]["low"]
        gt_hi  = gt_sims_by_method[method_label]["high"]
        gt_lo  = gt_sims_by_method[method_label]["low"]
        cross  = cross_sims_by_method.get(method_label, [])
        if not ctx_hi:
            continue
        frac_low = np.mean([s < 0.7 for s in cross]) if cross else float("nan")
        print(f"  {method_label:<16}  "
              f"{np.mean(ctx_hi):7.4f}  {np.mean(ctx_lo):7.4f}  "
              f"{np.mean(gt_hi) if gt_hi else 0.0:7.4f}  "
              f"{np.mean(gt_lo) if gt_lo else 0.0:7.4f}  "
              f"{np.mean(cross) if cross else 0.0:7.4f}  "
              f"{frac_low:5.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()