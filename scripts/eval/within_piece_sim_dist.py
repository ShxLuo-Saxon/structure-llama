r"""
scripts/eval/within_piece_sim_dist.py

Compute compound_sim for all within-piece cross-section pairs across all .npy
files in the sections dir.  Groups sections by piece_id (everything before the
first '_s' in the filename).  Plots a histogram and suggests bucket thresholds
for the similarity-bucket conditioning design.

compound_sim is imported from compound_metric.py (authoritative implementation).

CPU-only -- no torch, no GPU required.

Usage (from project root):
    python scripts/eval/within_piece_sim_dist.py \
        --sections_dir <SECTIONS_DIR>

Optional:
    --pairs_csv  path/to/sections/pairs.csv   (overlay sim_bucket distribution on histogram)
    --out_csv    path/to/output.csv           (save per-pair scores)
    --no_plot                                  (skip matplotlib, print thresholds only)
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eval.compound_metric import compound_sim


# ── piece grouping ────────────────────────────────────────────────────────────

def piece_id_from_filename(fname: str) -> str:
    """Extract piece_id: everything before the first '_s<digits>' token."""
    m = re.match(r"^(.+?)_s\d+_", fname)
    return m.group(1) if m else fname.replace(".npy", "")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections_dir", required=True,
                        help="Directory containing per-section .npy files")
    parser.add_argument("--pairs_csv", default=None,
                        help="Optional pairs.csv to overlay sim_bucket distribution on histogram")
    parser.add_argument("--out_csv", default=None,
                        help="Optional path to save per-pair compound scores")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip matplotlib, print stats only")
    args = parser.parse_args()

    sec_dir = Path(args.sections_dir)
    npy_files = sorted(sec_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {sec_dir}")

    # group by piece
    pieces: dict[str, list[Path]] = defaultdict(list)
    for f in npy_files:
        pid = piece_id_from_filename(f.name)
        pieces[pid].append(f)

    multi = {pid: fs for pid, fs in pieces.items() if len(fs) >= 2}
    print(f"Pieces with >= 2 sections: {len(multi)} / {len(pieces)}")

    # compute all within-piece directed pairs
    rows = []
    skipped = 0
    for pid, files in multi.items():
        arrays = {}
        for f in files:
            arr = np.load(str(f)).astype(float)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 4:
                arrays[f.name] = arr
        fnames = list(arrays.keys())
        for fa, fb in combinations(fnames, 2):
            score = compound_sim(arrays[fa], arrays[fb])
            if np.isnan(score):
                skipped += 1
                continue
            rows.append({"piece_id": pid, "file_a": fa, "file_b": fb, "compound_sim": score})
            score_ba = compound_sim(arrays[fb], arrays[fa])
            if not np.isnan(score_ba):
                rows.append({"piece_id": pid, "file_a": fb, "file_b": fa, "compound_sim": score_ba})

    df = pd.DataFrame(rows)
    print(f"Total within-piece pairs: {len(df)}  (skipped {skipped} NaN)")

    if df.empty:
        print("ERROR: no pairs computed -- check --sections_dir path and that .npy files exist.")
        return

    scores = df["compound_sim"].values
    print(f"\n-- Distribution --")
    print(f"  mean  {scores.mean():.4f}   std  {scores.std():.4f}")
    print(f"  min   {scores.min():.4f}   max  {scores.max():.4f}")
    for p in [10, 25, 33, 50, 67, 75, 90]:
        print(f"  p{p:<3d}  {np.percentile(scores, p):.4f}")

    # load sim_bucket distribution if pairs.csv given
    bucket_scores: dict[str, list[float]] = {}
    if args.pairs_csv:
        pairs = pd.read_csv(args.pairs_csv)
        if "sim_bucket" not in pairs.columns:
            print("WARNING: pairs.csv has no sim_bucket column — skipping overlay")
        else:
            for bkt, grp in pairs.groupby("sim_bucket"):
                bscores = []
                for _, row in grp.iterrows():
                    try:
                        a = np.load(str(sec_dir / row["context_file"])).astype(float)
                        b = np.load(str(sec_dir / row["target_file"])).astype(float)
                        s = compound_sim(a, b)
                        if not np.isnan(s):
                            bscores.append(s)
                    except Exception:
                        pass
                bucket_scores[bkt] = bscores
                print(f"  {bkt} (n={len(bscores)}): mean={np.mean(bscores):.4f}")

    # thresholds: p33 and p67 of the within-piece distribution
    thresh_high = round(float(np.percentile(scores, 67)), 2)
    thresh_mid  = round(float(np.percentile(scores, 33)), 2)

    n_high = (scores >= thresh_high).sum()
    n_mid  = ((scores >= thresh_mid) & (scores < thresh_high)).sum()
    n_low  = (scores < thresh_mid).sum()
    print(f"\n-- Suggested bucket thresholds --")
    print(f"  sim:high  >= {thresh_high:.2f}   n={n_high}  ({100*n_high/len(scores):.1f}%)")
    print(f"  sim:mid   {thresh_mid:.2f} -- {thresh_high:.2f}  n={n_mid}  ({100*n_mid/len(scores):.1f}%)")
    print(f"  sim:low   <  {thresh_mid:.2f}   n={n_low}  ({100*n_low/len(scores):.1f}%)")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved per-pair scores to {args.out_csv}")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt
    bucket_colors = {"sim:high": "green", "sim:mid": "orange", "sim:low": "tomato"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=40, alpha=0.6, color="steelblue", label="all within-piece pairs")
    for bkt, bscores in bucket_scores.items():
        ax.hist(bscores, bins=30, alpha=0.4,
                color=bucket_colors.get(bkt, "grey"),
                label=f"{bkt} (n={len(bscores)})")
    ax.axvline(thresh_high, color="darkgreen", linestyle="--", label=f"sim:high threshold ({thresh_high})")
    ax.axvline(thresh_mid,  color="firebrick", linestyle="--", label=f"sim:mid threshold ({thresh_mid})")
    ax.set_xlabel("compound_sim")
    ax.set_ylabel("count")
    ax.set_title("Within-piece cross-section similarity distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(sec_dir / "within_piece_sim_dist.png"), dpi=150)
    print(f"\nPlot saved to {sec_dir / 'within_piece_sim_dist.png'}")
    plt.show()


if __name__ == "__main__":
    main()
