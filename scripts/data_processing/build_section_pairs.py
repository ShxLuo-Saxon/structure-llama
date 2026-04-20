r"""
scripts/data_processing/build_section_pairs.py

Build the section-pair dataset for similarity-bucket conditioning.

  - No A/B structural labels. Every within-piece directed section pair is valid.
  - Conditioning signal = compound_sim similarity bucket (sim:high / sim:mid / sim:low).
  - Thresholds set from data distribution (p33=0.60, p67=0.78 across all within-piece pairs).
  - Excludes double-piano and mixed-instrument pieces (exclude_r10 column in metadata.xlsx).
  - Stratified split: by section count (2/3/4/5/6+), 70/15/15, at version-group level.
    Version groups (same base_name + same total_bars, different num_voices) always go
    to the same split to prevent data leakage across arrangement versions.
  - Works from existing per-section .npy files in sections/; does not re-extract sections.

Bucket thresholds (from within_piece_sim_dist.py, n=3652 pairs):
  sim:high  >= 0.78  (p67)  ~33%
  sim:mid   0.60-0.78        ~33%
  sim:low   < 0.60   (p33)  ~33%

Output: sections/pairs.csv
  pair_id, piece_id, split, context_file, target_file,
  context_ntokens, target_ntokens, compound_sim, sim_bucket, tgt_key_shift

Usage (from project root):
    python scripts/data_processing/build_section_pairs.py \
        --sections_dir <SECTIONS_DIR> \
        --metadata     outputs/metadata.xlsx

Optional:
    --thresh_high  0.78   (default p67)
    --thresh_mid   0.60   (default p33)
    --train_ratio  0.70
    --val_ratio    0.15
    --seed         42
    --dry_run             (print stats, do not write pairs.csv)
"""
from __future__ import annotations

import argparse
import csv
import re
import random
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval.compound_metric import compound_sim, chroma_vector  # noqa: E402


# ── Similarity bucket thresholds ─────────────────────────────────────────────

THRESH_HIGH = 0.78   # >= sim:high
THRESH_MID  = 0.60   # >= sim:mid, < sim:high; below = sim:low

SIM_HIGH = "sim:high"
SIM_MID  = "sim:mid"
SIM_LOW  = "sim:low"

MIN_TOKENS = 4   # minimum notes for a section to participate in any pair


def compute_key_shift(ctx: np.ndarray, tgt: np.ndarray,
                      min_gain: float = 0.1) -> int:
    """Semitone shift [0,11] that maximises chroma cosine; 0 if gain < min_gain."""
    ch_c = chroma_vector(ctx)
    ch_t = chroma_vector(tgt)
    nc, nt = np.linalg.norm(ch_c), np.linalg.norm(ch_t)
    if nc == 0 or nt == 0:
        return 0
    sim0 = float(np.dot(ch_c, ch_t) / (nc * nt))
    best_k, best_sim = 0, sim0
    for k in range(1, 12):
        s = float(np.dot(ch_c, np.roll(ch_t, k)) / (nc * nt))
        if s > best_sim:
            best_sim, best_k = s, k
    return best_k if best_k != 0 and (best_sim - sim0) >= min_gain else 0


def assign_bucket(score: float,
                  thresh_high: float = THRESH_HIGH,
                  thresh_mid:  float = THRESH_MID) -> str:
    if np.isnan(score):
        return SIM_MID   # fallback
    if score >= thresh_high:
        return SIM_HIGH
    if score >= thresh_mid:
        return SIM_MID
    return SIM_LOW


# ── Version-group helpers (ported from split_dataset.py) ─────────────────────

def _get_base_name(filename: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", str(filename)).strip()


def assign_version_groups(df: pd.DataFrame) -> dict[int, int]:
    """
    Returns {piece_id_int: group_id}.
    Pieces sharing base_name + total_bars but differing num_voices share a group.
    All others get a unique group_id.
    """
    df = df.copy()
    df["base_name"] = df["original_filename"].apply(_get_base_name)

    pair_members: dict[tuple, list] = {}
    for (base, bars), grp in df.groupby(["base_name", "total_bars"]):
        if len(grp) > 1 and grp["num_voices"].nunique() > 1:
            pair_members[(base, bars)] = grp["id"].tolist()

    id_to_group: dict[int, int] = {}
    assigned: set[int] = set()
    gid = 0

    for ids in pair_members.values():
        for fid in ids:
            id_to_group[fid] = gid
            assigned.add(fid)
        gid += 1

    for fid in df["id"]:
        if fid not in assigned:
            id_to_group[fid] = gid
            gid += 1

    return id_to_group


# ── Stratified split by section count ────────────────────────────────────────

SEC_COUNT_STRATA = [2, 3, 4, 5, 6]   # 6 = "6+" bucket

def _sec_stratum(n: int) -> int:
    return min(n, 6)


def stratified_split_by_sec_count(
    group_to_pids:    dict[int, list[str]],
    pid_to_sec_count: dict[str, int],
    train_ratio:      float = 0.70,
    val_ratio:        float = 0.15,
    seed:             int   = 42,
) -> dict[str, str]:
    """
    Returns {piece_id: 'train'/'val'/'test'}.
    Split is at group level; stratified by max section count within each group.
    """
    rng = random.Random(seed)

    # Build per-stratum list of groups
    strata: dict[int, list[int]] = defaultdict(list)
    for gid, pids in group_to_pids.items():
        max_sec = max(pid_to_sec_count.get(p, 0) for p in pids)
        strata[_sec_stratum(max_sec)].append(gid)

    pid_to_split: dict[str, str] = {}

    for stratum, gids in sorted(strata.items()):
        rng.shuffle(gids)
        n = len(gids)
        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1.0 - train_ratio - val_ratio)))
        # guard against over-allocation in tiny strata
        n_test = min(n_test, max(0, n - 1))
        n_val  = min(n_val,  max(0, n - n_test - 1))

        test_gids  = gids[:n_test]
        val_gids   = gids[n_test: n_test + n_val]
        train_gids = gids[n_test + n_val:]

        label = f"{stratum}+" if stratum == 6 else str(stratum)
        print(f"  sec_count={label:3s}  total={n:3d}  "
              f"train={len(train_gids):3d}  val={len(val_gids):2d}  test={len(test_gids):2d}")

        for gid in train_gids:
            for p in group_to_pids[gid]: pid_to_split[p] = "train"
        for gid in val_gids:
            for p in group_to_pids[gid]: pid_to_split[p] = "val"
        for gid in test_gids:
            for p in group_to_pids[gid]: pid_to_split[p] = "test"

    return pid_to_split


# ── Filename helpers ──────────────────────────────────────────────────────────

def piece_id_from_filename(fname: str) -> str:
    m = re.match(r"^(.+?)_s\d+_", fname)
    return m.group(1) if m else fname.replace(".npy", "")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections_dir", required=True,
                        help="Directory containing per-section .npy files")
    parser.add_argument("--metadata",     required=True,
                        help="Path to metadata.xlsx (must have exclude_r10 column)")
    parser.add_argument("--thresh_high",  type=float, default=THRESH_HIGH)
    parser.add_argument("--thresh_mid",   type=float, default=THRESH_MID)
    parser.add_argument("--train_ratio",  type=float, default=0.70)
    parser.add_argument("--val_ratio",    type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--dry_run",      action="store_true")
    args = parser.parse_args()

    sec_dir   = Path(args.sections_dir)
    meta_path = Path(args.metadata)

    # ── Load metadata ─────────────────────────────────────────────────────────
    df_meta = pd.read_excel(meta_path)
    df_meta["id"] = df_meta["id"].astype(int)

    if "exclude_r10" not in df_meta.columns:
        raise ValueError("metadata.xlsx missing 'exclude_r10' column -- "
                         "use outputs/metadata.xlsx from this repo.")

    excluded_ids: set[str] = set(
        str(int(row["id"])).zfill(5)
        for _, row in df_meta.iterrows()
        if row["exclude_r10"]
    )
    print(f"Pieces excluded (exclude_r10=True): {len(excluded_ids)}")

    # ── Build version groups ──────────────────────────────────────────────────
    id_to_group = assign_version_groups(df_meta)
    # group_id -> list of zero-padded piece_ids
    group_to_pids_meta: dict[int, list[str]] = defaultdict(list)
    for fid, gid in id_to_group.items():
        group_to_pids_meta[gid].append(str(int(fid)).zfill(5))

    # ── Collect pieces from sections dir ─────────────────────────────────────
    all_npy = sorted(sec_dir.glob("*.npy"))
    pieces: dict[str, list[Path]] = defaultdict(list)
    for f in all_npy:
        pid = piece_id_from_filename(f.name)
        pieces[pid].append(f)

    # Filter to pieces with >= 2 sections and not excluded
    usable_pids = {
        pid for pid, files in pieces.items()
        if len(files) >= 2 and pid not in excluded_ids
    }
    print(f"Total pieces in sections dir: {len(pieces)}")
    print(f"Excluded: {len([p for p in pieces if p in excluded_ids])}")
    print(f"Usable (>=2 sections, not excluded): {len(usable_pids)}")

    # Section count per usable piece
    pid_to_sec_count = {pid: len(pieces[pid]) for pid in usable_pids}

    # Filter version groups to usable pieces only
    group_to_pids: dict[int, list[str]] = {}
    for gid, pids in group_to_pids_meta.items():
        usable = [p for p in pids if p in usable_pids]
        if usable:
            group_to_pids[gid] = usable

    # ── Stratified split ──────────────────────────────────────────────────────
    print(f"\nStratified split by section count (seed={args.seed}):")
    pid_to_split = stratified_split_by_sec_count(
        group_to_pids, pid_to_sec_count,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Verify coverage
    unassigned = usable_pids - set(pid_to_split.keys())
    if unassigned:
        print(f"WARNING: {len(unassigned)} pieces not assigned to any split -- "
              f"check metadata id coverage. Assigning to train.")
        for p in unassigned:
            pid_to_split[p] = "train"

    split_counts = defaultdict(int)
    for s in pid_to_split.values():
        split_counts[s] += 1
    print(f"\n  train={split_counts['train']}  val={split_counts['val']}  "
          f"test={split_counts['test']}  total={sum(split_counts.values())}")

    # ── Build pairs ───────────────────────────────────────────────────────────
    print("\nComputing within-piece pairs...")
    pair_rows = []
    pair_id   = 0
    skipped_nan = 0
    bucket_counts: dict[str, int] = {SIM_HIGH: 0, SIM_MID: 0, SIM_LOW: 0}

    for pid in sorted(usable_pids):
        files = sorted(pieces[pid])
        split = pid_to_split.get(pid, "train")

        # Load all section arrays for this piece
        arrays: dict[str, np.ndarray] = {}
        for f in files:
            arr = np.load(str(f)).astype(float)
            if arr.ndim == 2 and arr.shape[0] >= MIN_TOKENS and arr.shape[1] >= 4:
                arrays[f.name] = arr

        fnames = list(arrays.keys())
        if len(fnames) < 2:
            continue

        # All directed pairs (a -> b and b -> a are separate)
        for fa, fb in permutations(fnames, 2):
            score = compound_sim(arrays[fa], arrays[fb])
            if np.isnan(score):
                skipped_nan += 1
                continue
            bucket = assign_bucket(score, args.thresh_high, args.thresh_mid)
            key_shift = compute_key_shift(arrays[fa], arrays[fb])
            pair_rows.append({
                "pair_id":          pair_id,
                "piece_id":         pid,
                "split":            split,
                "context_file":     fa,
                "target_file":      fb,
                "context_ntokens":  len(arrays[fa]),
                "target_ntokens":   len(arrays[fb]),
                "compound_sim":     round(float(score), 6),
                "sim_bucket":       bucket,
                "tgt_key_shift":    key_shift,
            })
            pair_id += 1
            bucket_counts[bucket] += 1

    print(f"Total pairs: {pair_id}  (skipped NaN: {skipped_nan})")

    # ── Summary ───────────────────────────────────────────────────────────────
    def _count(split=None, bucket=None):
        return sum(
            1 for r in pair_rows
            if (split  is None or r["split"]      == split)
            and (bucket is None or r["sim_bucket"] == bucket)
        )

    print(f"\n{'='*60}")
    print("RUN 10 SECTION PAIR DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Thresholds: sim:high >= {args.thresh_high}  "
          f"sim:mid >= {args.thresh_mid}  sim:low < {args.thresh_mid}")
    print(f"\n  {'Split':<8} {'sim:high':>9} {'sim:mid':>9} {'sim:low':>9} {'total':>7}")
    print(f"  {'-'*45}")
    for sp in ["train", "val", "test", "ALL"]:
        sp_arg = None if sp == "ALL" else sp
        h = _count(sp_arg, SIM_HIGH)
        m = _count(sp_arg, SIM_MID)
        l = _count(sp_arg, SIM_LOW)
        print(f"  {sp:<8} {h:>9} {m:>9} {l:>9} {h+m+l:>7}")

    if pair_rows:
        ctx_lens = [r["context_ntokens"] for r in pair_rows]
        tgt_lens = [r["target_ntokens"]  for r in pair_rows]
        print(f"\n  Context tokens: min={min(ctx_lens)}  max={max(ctx_lens)}  "
              f"mean={sum(ctx_lens)/len(ctx_lens):.0f}")
        print(f"  Target  tokens: min={min(tgt_lens)}  max={max(tgt_lens)}  "
              f"mean={sum(tgt_lens)/len(tgt_lens):.0f}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_csv = sec_dir / "pairs.csv"
    fieldnames = ["pair_id", "piece_id", "split", "context_file", "target_file",
                  "context_ntokens", "target_ntokens", "compound_sim",
                  "sim_bucket", "tgt_key_shift"]

    if args.dry_run:
        print(f"\n[dry_run] Would write {len(pair_rows)} pairs to: {out_csv}")
    else:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_rows)
        print(f"\nWrote {len(pair_rows)} pairs to: {out_csv}")


if __name__ == "__main__":
    main()
