r"""
scripts/training/dataset.py

Dataset for StructureLlama section-level pair training.

SectionPairDataset
    Loads pairs.csv (built by build_section_pairs.py).
    Each sample is one (context_section, sim_bucket, target_section) pair.

    Sequence format (M1 / M3, use_proto=True):
        [SOS] [ctx_notes] [SOC] [sim:high/mid/low] [PROTO] [bar1_tgt] [PROTO] [bar2_tgt] ... [EOC]

        - sim:high/mid/low (after SOC): tells model what similarity level to target.
        - PROTO (at each target bar boundary): embedding overridden with proto_proj(mean(embed(ctx)))
          at model forward time. Tells model WHAT the context sounds like.
        - condition_mask: True on [SOS..sim_bucket] prefix AND on all PROTO positions (no loss).
          Loss is computed only on target note and EOC positions.

    Sequence format (M1 / M3, use_proto=False):
        [SOS] [ctx_notes] [SOC] [sim:high/mid/low] [tgt_notes] [EOC]

        - No PROTO tokens. The model must rely on attention to context tokens and the
          bucket label to determine what to generate.
        - condition_mask: True on [SOS..sim_bucket], False on [tgt_notes..EOC].

    Sequence format (M2 -- unconditional baseline):
        [SOS] [ctx_notes] [SOC] [tgt_notes] [EOC]
        condition_mask: True on [SOS..SOC], False on [tgt_notes..EOC]

    Onset ticks in both sections are normalized to start at 0 so that:
      (a) onset embeddings stay within vocab range;
      (b) each section is treated as a self-contained temporal unit.

condition_mode:
    "m1" -- correct sim_bucket token (+ PROTO if use_proto=True)
    "m2" -- no sim_bucket token, no PROTO; unconditional baseline
    "m3" -- randomly wrong sim_bucket token (one of the other 2 buckets, never correct);
            PROTO inserted only if use_proto=True

Similarity bucket tokens:
    sim:high  >= 0.78  (p67 of within-piece pair distribution)
    sim:mid   0.60-0.78
    sim:low   < 0.60   (p33)

Key normalisation (all splits):
    tgt_key_shift column in pairs.csv records the semitone shift that maximises
    cosine(ctx_chroma, roll(tgt_chroma, k)). Applied to target before augmentation.

Transposition augmentation (train split only):
    Random semitone shift in {0, ..., +11} applied to BOTH ctx and tgt with the SAME offset.
    Key normalisation applied first so ctx and tgt share the same tonal centre.

Bucket-balanced sampling (train split only):
    Each bucket (sim:high/mid/low) is sampled with equal probability per batch via
    WeightedRandomSampler. Weights = 1 / bucket_count for each sample.

min_buckets:
    When > 0, filter to contexts that appear in at least min_buckets distinct sim_bucket
    values within the split. Ensures the model sees the same context with different labels.

PairCollator
    Pads variable-length sequences to batch max length.
    Padding uses PAD compound [-3]*6; padded positions masked from loss and attention.

Exports:
    SECTION_TOKEN_MAP -- pass as additional_token_map to model.forward()
    PROTO_TOKEN       -- compound token ID for prototype anchor tokens
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.tokenizer import MusicTokenizer


# ── Similarity bucket / prototype token IDs ──────────────────────────────────

SIM_HIGH_TOKEN = -6   # sim:high (compound_sim >= 0.78)
SIM_MID_TOKEN  = -7   # sim:mid  (compound_sim 0.60-0.78)
SIM_LOW_TOKEN  = -9   # sim:low  (compound_sim < 0.60)
PROTO_TOKEN    = -8   # prototype anchor; embedding overridden at model forward time

# Maps negative compound token IDs -> supplementary_embedding indices
# Index layout: 0=SOS/PAD, 1=(unused), 2=SOC, 3=EOC,
#               4=sim:high, 5=sim:mid, 6=PROTO, 7=sim:low
# Total supplementary embeddings: 8  (model config: num_additional_tokens=8)
SECTION_TOKEN_MAP: dict = {
    -3: 0,               # PAD -> SOS slot
    -4: 2,               # SOC
    -5: 3,               # EOC
    SIM_HIGH_TOKEN: 4,   # sim:high
    SIM_MID_TOKEN:  5,   # sim:mid
    PROTO_TOKEN:    6,   # prototype anchor
    SIM_LOW_TOKEN:  7,   # sim:low
}

_BUCKET_TOKEN = {
    "sim:high": [SIM_HIGH_TOKEN] * 6,
    "sim:mid":  [SIM_MID_TOKEN]  * 6,
    "sim:low":  [SIM_LOW_TOKEN]  * 6,
}
_OTHER_BUCKETS = {
    "sim:high": ["sim:mid", "sim:low"],
    "sim:mid":  ["sim:high", "sim:low"],
    "sim:low":  ["sim:high", "sim:mid"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_onsets(tokens: np.ndarray) -> np.ndarray:
    tokens = tokens.copy()
    if len(tokens) > 0:
        tokens[:, 0] -= int(tokens[0, 0])
    return tokens


def _bars_from_filename(fname: str) -> str:
    """Extract bar range string from section filename, e.g. '5-20' from '00002_s0_A_5-20.npy'."""
    m = re.search(r'_(\d+-\d+)\.npy$', fname)
    return m.group(1) if m else "1-8"


def _parse_num_bars(bar_range_str: str) -> int:
    parts = str(bar_range_str).split("-")
    if len(parts) == 2:
        try:
            return int(parts[1]) - int(parts[0]) + 1
        except ValueError:
            pass
    return 1


def _get_bar_note_indices(tgt_onsets: np.ndarray, num_bars: int) -> list:
    if num_bars <= 1 or len(tgt_onsets) == 0:
        return [0]
    max_onset = int(tgt_onsets[-1])
    if max_onset == 0:
        return [0]
    bar_size = max_onset / max(num_bars - 1, 1)
    seen = set()
    indices = []
    for bar_i in range(num_bars):
        idx = min(int(np.searchsorted(tgt_onsets, bar_i * bar_size)), len(tgt_onsets) - 1)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return sorted(indices)


def _transpose_tokens(tokens: np.ndarray, semitones: int) -> np.ndarray:
    if semitones == 0:
        return tokens
    result = tokens.copy()
    midi = result[:, 2].astype(int) * 12 + result[:, 3].astype(int) + semitones
    midi = np.clip(midi, 0, 127)
    result[:, 2] = (midi // 12).astype(tokens.dtype)
    result[:, 3] = (midi % 12).astype(tokens.dtype)
    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class SectionPairDataset(Dataset):
    """
    Per-pair dataset for similarity-bucket conditioning.
    Returns un-padded samples with keys:
        input_ids      : list[list[int]]  length T
        labels         : list[list[int]]  length T
        condition_mask : list[bool]       length T
        sample_weight  : float
    """

    def __init__(
        self,
        sections_dir: str,
        pairs_csv:    str,
        tokenizer:    MusicTokenizer,
        partition:    str = "train",
        condition_mode: str = "m1",
        max_seq_len:  int = 1024,
        max_samples:  int = 0,
        min_buckets:  int = 0,
        use_proto:    bool = True,
    ):
        if condition_mode not in ("m1", "m2", "m3"):
            raise ValueError(f"condition_mode must be 'm1'/'m2'/'m3', got {condition_mode!r}")

        self.sections_dir   = pathlib.Path(sections_dir)
        self.tokenizer      = tokenizer
        self.partition      = partition
        self.condition_mode = condition_mode
        self.use_proto      = use_proto
        self.max_section_len = (max_seq_len - 5) // 2

        df = pd.read_csv(pairs_csv)
        df = df[df["split"] == partition].reset_index(drop=True)

        # Filter to contexts appearing in >= min_buckets distinct buckets
        if min_buckets > 0:
            ctx_bucket_sets = df.groupby("context_file")["sim_bucket"].apply(set)
            keep_ctxs = ctx_bucket_sets[ctx_bucket_sets.apply(len) >= min_buckets].index
            before = len(df)
            df = df[df["context_file"].isin(keep_ctxs)].reset_index(drop=True)
            print(f"  min_buckets={min_buckets}: {before} -> {len(df)} pairs "
                  f"({len(keep_ctxs)} contexts)")

        self.pairs: list[dict] = df.to_dict("records")

        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]

        # M3: assign a randomly wrong bucket to each pair (0% correct labels)
        if condition_mode == "m3":
            rng = np.random.default_rng(seed=0)
            self.pairs = [
                dict(r, sim_bucket=rng.choice(_OTHER_BUCKETS[r["sim_bucket"]]))
                for r in self.pairs
            ]

        # Bucket weights for WeightedRandomSampler (train only)
        # Weight = 1 / count_of_that_bucket → equal expected draws per bucket
        from collections import Counter
        counts = Counter(r["sim_bucket"] for r in self.pairs)
        self.sample_weights: list[float] = [
            1.0 / counts[r["sim_bucket"]] for r in self.pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        row = self.pairs[idx]

        ctx = _normalize_onsets(
            np.load(str(self.sections_dir / row["context_file"])).astype(np.int64)
        )[: self.max_section_len]
        tgt = _normalize_onsets(
            np.load(str(self.sections_dir / row["target_file"])).astype(np.int64)
        )[: self.max_section_len]

        # Key normalisation: shift target to match context tonal centre
        key_shift = int(row.get("tgt_key_shift", 0))
        if key_shift != 0:
            tgt = _transpose_tokens(tgt, key_shift)

        # Transposition augmentation (train only; same shift for ctx and tgt)
        if self.partition == "train":
            semitones = int(np.random.randint(0, 12))
            if semitones != 0:
                ctx = _transpose_tokens(ctx, semitones)
                tgt = _transpose_tokens(tgt, semitones)

        ctx_list = ctx.tolist()
        tgt_list = tgt.tolist()
        tok   = self.tokenizer
        sos   = tok.sos_token_compound
        soc   = tok.soc_token_compound
        eoc   = tok.eoc_token_compound
        proto = [PROTO_TOKEN] * 6
        dummy = tok.sos_label

        all_tgt_labels = self._make_target_labels(tgt)
        note_labels    = all_tgt_labels[:-1]
        eoc_label      = all_tgt_labels[-1]

        if self.condition_mode == "m2":
            prefix_len    = 1 + len(ctx_list) + 1
            target_tokens = tgt_list + [eoc]
            target_labels = note_labels + [eoc_label]
            target_cond   = [False] * len(target_tokens)
            input_ids     = [sos] + ctx_list + [soc] + target_tokens

        else:
            # M1 / M3: sim_bucket token (+ PROTO at bar boundaries if use_proto)
            bucket_tok = _BUCKET_TOKEN[row["sim_bucket"]]

            if self.use_proto:
                # PROTO tokens at bar boundaries
                target_bars = _bars_from_filename(row["target_file"])
                num_bars    = _parse_num_bars(target_bars)
                bar_indices = set(_get_bar_note_indices(tgt[:, 0], num_bars))

                target_tokens = []
                target_labels = []
                target_cond   = []
                for note_idx, (note, nlabel) in enumerate(zip(tgt_list, note_labels)):
                    if note_idx in bar_indices:
                        target_tokens.append(proto)
                        target_labels.append(dummy)
                        target_cond.append(True)
                    target_tokens.append(note)
                    target_labels.append(nlabel)
                    target_cond.append(False)
                target_tokens.append(eoc)
                target_labels.append(eoc_label)
                target_cond.append(False)
            else:
                # No PROTO tokens: just bucket + target notes
                target_tokens = tgt_list + [eoc]
                target_labels = note_labels + [eoc_label]
                target_cond   = [False] * len(target_tokens)

            prefix_len = 1 + len(ctx_list) + 1 + 1   # SOS + ctx + SOC + bucket
            input_ids  = [sos] + ctx_list + [soc, bucket_tok] + target_tokens

        labels         = [dummy] * prefix_len + target_labels
        condition_mask = [True]  * prefix_len + target_cond

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "condition_mask": condition_mask,
            "sample_weight":  1.0,
        }

    def _make_target_labels(self, tgt: np.ndarray) -> List[List[int]]:
        tok     = self.tokenizer
        wrapped = [tok.sos_token_compound] + tgt.tolist()
        all_labels = tok.encode_series_labels(wrapped, if_added_sos=True, if_added_eos=False)
        return all_labels[1:] + [tok.eos_label]


# ── Collator ──────────────────────────────────────────────────────────────────

class PairCollator:
    def __init__(self, tokenizer: MusicTokenizer):
        self.pad_compound = tokenizer.pad_token_compound
        self.pad_label    = tokenizer.sos_label

    def __call__(self, batch: List[dict]) -> dict:
        max_len = max(len(s["input_ids"]) for s in batch)
        ids_out, lbl_out, cond_out, attn_out, weight_out = [], [], [], [], []

        for s in batch:
            T       = len(s["input_ids"])
            pad_len = max_len - T
            ids_out.append(  s["input_ids"]      + [self.pad_compound] * pad_len)
            lbl_out.append(  s["labels"]         + [self.pad_label]    * pad_len)
            cond_out.append( s["condition_mask"]  + [True]              * pad_len)
            attn_out.append( [1] * T              + [0]                 * pad_len)
            weight_out.append(s.get("sample_weight", 1.0))

        return {
            "input_ids":      torch.tensor(ids_out,    dtype=torch.long),
            "labels":         torch.tensor(lbl_out,    dtype=torch.long),
            "condition_mask": torch.tensor(cond_out,   dtype=torch.bool),
            "attention_mask": torch.tensor(attn_out,   dtype=torch.long),
            "sample_weights": torch.tensor(weight_out, dtype=torch.float),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_pair_dataloader(
    sections_dir:   str,
    pairs_csv:      str,
    tokenizer:      MusicTokenizer,
    partition:      str = "train",
    condition_mode: str = "m1",
    max_seq_len:    int = 1024,
    batch_size:     int = 4,
    num_workers:    int = 0,
    max_samples:    int = 0,
    min_buckets:    int = 0,
    use_proto:      bool = True,
) -> DataLoader:
    ds       = SectionPairDataset(
        sections_dir=sections_dir,
        pairs_csv=pairs_csv,
        tokenizer=tokenizer,
        partition=partition,
        condition_mode=condition_mode,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        min_buckets=min_buckets,
        use_proto=use_proto,
    )
    collator = PairCollator(tokenizer)

    # Bucket-balanced sampling on train; plain shuffle on val/test
    if partition == "train":
        weights = torch.tensor(ds.sample_weights, dtype=torch.float)
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          collate_fn=collator, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collator, num_workers=num_workers, pin_memory=True)
