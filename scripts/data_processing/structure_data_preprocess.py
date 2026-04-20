r"""
scripts/data_processing/structure_data_preprocess.py

Tokenise MIDI files and extract per-section compound-token arrays.

For each annotated piece:
  1. Load the MIDI (produced by mxl_to_mid.py) and convert to compound tokens
     at 100 ticks/s, normalised to 120 BPM.
  2. Read the MXL source file to get exact bar onset positions (handles
     anacrusis and time-signature changes transparently via music21).
  3. Use the annotation JSON to locate section bar ranges, then slice the
     compound-token array into per-section arrays.
  4. Save each section as <piece_id>_s<idx>_<label>_<bar_start>-<bar_end>.npy

Output files are written to --out_dir and are the direct input to
build_section_pairs.py.

Input assumptions
-----------------
  MIDI files:       produced by mxl_to_mid.py -- repeats unrolled, flat 120 BPM.
                    Do NOT use MIDI downloaded directly from NinSheetMusic.
  MXL files:        original source files (or manually corrected versions in
                    <mxl_dir>/fixes/).  Used only for bar offset computation.
  Annotation JSONs: one per piece, naming pieces by 5-digit zero-padded ID.

Usage:
    python scripts/data_processing/structure_data_preprocess.py \
        --mid_dir        <DATASET>/processed/mid \
        --mxl_dir        <DATASET>/processed/mxl \
        --annotation_dir <DATASET>/processed/annotation/v1.2_singleApi \
        --out_dir        <DATASET>/processed/npy/sections

Optional:
    --fixes_dir  <DATASET>/processed/mxl/fixes   (default: <mxl_dir>/fixes)
    --min_tokens 4                                (skip sections shorter than this)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from music21 import converter
from tqdm import tqdm

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.tokenizer import MusicTokenizer

MAX_DURATION = 1023   # tokenizer vocab max; longer notes are ornament artefacts
TIME_RESOLUTION = 100  # ticks/s — must match MusicTokenizer default
NORMALIZED_BPM  = 120.0


# ── Bar offsets from MXL ──────────────────────────────────────────────────────

def get_bar_offsets(mxl_path: pathlib.Path) -> dict[int, float]:
    """Return {bar_num (1-indexed): quarter-length offset} from MXL via music21.

    Includes a dummy entry at N_bars+1 (end of last bar) so callers can compute
    bar durations.  Returns empty dict on parse failure.
    """
    try:
        score    = converter.parse(str(mxl_path))
        measures = score.parts[0].getElementsByClass("Measure")
        offsets  = {i: float(m.offset) for i, m in enumerate(measures, start=1)}
        last     = measures[-1]
        offsets[len(measures) + 1] = float(last.offset + last.duration.quarterLength)
        return offsets
    except Exception as e:
        print(f"  Warning: could not parse MXL {mxl_path.name}: {e}")
        return {}


def bar_offsets_to_ticks(bar_offsets: dict[int, float]) -> np.ndarray:
    """Convert bar offsets (quarter-lengths at 120 BPM) to tokenizer tick positions.

    Returns array of length N_bars+1 (includes end-of-last-bar dummy entry),
    indexed 0-based: ticks[i] = onset tick of bar i+1.
    """
    spb = 60.0 / NORMALIZED_BPM   # seconds per beat at 120 BPM
    n   = max(bar_offsets.keys())
    return np.array(
        [round(TIME_RESOLUTION * bar_offsets[b] * spb) for b in range(1, n + 1)],
        dtype=np.int64,
    )


# ── Section extraction ────────────────────────────────────────────────────────

def extract_section(tokens: np.ndarray,
                    bar_ticks: np.ndarray,
                    bar_start: int,
                    bar_end: int) -> np.ndarray:
    """Slice compound tokens for bars [bar_start, bar_end] (1-indexed, inclusive).

    bar_ticks[i] = onset tick of bar i+1 (0-indexed array, so bar 1 = index 0).
    Returns empty array if bar_start exceeds the bar count in bar_ticks (e.g.
    annotation bar range is beyond what the MXL contains — can happen for pieces
    that required manual MXL fixes but were processed from raw MXL instead).
    """
    b0 = bar_start - 1
    b1 = bar_end   - 1
    n  = len(bar_ticks)
    if b0 >= n:
        return tokens[0:0]   # out-of-range section; caller skips via min_tokens
    lo = int(bar_ticks[b0])
    hi = int(bar_ticks[b1 + 1]) if b1 + 1 < n else int(tokens[:, 0].max()) + 1
    mask = (tokens[:, 0] >= lo) & (tokens[:, 0] < hi)
    return tokens[mask]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-section compound-token .npy files from MIDI + MXL + annotations.")
    parser.add_argument("--mid_dir",        required=True,
                        help="Directory of .mid files (from mxl_to_mid.py)")
    parser.add_argument("--mxl_dir",        required=True,
                        help="Directory of .mxl source files")
    parser.add_argument("--annotation_dir", required=True,
                        help="Directory of annotation .json files")
    parser.add_argument("--out_dir",        required=True,
                        help="Output directory for per-section .npy files")
    parser.add_argument("--fixes_dir",      default=None,
                        help="Manually corrected .mxl files (default: <mxl_dir>/fixes)")
    parser.add_argument("--min_tokens",     type=int, default=4,
                        help="Skip sections with fewer than this many notes")
    args = parser.parse_args()

    mid_dir   = pathlib.Path(args.mid_dir)
    mxl_dir   = pathlib.Path(args.mxl_dir)
    ann_dir   = pathlib.Path(args.annotation_dir)
    out_dir   = pathlib.Path(args.out_dir)
    fixes_dir = pathlib.Path(args.fixes_dir) if args.fixes_dir else mxl_dir / "fixes"
    out_dir.mkdir(parents=True, exist_ok=True)

    piece_ids = sorted(int(p.stem) for p in ann_dir.glob("*.json"))
    print(f"Found {len(piece_ids)} annotated pieces.")

    tokenizer = MusicTokenizer()
    n_sections = n_skipped = n_pieces_ok = 0

    for pid in tqdm(piece_ids, desc="Preprocessing"):
        fid      = f"{pid:05d}"
        mid_path = mid_dir / f"{fid}.mid"
        ann_path = ann_dir / f"{fid}.json"

        if not mid_path.exists():
            print(f"  {fid}: MIDI missing — skipped")
            n_skipped += 1
            continue

        # ── 1. MIDI → compound tokens ─────────────────────────────────────────
        tokens = np.array(
            MusicTokenizer.midi_to_compound(str(mid_path),
                                            calibrate_to_default_tempo=True),
            dtype=np.int64,
        )
        # Drop ornament/grace notes (near-infinite duration from MXL conversion)
        n_before = len(tokens)
        tokens   = tokens[tokens[:, 1] <= MAX_DURATION]
        if len(tokens) < n_before:
            print(f"  {fid}: dropped {n_before - len(tokens)} ornament note(s)")

        if len(tokens) == 0:
            print(f"  {fid}: no tokens after filtering — skipped")
            n_skipped += 1
            continue

        # ── 2. Bar offsets from MXL ───────────────────────────────────────────
        mxl_path = fixes_dir / f"{fid}.mxl"
        if not mxl_path.exists():
            mxl_path = mxl_dir / f"{fid}.mxl"

        if not mxl_path.exists():
            print(f"  {fid}: MXL missing — skipped (bar offsets required)")
            n_skipped += 1
            continue

        bar_offsets = get_bar_offsets(mxl_path)
        if not bar_offsets:
            print(f"  {fid}: MXL parse failed — skipped")
            n_skipped += 1
            continue

        bar_ticks = bar_offsets_to_ticks(bar_offsets)

        # ── 3. Load annotation + extract per-section arrays ───────────────────
        try:
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {fid}: annotation parse error ({e}) — skipped")
            n_skipped += 1
            continue

        sections = annotation.get("Section", [])
        # X = intro/transition/connector; s = stinger. Neither participates in
        # structural similarity pairs, so they are excluded here rather than
        # leaving them in for build_section_pairs.py to silently include.
        sections = [e for e in sections if e.get("Section", "s") not in ("X", "s")]
        if len(sections) < 2:
            print(f"  {fid}: fewer than 2 named sections — skipped")
            n_skipped += 1
            continue

        for idx, entry in enumerate(sections):
            bs, be  = entry["BarRange"]
            label   = entry.get("Section", "s")

            if bs - 1 >= len(bar_ticks):
                print(f"  {fid}: section {label} bars {bs}-{be} exceeds MXL bar count "
                      f"({len(bar_ticks)}) — skipped (manual MXL fix may be needed; "
                      f"see fix_log in outputs/metadata.xlsx)")
                continue

            sec_tok = extract_section(tokens, bar_ticks, bs, be)

            if len(sec_tok) < args.min_tokens:
                continue

            # Normalise onsets to start at 0
            sec_tok = sec_tok.copy()
            sec_tok[:, 0] -= sec_tok[0, 0]

            fname = f"{fid}_s{idx}_{label}_{bs}-{be}.npy"
            np.save(str(out_dir / fname), sec_tok)
            n_sections += 1

        n_pieces_ok += 1

    print(f"\nDone.  Pieces: {n_pieces_ok}  Sections saved: {n_sections}  Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
