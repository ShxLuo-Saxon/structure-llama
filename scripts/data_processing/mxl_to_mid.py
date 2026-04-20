r"""
scripts/data_processing/mxl_to_mid.py

Convert MusicXML (.mxl) files to MIDI for StructureLlama preprocessing.

Why MXL, not downloaded MIDI
-----------------------------
NinSheetMusic distributes both .mxl (sheet music) and .mid files.  The
distributed MIDI files are NOT suitable for this project because most pieces
use repeat barlines, D.C. al Coda, Segno, or 1st/2nd ending brackets that
are left folded.  StructureLlama sections are defined on the linear playback
sequence, so repeats must be fully unrolled before tokenisation.

This script uses music21 to:
  1. Strip repeat barlines, D.C./Segno expressions, and 1st/2nd ending brackets
     so the score is written out in full linear playback order.
  2. Strip MetronomeMark events so the MIDI is exported at flat 120 BPM,
     keeping note onset ticks consistent with the bar offsets computed from
     the same MXL by structure_data_preprocess.py.

Fixes directory
---------------
28 of 554 pieces required manual MXL corrections before conversion — complex
D.S./Coda/2nd-ending notation that music21 cannot strip automatically.  The
corrected files cannot be redistributed (copyright).  If you have them, place
them in <mxl_dir>/fixes/ and this script will use them automatically.

Without the corrections, those pieces may produce MIDI with incorrect playback
order (wrong repeat unrolling), leading to wrong section boundaries downstream.
They represent ~5 % of the dataset; overall results are not significantly
affected.  The fix_log column in outputs/metadata.xlsx lists exactly what each
correction involved (e.g. "unroll endings", "remove ds") so corrections can be
applied manually if needed.

Usage:
    python scripts/data_processing/mxl_to_mid.py \
        --mxl_dir        <DATASET>/processed/mxl \
        --annotation_dir <DATASET>/processed/annotation/v1.2_singleApi \
        --out_dir        <DATASET>/processed/mid
"""
from __future__ import annotations

import argparse
import pathlib

from tqdm import tqdm
from music21 import converter, bar, repeat, spanner
from music21 import tempo as m21tempo


def convert(mxl_path: pathlib.Path, mid_path: pathlib.Path) -> bool:
    """Strip repeats and tempo marks, export at flat 120 BPM."""
    try:
        score = converter.parse(str(mxl_path))
        for part in score.parts:
            for el in part.recurse().getElementsByClass(
                    [bar.Repeat, repeat.RepeatExpression]):
                part.remove(el, recurse=True)
            for sp in part.recurse().getElementsByClass(spanner.RepeatBracket):
                part.remove(sp, recurse=True)
            for el in part.recurse().getElementsByClass(m21tempo.MetronomeMark):
                part.remove(el, recurse=True)
        score.write("midi", fp=str(mid_path))
        return mid_path.exists()
    except Exception as e:
        print(f"  [error] {mxl_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .mxl files to .mid for StructureLlama.")
    parser.add_argument("--mxl_dir",        required=True,
                        help="Directory containing .mxl source files")
    parser.add_argument("--annotation_dir", required=True,
                        help="Directory of annotation .json files "
                             "(determines which piece IDs to convert)")
    parser.add_argument("--out_dir",        required=True,
                        help="Output directory for .mid files")
    parser.add_argument("--fixes_dir",      default=None,
                        help="Directory of manually corrected .mxl files "
                             "(default: <mxl_dir>/fixes)")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Reconvert files that already have a .mid output")
    args = parser.parse_args()

    mxl_dir   = pathlib.Path(args.mxl_dir)
    ann_dir   = pathlib.Path(args.annotation_dir)
    out_dir   = pathlib.Path(args.out_dir)
    fixes_dir = pathlib.Path(args.fixes_dir) if args.fixes_dir else mxl_dir / "fixes"
    out_dir.mkdir(parents=True, exist_ok=True)

    piece_ids = sorted(int(p.stem) for p in ann_dir.glob("*.json"))
    print(f"Found {len(piece_ids)} annotated pieces to convert.")

    ok = fail = skip = 0
    for pid in tqdm(piece_ids, desc="MXL → MIDI"):
        mid_path = out_dir / f"{pid:05d}.mid"

        mxl_path = fixes_dir / f"{pid:05d}.mxl"
        has_fix  = mxl_path.exists()
        if not has_fix:
            mxl_path = mxl_dir / f"{pid:05d}.mxl"

        if mid_path.exists() and not has_fix and not args.overwrite:
            skip += 1
            continue

        if not mxl_path.exists():
            print(f"  WARNING: no MXL for {pid:05d} — skipped")
            fail += 1
            continue

        if convert(mxl_path, mid_path):
            ok += 1
        else:
            fail += 1

    print(f"\nDone.  Converted: {ok}  Failed: {fail}  Skipped: {skip}")


if __name__ == "__main__":
    main()
