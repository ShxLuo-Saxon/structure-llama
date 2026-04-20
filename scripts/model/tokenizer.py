"""
Compound music token tokenizer for StructureLlama.

Core class copied from Moonbeam-MIDI-Foundation-Model/src/llama_recipes/datasets/music_tokenizer.py.
Extended with:
  - soc_token_compound / eoc_token_compound built-in
  - encode_series_with_conditions() for prepend-based conditioning
  - from_config() class method to build tokenizer from model config + token dict

No external Moonbeam imports.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None  # torch is only needed for encode_series / training; midi_to_compound works without it


def _require_torch(name: str = "this method"):
    if _torch is None:
        raise ImportError(f"torch is required for {name}. Install it with: pip install torch")
    return _torch


# ──────────────────────────────────────────────────────────────
# Pitch helpers
# ──────────────────────────────────────────────────────────────

def pitch_to_octave_pitch_class(pitch: int) -> Tuple[int, int]:
    return pitch // 12, pitch % 12


def octave_pitch_class_to_pitch(octave: int, pitch_class: int) -> int:
    return int(octave * 12 + pitch_class)


# ──────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────

class MusicTokenizer:
    """
    Tokenizer for the Moonbeam compound token representation.

    Each note event is encoded as a 6-tuple:
        [onset_ticks, duration_ticks, octave, pitch_class, instrument, velocity]

    The model operates with *language tokens* — a flat 7-integer representation per event:
        [sos_out_id, timeshift_id, dur_id, octave_id, pitch_id, inst_id, vel_id]
    where sos_out is always 0 and timeshift = delta from previous onset.

    Special compound tokens (all 6 values equal to the same negative integer):
        SOS  = [-1]*6    EOS  = [-2]*6
        PAD  = [-3]*6    SOC  = [-4]*6   (start of condition)
        EOC  = [-5]*6                    (end of condition)
    """

    def __init__(
        self,
        timeshift_vocab_size: int = 1026,
        dur_vocab_size: int = 1026,
        octave_vocab_size: int = 13,
        pitch_class_vocab_size: int = 14,
        instrument_vocab_size: int = 131,
        velocity_vocab_size: int = 130,
        sos_token: int = -1,
        eos_token: int = -2,
        pad_token: int = -3,
        soc_token: int = -4,
        eoc_token: int = -5,
    ):
        self.timeshift_vocab_size = timeshift_vocab_size
        self.dur_vocab_size = dur_vocab_size
        self.octave_vocab_size = octave_vocab_size
        self.pitch_class_vocab_size = pitch_class_vocab_size
        self.instrument_vocab_size = instrument_vocab_size
        self.velocity_vocab_size = velocity_vocab_size
        self.sos_out_vocab_size = 1  # sos_out token at the output side

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.soc_token = soc_token
        self.eoc_token = eoc_token

        # Compound (6-tuple) forms of special tokens
        self.sos_token_compound = [sos_token] * 6
        self.eos_token_compound = [eos_token] * 6
        self.pad_token_compound = [pad_token] * 6
        self.soc_token_compound = [soc_token] * 6
        self.eoc_token_compound = [eoc_token] * 6

        # SOS/EOS language token labels (7-tuple each)
        self.sos_timeshift = timeshift_vocab_size - 2
        self.eos_timeshift = timeshift_vocab_size - 1
        self.sos_dur = dur_vocab_size - 2
        self.eos_dur = dur_vocab_size - 1
        self.sos_octave = octave_vocab_size - 2
        self.eos_octave = octave_vocab_size - 1
        self.sos_pitch_class = pitch_class_vocab_size - 2
        self.eos_pitch_class = pitch_class_vocab_size - 1
        self.sos_instrument = instrument_vocab_size - 2
        self.eos_instrument = instrument_vocab_size - 1
        self.sos_velocity = velocity_vocab_size - 2
        self.eos_velocity = velocity_vocab_size - 1

        # sos_out is always 0
        self.sos_out = 0
        self.sos_label = [self.sos_out, self.sos_timeshift, self.sos_dur,
                          self.sos_octave, self.sos_pitch_class,
                          self.sos_instrument, self.sos_velocity]
        self.eos_label = [self.sos_out, self.eos_timeshift, self.eos_dur,
                          self.eos_octave, self.eos_pitch_class,
                          self.eos_instrument, self.eos_velocity]

        # Language token offset dictionaries (value → flat vocab index)
        self.sos_out_dict = {i: i for i in range(self.sos_out_vocab_size)}
        self.timeshift_dict = {i: i + self.sos_out_vocab_size
                               for i in range(self.timeshift_vocab_size)}
        self.duration_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size
                              for i in range(self.dur_vocab_size)}
        self.octave_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size + self.dur_vocab_size
                            for i in range(self.octave_vocab_size)}
        self.pitch_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size
                               + self.dur_vocab_size + self.octave_vocab_size
                           for i in range(self.pitch_class_vocab_size)}
        self.instrument_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size
                                    + self.dur_vocab_size + self.octave_vocab_size
                                    + self.pitch_class_vocab_size
                                for i in range(self.instrument_vocab_size)}
        self.velocity_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size
                                   + self.dur_vocab_size + self.octave_vocab_size
                                   + self.pitch_class_vocab_size + self.instrument_vocab_size
                              for i in range(self.velocity_vocab_size)}

        # Reverse dicts for decoding
        self.sos_out_dict_decode = {v: k for k, v in self.sos_out_dict.items()}
        self.timeshift_dict_decode = {v: k for k, v in self.timeshift_dict.items()}
        self.duration_dict_decode = {v: k for k, v in self.duration_dict.items()}
        self.octave_dict_decode = {v: k for k, v in self.octave_dict.items()}
        self.pitch_dict_decode = {v: k for k, v in self.pitch_dict.items()}
        self.instrument_dict_decode = {v: k for k, v in self.instrument_dict.items()}
        self.velocity_dict_decode = {v: k for k, v in self.velocity_dict.items()}

        # Will be populated by from_config() if a token dict is provided
        self.additional_token_map: Dict[int, int] = {}  # token_id → embedding_index

    # ──────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config, token_dict_path: Optional[str] = None) -> "MusicTokenizer":
        """
        Build a MusicTokenizer from a model config namespace and optional token dict JSON.

        Args:
            config: SimpleNamespace from config.py with vocab size fields.
            token_dict_path: Path to indexed_tokens_dict.json produced by
                             prepare_condition_csv.py.
        """
        tok = cls(
            timeshift_vocab_size=config.onset_vocab_size,
            dur_vocab_size=config.dur_vocab_size,
            octave_vocab_size=config.octave_vocab_size,
            pitch_class_vocab_size=config.pitch_class_vocab_size,
            instrument_vocab_size=config.instrument_vocab_size,
            velocity_vocab_size=config.velocity_vocab_size,
            sos_token=config.sos_token,
            eos_token=config.eos_token,
            pad_token=config.pad_token,
            soc_token=config.soc_token,
            eoc_token=config.eoc_token,
        )
        if token_dict_path is not None:
            tok._build_additional_token_map(token_dict_path, config)
        return tok

    def _build_additional_token_map(self, token_dict_path: str, config) -> None:
        """
        Build self.additional_token_map: {token_id (negative int) → embedding_index}.

        Embedding index layout in supplementary_embedding:
            0  → SOS
            1  → EOS
            2  → SOC
            3  → EOC
            4  → first metadata token value
            5  → second metadata token value
            ...
            4+N_meta_values → first structure token
            ...

        Args:
            token_dict_path: Path to indexed_tokens_dict.json.
            config: model config namespace (needs soc_token, eoc_token).
        """
        with open(token_dict_path, "r") as f:
            token_dict: Dict[str, int] = json.load(f)

        # Fixed slots for SOS/EOS/SOC/EOC
        self.additional_token_map[config.soc_token] = 2
        self.additional_token_map[config.eoc_token] = 3

        # Assign remaining entries (metadata and structure) starting at index 4
        # Sort by token_id descending (most negative = highest index) for determinism
        other_entries = [(v, k) for k, v in token_dict.items()]  # (token_id, name)
        other_entries.sort(key=lambda x: x[0])  # sort by token_id ascending (most negative first)
        for idx, (token_id, _) in enumerate(other_entries):
            if token_id not in self.additional_token_map:
                self.additional_token_map[token_id] = 4 + idx

    # ──────────────────────────────────────────────────────────
    # Core encode / decode
    # ──────────────────────────────────────────────────────────

    def encode_single(self, raw_token: List) -> List[int]:
        """Pass-through: raw token is already [onset, dur, oct, pitch, inst, vel]."""
        onset, duration, octave, pitch_class, instrument, velocity = raw_token
        return [onset, duration, octave, pitch_class, instrument, velocity]

    def encode_series(
        self,
        raw_token_series,
        if_add_sos: bool = True,
        if_add_eos: bool = True,
    ) -> List[List[int]]:
        """Encode a sequence of raw tokens, optionally adding SOS/EOS."""
        out = [self.encode_single(x) for x in raw_token_series]
        if if_add_sos:
            out = [self.sos_token_compound] + out
        if if_add_eos:
            out = out + [self.eos_token_compound]
        return out

    def encode_series_labels(
        self,
        encoded_tokens: List[List[int]],
        if_added_sos: bool = True,
        if_added_eos: bool = True,
    ) -> List[List[int]]:
        """
        Convert compound tokens to language token labels (7-int each).

        Strips SOS/EOS from input, converts absolute onsets to delta timeshifts,
        then re-adds SOS/EOS labels.
        """
        torch = _require_torch("encode_series_labels")
        t = torch.tensor(encoded_tokens)  # (N, 6)

        if if_added_sos:
            t = t[1:]
        if if_added_eos:
            t = t[:-1]

        # Delta onsets (timeshift)
        timeshift_labels = torch.diff(t[:, 0], prepend=torch.tensor([0]))

        # [0, timeshift, dur, oct, pitch, inst, vel]
        output = torch.cat(
            [torch.zeros(t.shape[0], 1), timeshift_labels.unsqueeze(-1), t[:, 1:]],
            dim=-1,
        )  # (N, 7)

        if if_added_sos:
            output = torch.cat(
                [torch.tensor(self.sos_label).unsqueeze(0), output], dim=0
            )
        if if_added_eos:
            output = torch.cat(
                [output, torch.tensor(self.eos_label).unsqueeze(0)], dim=0
            )

        output_list = output.tolist()
        labels = [self.convert_to_language_tokens(x) for x in output_list]
        return labels

    def convert_to_language_tokens(self, x: List) -> List[int]:
        """Map a 7-value label [sos_out, timeshift, dur, oct, pitch, inst, vel] to flat indices.

        Timeshift and duration are clamped to their valid vocab ranges so that
        unusually long notes or gaps (outside the 0-1023 tick range) do not cause
        a KeyError at training time.
        """
        max_timeshift = self.timeshift_vocab_size - 3  # last 2 slots reserved for SOS/EOS
        max_dur       = self.dur_vocab_size - 3
        timeshift = min(max(int(x[1]), 0), max_timeshift)
        dur       = min(max(int(x[2]), 0), max_dur)
        return [
            self.sos_out_dict[int(x[0])],
            self.timeshift_dict[timeshift],
            self.duration_dict[dur],
            self.octave_dict[int(x[3])],
            self.pitch_dict[int(x[4])],
            self.instrument_dict[int(x[5])],
            self.velocity_dict[int(x[6])],
        ]

    def convert_from_language_tokens(self, inp: "torch.Tensor") -> "torch.Tensor":
        """
        Decode flat language token indices back to compound values.

        inp: (..., 6)  — 6 attribute indices (no sos_out column)
        Returns: (..., 6) — [timeshift, dur, oct, pitch, inst, vel]
        """
        original_shape = inp.shape
        inp_flat = inp.view(-1, original_shape[-1])
        out = []
        for x in inp_flat:
            out.append([
                self.timeshift_dict_decode[x[0].item()],
                self.duration_dict_decode[x[1].item()],
                self.octave_dict_decode[x[2].item()],
                self.pitch_dict_decode[x[3].item()],
                self.instrument_dict_decode[x[4].item()],
                self.velocity_dict_decode[x[5].item()],
            ])
        torch = _require_torch("convert_from_language_tokens")
        result = torch.tensor(out)
        return result.view(*original_shape[:-1], -1)

    # ──────────────────────────────────────────────────────────
    # Prepend-based conditioning (new for StructureLlama)
    # ──────────────────────────────────────────────────────────

    def encode_series_with_conditions(
        self,
        raw_tokens,
        condition_npy: np.ndarray,
        metadata_token_ids: List[int],
    ) -> Tuple[List[List[int]], List[List[int]], List[bool]]:
        """
        Build the full input sequence with prepended condition tokens.

        Sequence layout:
            [SOS] + [meta_tokens] + [struct_tokens] + [SOC] + [music_tokens] + [EOC]

        Args:
            raw_tokens:          (N, 6) array or list of raw compound tokens (no SOS/EOS).
            condition_npy:       (M, 6) int array from _condition.npy.
                                 Each row is [struct_token_id]*6.
            metadata_token_ids:  List of negative token IDs, e.g. [-6, -7].

        Returns:
            input_ids:       List of 6-tuples (full sequence).
            labels:          List of 7-tuples (language tokens).
                             Dummy sos_label at every prefix position (no loss).
            condition_mask:  List[bool], True at prefix positions (SOS+meta+struct+SOC).
        """
        # --- build input_ids ---
        meta_tokens = [[t] * 6 for t in metadata_token_ids]
        cond_tokens = condition_npy.tolist()  # list of 6-lists
        music_with_delimiters = self.encode_series(raw_tokens, if_add_sos=False, if_add_eos=False)

        input_ids = (
            [self.sos_token_compound]
            + meta_tokens
            + cond_tokens
            + [self.soc_token_compound]
            + music_with_delimiters
            + [self.eoc_token_compound]
        )

        # --- build labels ---
        prefix_len = 1 + len(meta_tokens) + len(cond_tokens) + 1  # SOS + meta + struct + SOC
        dummy_label = self.sos_label  # [0, sos_t, sos_d, sos_o, sos_p, sos_i, sos_v]

        # Music labels: music + [EOC]
        # Do NOT prepend SOS here — encode_series_labels with if_added_sos=False gives
        # N+1 labels matching the N+1 music tokens in input_ids (music + EOC).
        # Including SOS would produce N+2 labels, making labels 1 longer than input_ids.
        music_labels = self.encode_series_labels(
            music_with_delimiters + [self.eoc_token_compound],
            if_added_sos=False,
            if_added_eos=True,
        )

        labels = [dummy_label] * prefix_len + music_labels

        # --- build condition mask ---
        condition_mask = [True] * prefix_len + [False] * len(music_labels)

        return input_ids, labels, condition_mask

    # ──────────────────────────────────────────────────────────
    # MIDI ↔ compound conversion (static)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def midi_to_compound(
        midifile,
        TIME_RESOLUTION: int = 100,
        debug: bool = False,
        calibrate_to_default_tempo: bool = False,
    ) -> List[List[int]]:
        """
        Convert a MIDI file to a list of compound tokens.

        Args:
            midifile: File path (str) or mido.MidiFile object.
            TIME_RESOLUTION: Ticks per second (default 100).
            debug: Print warnings for unclosed notes etc.

        Returns:
            List of [onset_ticks, duration_ticks, octave, pitch_class, instrument, velocity].
        """
        if isinstance(midifile, str):
            midi = mido.MidiFile(midifile)
        else:
            midi = midifile

        tokens = []
        note_idx = 0
        open_notes: Dict = defaultdict(list)
        time = 0.0
        instruments: Dict[int, int] = defaultdict(int)
        tempo = 500000  # default: 120 BPM

        for message in midi:
            time += message.time

            if message.time < 0:
                raise ValueError("Negative time in MIDI message")

            if message.type == "program_change":
                instruments[message.channel] = message.program

            elif message.type in ("note_on", "note_off"):
                instr = 128 if message.channel == 9 else instruments[message.channel]

                if message.type == "note_on" and message.velocity > 0:
                    if calibrate_to_default_tempo:
                        time_ticks = round(TIME_RESOLUTION * time * 500000 / tempo)
                    else:
                        time_ticks = round(TIME_RESOLUTION * time)
                    octave, pitch_class = pitch_to_octave_pitch_class(message.note)
                    tokens.append([time_ticks, -1, octave, pitch_class, instr, message.velocity])
                    open_notes[(instr, message.note, message.channel)].append((note_idx, time))
                    note_idx += 1

                else:  # note_off or note_on with velocity=0
                    key = (instr, message.note, message.channel)
                    try:
                        open_idx, onset_time = open_notes[key].pop(0)
                    except IndexError:
                        if debug:
                            print("WARNING: ignoring bad offset")
                    else:
                        if calibrate_to_default_tempo:
                            dur = round(TIME_RESOLUTION * 500000 / tempo * (time - onset_time))
                        else:
                            dur = round(TIME_RESOLUTION * (time - onset_time))
                        if dur == 0:
                            dur = 1
                        tokens[open_idx][1] = dur

            elif message.type == "set_tempo":
                tempo = message.tempo

        # Hard-close all unclosed notes with 250 ms duration
        unclosed = 0
        for v in open_notes.values():
            unclosed += len(v)
            for open_idx, _ in v:
                tokens[open_idx][1] = TIME_RESOLUTION // 4

        if debug and unclosed > 0:
            print(f"WARNING: {unclosed} unclosed notes")

        return tokens

    @staticmethod
    def compound_to_midi(
        tokens,
        TIME_RESOLUTION: int = 100,
        debug: bool = False,
    ) -> mido.MidiFile:
        """
        Convert compound tokens back to a MIDI file.

        Args:
            tokens: (N, 6) array or list of [onset, dur, oct, pitch_class, inst, vel].
            TIME_RESOLUTION: Ticks per second (must match midi_to_compound).

        Returns:
            mido.MidiFile
        """
        mid = mido.MidiFile()
        mid.ticks_per_beat = TIME_RESOLUTION // 2  # 2 beats/second at tempo=120

        time_index: Dict = defaultdict(list)
        for row in tokens:
            t, dur, octave, pitch_class, instrument, velocity = row
            note = octave_pitch_class_to_pitch(octave, pitch_class)
            # Skip boundary/sentinel sub-tokens (GRU sos/eos markers leak into note slots
            # at section transitions in M1).  These are not real note events.
            if not (0 <= note <= 127 and 0 <= velocity <= 127):
                continue
            time_index[(t, 0)].append((note, instrument, velocity))
            time_index[(t + dur, 1)].append((note, instrument, velocity))

        track_idx: Dict = {}
        num_tracks = 0

        for (t, etype) in sorted(time_index.keys()):
            for note, instrument, velocity in time_index[(t, etype)]:
                if etype == 0:  # onset
                    if instrument not in track_idx:
                        previous_time = 0
                        track = mido.MidiTrack()
                        mid.tracks.append(track)
                        if instrument == 128:
                            idx = 9
                            track.append(mido.Message("program_change", channel=idx, program=0))
                        else:
                            # vocab has 131 slots: 0-127 = GM, 128 = percussion,
                            # 129 = sos_instrument, 130 = eos_instrument (GRU boundary sub-tokens).
                            # Training data is piano-only, so fall back to 0 (Acoustic Grand Piano).
                            program = instrument if instrument <= 127 else 0
                            if debug and program != instrument:
                                print(f"WARNING: instrument token {instrument} out of MIDI range, clamped to {program} (piano)")
                            # Clamp channel to 0-15, skip drum channel 9.
                            # num_tracks can exceed 15 if model hallucinates many instruments.
                            raw_idx = num_tracks if num_tracks < 9 else num_tracks + 1
                            idx = min(raw_idx, 14)   # MIDI channels 0-15; 9=drums reserved
                            track.append(mido.Message("program_change", channel=idx, program=program))
                        num_tracks += 1
                        if num_tracks == 9:
                            num_tracks += 1  # skip drum channel
                        track_idx[instrument] = (track, 0, idx)

                    track, prev_t, idx = track_idx[instrument]
                    track.append(mido.Message("note_on", note=note, channel=idx,
                                              velocity=velocity, time=t - prev_t))
                    track_idx[instrument] = (track, t, idx)

                else:  # offset
                    if instrument not in track_idx:
                        if debug:
                            print("IGNORING bad offset")
                        continue
                    track, prev_t, idx = track_idx[instrument]
                    track.append(mido.Message("note_off", note=note, channel=idx,
                                              time=t - prev_t))
                    track_idx[instrument] = (track, t, idx)

        return mid
