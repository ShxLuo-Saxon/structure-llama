"""
Model configuration loader for StructureLlama.

Loads a JSON config file (e.g. model_config_structure_llama_309M.json) and returns
a SimpleNamespace that downstream code can access with attribute syntax.

No external Moonbeam imports.
"""

import json
from types import SimpleNamespace
from typing import Union


def load_config(json_path: str) -> SimpleNamespace:
    """
    Load a model config JSON and return a SimpleNamespace.

    The 'decoder' sub-dict is also converted to a SimpleNamespace so that
    gru_decoder.OutputGRU(config.decoder) works.

    Args:
        json_path: Path to the config JSON file.

    Returns:
        SimpleNamespace with all config fields accessible as attributes.
        config.decoder is itself a SimpleNamespace.
    """
    with open(json_path, "r") as f:
        d = json.load(f)

    # Convert 'decoder' sub-dict to SimpleNamespace
    if "decoder" in d and isinstance(d["decoder"], dict):
        d["decoder"] = SimpleNamespace(**d["decoder"])

    cfg = SimpleNamespace(**d)
    return cfg


def config_to_dict(cfg: SimpleNamespace) -> dict:
    """Convert a SimpleNamespace config back to a plain dict (for serialisation)."""
    d = vars(cfg).copy()
    if "decoder" in d and isinstance(d["decoder"], SimpleNamespace):
        d["decoder"] = vars(d["decoder"])
    return d
