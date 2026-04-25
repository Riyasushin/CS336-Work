from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "TINY_TRAINING_DATA_ASSETS"
_DEFAULT_DIR = Path.home() / ".cache" / "tiny_training_data"


def assets_dir() -> Path:
    env = os.environ.get(_ENV_VAR)
    return Path(env) if env else _DEFAULT_DIR


def asset_path(filename: str) -> Path:
    path = assets_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Asset {filename!r} not found at {path}. "
            f"Set ${_ENV_VAR} to the directory holding the model, "
            f"or place the file at the default location ({_DEFAULT_DIR})."
        )
    return path
