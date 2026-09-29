"""Resource and writable-data locations for source and frozen builds."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[1]


def user_data_dir() -> Path:
    if not getattr(sys, "frozen", False):
        return Path.cwd()
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return parent / "Motion Vision"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Motion Vision"
    parent = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return parent / "motion-vision"


def initialize_frozen_runtime() -> None:
    """Use per-user writable state and load optional secrets from its .env."""
    if not getattr(sys, "frozen", False):
        return
    data_dir = user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(data_dir)
    try:
        from app_config import load_env_file

        load_env_file(data_dir / ".env")
    except OSError:
        pass
