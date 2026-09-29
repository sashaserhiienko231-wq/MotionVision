from __future__ import annotations

import json
from pathlib import Path

from paths import resource_root


def application_version() -> str:
    try:
        data = json.loads((resource_root() / "config" / "version.json").read_text(encoding="utf-8"))
        return str(data["version"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "1.0.0"
