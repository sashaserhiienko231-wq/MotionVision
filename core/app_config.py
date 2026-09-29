from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from camera import CameraRequest, parse_fps, parse_resolution


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE lines without overwriting shell environment."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    camera: CameraRequest = CameraRequest()
    auto_record_on_start: bool = False
    face_recognition_enabled: bool = True
    face_match_threshold: float = 0.6
    face_alert_cooldown: int = 30
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_max_video_mb: int = 45

    @classmethod
    def from_env(cls) -> AppConfig:
        load_env_file()
        try:
            camera_index = max(0, int(os.getenv("CAMERA_INDEX", "0")))
        except ValueError:
            camera_index = 0
        try:
            fps = parse_fps(os.getenv("CAMERA_FPS", "Auto"))
        except ValueError:
            fps = None
        try:
            resolution = parse_resolution(os.getenv("CAMERA_RESOLUTION", "Auto"))
        except ValueError:
            resolution = None
        try:
            cooldown = max(0, min(86400, int(os.getenv("FACE_ALERT_COOLDOWN", "30"))))
        except ValueError:
            cooldown = 30
        try:
            max_video_mb = max(1, min(2000, int(os.getenv("TELEGRAM_MAX_VIDEO_MB", "45"))))
        except ValueError:
            max_video_mb = 45
        return cls(
            camera_index=camera_index,
            camera=CameraRequest(resolution, fps),
            auto_record_on_start=os.getenv("AUTO_RECORD_ON_START", "false").lower() in {"1", "true", "yes", "on"},
            face_recognition_enabled=_env_bool("FACE_RECOGNITION_ENABLED", True),
            face_match_threshold=_env_float("FACE_MATCH_THRESHOLD", 0.6, 0.0, 1.0),
            face_alert_cooldown=cooldown,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            telegram_max_video_mb=max_video_mb,
        )


class WindowState:
    """Persist window and active camera state in non-secret JSON."""
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.data: dict[str, Any] = value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            self.data = {}

    def update(self, **values: Any) -> None:
        self.data.update(values)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
