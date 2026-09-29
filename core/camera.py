"""Camera mode negotiation helpers.

OpenCV drivers frequently ignore requested properties. The helpers in this
module only report dimensions observed on frames; reported CAP_PROP values are
kept as diagnostic hints and never presented as proof of actual output.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

RESOLUTIONS: tuple[tuple[int, int] | None, ...] = (
    None,
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
FPS_OPTIONS: tuple[int | None, ...] = (None, 30, 60, 120)


@dataclass(frozen=True)
class CameraRequest:
    resolution: tuple[int, int] | None = None
    fps: int | None = None


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: float

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps:.0f} FPS"


@dataclass
class CameraReadRecovery:
    """Count consecutive bad reads before reconnecting a live camera."""

    reconnect_after: int = 30
    consecutive_failures: int = 0

    def observe(self, success: bool, frame: Any | None) -> bool:
        valid = (
            success
            and frame is not None
            and getattr(frame, "ndim", 0) >= 2
            and min(frame.shape[:2]) >= 2
        )
        if valid:
            self.consecutive_failures = 0
            return False
        self.consecutive_failures += 1
        return self.consecutive_failures >= max(1, self.reconnect_after)


def parse_resolution(value: str) -> tuple[int, int] | None:
    if value.lower() == "auto":
        return None
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, TypeError):
        raise ValueError("Resolution must be Auto or WIDTHxHEIGHT") from None
    if width <= 0 or height <= 0 or (width, height) not in RESOLUTIONS:
        raise ValueError("Resolution must be Auto, 640x480, 1280x720, 1920x1080, 2560x1440, or 3840x2160")
    return width, height


def parse_fps(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    try:
        fps = int(value)
    except (ValueError, TypeError):
        raise ValueError("FPS must be Auto, 30, 60, or 120") from None
    if fps not in (30, 60, 120):
        raise ValueError("FPS must be Auto, 30, 60, or 120")
    return fps


def _mode_candidates(request: CameraRequest) -> list[CameraRequest]:
    resolutions = sorted(
        (item for item in RESOLUTIONS if item is not None),
        key=lambda size: size[0] * size[1], reverse=True,
    )
    if request.resolution is None:
        pass
    else:
        requested_area = request.resolution[0] * request.resolution[1]
        resolutions = [size for size in resolutions if size[0] * size[1] <= requested_area]
        resolutions.sort(key=lambda size: (size != request.resolution, -(size[0] * size[1])))
    if request.fps is None:
        fps_values = [120, 60, 30]
    else:
        fps_values = [fps for fps in (120, 60, 30) if fps <= request.fps]
    candidates: list[CameraRequest] = []
    for resolution in resolutions:
        for fps in fps_values:
            candidate = CameraRequest(resolution, fps)
            if candidate not in candidates:
                candidates.append(candidate)
    candidates.append(CameraRequest())
    return candidates


def open_camera(
    index: int,
    request: CameraRequest,
    capture_factory: Callable[[int], Any],
    *,
    prop_width: int,
    prop_height: int,
    prop_fps: int,
    prop_buffer: int,
    sample_frames: int = 4,
) -> tuple[Any, CameraMode]:
    """Try likely modes and keep the capture with the best real pixel rate."""
    capture = capture_factory(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera {index}")
    best_mode: CameraMode | None = None
    best_request: CameraRequest | None = None
    best_score = -1.0
    tried_actual: set[tuple[int, int, int]] = set()
    def configure(capture: Any, candidate: CameraRequest) -> None:
        if candidate.resolution:
            capture.set(prop_width, candidate.resolution[0])
            capture.set(prop_height, candidate.resolution[1])
        if candidate.fps:
            capture.set(prop_fps, candidate.fps)
        capture.set(prop_buffer, 1)

    def sample(capture: Any, count: int) -> tuple[Any | None, float]:
        frame = None
        valid_count = 0
        started = time.perf_counter()
        for _ in range(max(1, count)):
            ok, candidate_frame = capture.read()
            if ok and candidate_frame is not None and getattr(candidate_frame, "ndim", 0) >= 2:
                frame = candidate_frame
                valid_count += 1
        elapsed = time.perf_counter() - started
        measured = (valid_count - 1) / elapsed if elapsed > 1e-4 and valid_count > 1 else 0.0
        return frame, measured

    try:
        for candidate in _mode_candidates(request):
            configure(capture, candidate)
            frame, measured_fps = sample(capture, sample_frames)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            reported_fps = float(capture.get(prop_fps) or 0.0)
            if 1.0 <= measured_fps <= 240.0:
                reported_fps = measured_fps
            elif not (1.0 <= reported_fps <= 240.0):
                reported_fps = 0.0
            key = (width, height, round(reported_fps))
            if key in tried_actual:
                continue
            tried_actual.add(key)
            mode = CameraMode(width, height, reported_fps)
            score = width * height * max(reported_fps, 1.0)
            if score > best_score:
                best_request, best_mode, best_score = candidate, mode, score
            resolution_matches = request.resolution is None or (width, height) == request.resolution
            fps_matches = request.fps is None or (reported_fps > 0 and abs(reported_fps - request.fps) <= max(2, request.fps * 0.05))
            # Auto is a fallback probe after all requested modes have been
            # sampled. Returning it here would skip the best mode discovered
            # earlier when its short FPS sample differs by a few frames.
            if (request.resolution is not None or request.fps is not None) and candidate == request and resolution_matches and fps_matches:
                return capture, mode
        if best_mode is None or best_request is None:
            capture.release()
            raise RuntimeError(f"Could not read frames from camera {index}")
        configure(capture, best_request)
        frame, measured_fps = sample(capture, sample_frames)
        if frame is None:
            capture.release()
            raise RuntimeError(f"Camera {index} stopped producing frames")
        height, width = frame.shape[:2]
        reported_fps = measured_fps if 1.0 <= measured_fps <= 240.0 else float(capture.get(prop_fps) or best_mode.fps)
        if not 1.0 <= reported_fps <= 240.0:
            reported_fps = best_mode.fps
        return capture, CameraMode(width, height, reported_fps)
    except Exception:
        capture.release()
        raise
