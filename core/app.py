from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
import psutil
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from actions import (
    ActionEvent,
    ActionRecognizer,
    LandmarkPoint,
    TrackingResults,
    tracking_results_from_tasks,
)
from app_config import AppConfig, WindowState
from camera import (
    CameraMode,
    CameraReadRecovery,
    CameraRequest,
    open_camera,
    parse_fps,
    parse_resolution,
)
from face_identity import FaceDatabase, FaceTrack, FaceTracker, face_box
from hud_drag import GestureHudController, HudPanelLayout
from paths import initialize_frozen_runtime, resource_root
from sface import LocalFaceEmbedder
from telegram_bot import TelegramBot
from version import application_version
from windowing import (
    WindowManager,
    WindowRect,
    centered_window_rect,
    normalize_window_rect,
)

MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

FACE_OVAL_CONNECTIONS = (
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
)
PALM_BOUNDARY = (0, 1, 5, 9, 13, 17)


@dataclass
class RuntimeMetrics:
    camera_fps: float = 0.0
    detection_fps: float = 0.0
    render_fps: float = 0.0
    inference_ms: float = 0.0
    frame_latency_ms: float = 0.0
    total_frames: int = 0
    total_actions: int = 0
    face_detected: bool = False
    body_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    resolution: str = "--"
    active_models: str = "Pose Lite / Hands / Face"
    current_mode: str = "Balanced Mode"
    pose_detection_fps: float = 0.0
    hand_detection_fps: float = 0.0
    face_detection_fps: float = 0.0
    face_recognition_fps: float = 0.0
    requested_camera: str = "Auto"
    actual_camera: str = "--"
    inference_resolution: str = "--"
    faces_found: int = 0


@dataclass(frozen=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def intersection_area(self, other: PixelRect) -> int:
        width = max(0, min(self.right, other.right) - max(self.x, other.x))
        height = max(0, min(self.bottom, other.bottom) - max(self.y, other.y))
        return width * height

    def inside(self, width: int, height: int, margin: int = 20) -> bool:
        return (
            self.x >= margin
            and self.y >= margin
            and self.right <= width - margin
            and self.bottom <= height - margin
        )


@dataclass(frozen=True)
class ViewportTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    crop_x: float
    crop_y: float

    @classmethod
    def cover(
        cls,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> ViewportTransform:
        scale = max(target_width / source_width, target_height / source_height)
        scaled_width = source_width * scale
        scaled_height = source_height * scale
        return cls(
            source_width,
            source_height,
            target_width,
            target_height,
            scale,
            max(0.0, (scaled_width - target_width) / 2),
            max(0.0, (scaled_height - target_height) / 2),
        )

    def point(self, normalized_x: float, normalized_y: float) -> tuple[int, int]:
        x = round(normalized_x * self.source_width * self.scale - self.crop_x)
        y = round(normalized_y * self.source_height * self.scale - self.crop_y)
        return x, y

    def bounds(self, landmarks: list[Any], padding: int = 0) -> PixelRect | None:
        if not landmarks:
            return None
        points = [self.point(float(item.x), float(item.y)) for item in landmarks]
        left = max(0, min(point[0] for point in points) - padding)
        top = max(0, min(point[1] for point in points) - padding)
        right = min(self.target_width, max(point[0] for point in points) + padding + 1)
        bottom = min(self.target_height, max(point[1] for point in points) + padding + 1)
        if right <= left or bottom <= top:
            return None
        return PixelRect(left, top, max(1, right - left), max(1, bottom - top))


class HudLayoutEngine:
    def __init__(self, width: int, height: int, margin: int = 20) -> None:
        self.width = width
        self.height = height
        self.margin = margin

    def place_panel(
        self,
        panel_width: int,
        panel_height: int,
        obstacles: tuple[PixelRect, ...] = (),
        reserved: tuple[PixelRect, ...] = (),
        anchor: str = "left",
    ) -> PixelRect:
        panel_width = min(panel_width, max(0, self.width - self.margin * 2))
        panel_height = min(panel_height, max(0, self.height - self.margin * 2))
        if panel_width <= 0 or panel_height <= 0:
            return PixelRect(self.margin, self.margin, 0, 0)

        left_x = self.margin
        right_x = self.width - self.margin - panel_width
        top_y = self.margin
        bottom_y = self.height - self.margin - panel_height
        center_x = (self.width - panel_width) // 2
        center_y = (self.height - panel_height) // 2
        candidates = {
            PixelRect(x, y, panel_width, panel_height)
            for x in (left_x, center_x, right_x)
            for y in (top_y, center_y, bottom_y)
        }

        preferred_x = left_x if anchor == "left" else right_x

        def score(rectangle: PixelRect) -> tuple[int, int, int]:
            tracked_overlap = sum(
                rectangle.intersection_area(obstacle) for obstacle in obstacles
            )
            panel_overlap = sum(
                rectangle.intersection_area(panel) for panel in reserved
            )
            anchor_distance = abs(rectangle.x - preferred_x) + abs(rectangle.y - top_y)
            return tracked_overlap, panel_overlap, anchor_distance

        return min(candidates, key=score)

    def place_hand_label(
        self,
        hand_bounds: PixelRect,
        label_width: int,
        label_height: int,
        obstacles: tuple[PixelRect, ...] = (),
        reserved: tuple[PixelRect, ...] = (),
    ) -> PixelRect | None:
        label_width = min(label_width, self.width - self.margin * 2)
        label_height = min(label_height, self.height - self.margin * 2)
        y = hand_bounds.y - label_height - 6
        if y < self.margin:
            return None

        center_x = hand_bounds.x + (hand_bounds.width - label_width) // 2
        x_candidates = {
            center_x,
            hand_bounds.x,
            hand_bounds.right - label_width,
            self.margin,
            self.width - self.margin - label_width,
        }
        candidates = [
            PixelRect(x, y, label_width, label_height)
            for x in x_candidates
            if PixelRect(x, y, label_width, label_height).inside(
                self.width, self.height, self.margin
            )
        ]
        if not candidates:
            return None

        def score(rectangle: PixelRect) -> tuple[int, int, int]:
            tracked_overlap = sum(
                rectangle.intersection_area(obstacle) for obstacle in obstacles
            )
            label_overlap = sum(
                rectangle.intersection_area(label) for label in reserved
            )
            center_distance = abs(
                rectangle.x + rectangle.width // 2
                - (hand_bounds.x + hand_bounds.width // 2)
            )
            return tracked_overlap, label_overlap, center_distance

        return min(candidates, key=score)


@dataclass(frozen=True)
class ProcessingMode:
    name: str
    pose_interval: int
    face_interval: int
    inference_scale: float


PROCESSING_MODES = (
    ProcessingMode("Quality Mode", pose_interval=1, face_interval=1, inference_scale=1.0),
    ProcessingMode("Balanced Mode", pose_interval=2, face_interval=3, inference_scale=0.75),
    ProcessingMode("Performance Mode", pose_interval=4, face_interval=4, inference_scale=0.5),
)


class FrameInferenceScheduler:
    @staticmethod
    def should_run(frame_index: int, interval: int, force: bool = False) -> bool:
        return force or frame_index % max(1, interval) == 0


class MediaPipeTasks:
    def __init__(self, models_dir: Path, *, delegate: Any | None = None) -> None:
        self.models_dir = models_dir
        self.landmarkers: list[Any] = []
        self.last_timestamp_ms = -1
        self._rgb_buffer: np.ndarray | None = None
        self._inference_bgr_buffer: np.ndarray | None = None
        self._input_shape: tuple[int, ...] | None = None
        self._frame_index = -1
        self._force_refresh = True
        self.mode = PROCESSING_MODES[1]
        self._cached_pose_landmarks: list[LandmarkPoint] | None = None
        self._cached_poses_landmarks: list[list[LandmarkPoint]] = []
        self._cached_face_landmarks: list[LandmarkPoint] | None = None
        self._cached_faces_landmarks: list[list[LandmarkPoint]] = []
        self.last_inference_ms = 0.0
        self.last_pose_ms = self.last_hand_ms = self.last_face_ms = 0.0
        self.pose_detection_fps = self.hand_detection_fps = self.face_detection_fps = 0.0
        self._last_task_run_at: dict[str, float] = {}
        self.model_paths: dict[str, Path] = {}
        self._face_image_landmarker: Any | None = None
        self._face_embedder: LocalFaceEmbedder | None = None
        model_paths = {
            filename: self._ensure_model(filename, url)
            for filename, url in MODEL_URLS.items()
        }
        self.model_paths = model_paths
        try:
            base_options = mp_tasks.BaseOptions
            video_mode = vision.RunningMode.VIDEO

            def task_base_options(filename: str) -> Any:
                options: dict[str, Any] = {
                    "model_asset_path": str(model_paths[filename]),
                }
                if delegate is not None:
                    options["delegate"] = delegate
                return base_options(**options)

            self.pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=task_base_options("pose_landmarker_lite.task"),
                    running_mode=video_mode,
                    num_poses=4,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self.landmarkers.append(self.pose)
            self.hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=task_base_options("hand_landmarker.task"),
                    running_mode=video_mode,
                    num_hands=8,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self.landmarkers.append(self.hands)
            self.face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=task_base_options("face_landmarker.task"),
                    running_mode=video_mode,
                    num_faces=10,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self.landmarkers.append(self.face)
        except Exception:
            self.close()
            raise

    def _ensure_model(self, filename: str, url: str) -> Path:
        path = self.models_dir / filename
        if path.is_file() and path.stat().st_size > 0:
            return path

        bundled_path = resource_root() / "models" / filename
        if bundled_path.is_file() and bundled_path.stat().st_size > 0:
            return bundled_path

        self.models_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".download")
        if temporary_path.is_file() and temporary_path.stat().st_size > 100_000:
            return temporary_path
        try:
            with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary_path.stat().st_size <= 100_000:
                raise RuntimeError(f"Downloaded model is incomplete: {filename}")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    def set_mode(self, mode: ProcessingMode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._force_refresh = True

    def detect(self, bgr_frame: np.ndarray) -> TrackingResults:
        if not hasattr(self, "_last_task_run_at"):
            self._last_task_run_at = {}
        self._frame_index += 1
        timestamp_ms = max(self.last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self.last_timestamp_ms = timestamp_ms

        height, width = bgr_frame.shape[:2]
        if self._input_shape != bgr_frame.shape:
            self._input_shape = bgr_frame.shape
            self._cached_pose_landmarks = None
            self._cached_poses_landmarks = []
            self._cached_face_landmarks = None
            self._cached_faces_landmarks = []
            self._force_refresh = True

        if self.mode.inference_scale < 1.0:
            inference_width = max(1, round(width * self.mode.inference_scale))
            inference_height = max(1, round(height * self.mode.inference_scale))
            buffer_shape = (inference_height, inference_width, bgr_frame.shape[2])
            if self._inference_bgr_buffer is None or self._inference_bgr_buffer.shape != buffer_shape:
                self._inference_bgr_buffer = np.empty(buffer_shape, dtype=bgr_frame.dtype)
            inference_frame = cv2.resize(
                bgr_frame,
                (inference_width, inference_height),
                dst=self._inference_bgr_buffer,
                interpolation=cv2.INTER_AREA,
            )
        else:
            inference_frame = bgr_frame

        if self._rgb_buffer is None or self._rgb_buffer.shape != inference_frame.shape:
            self._rgb_buffer = np.empty_like(inference_frame)
        cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb_buffer)
        started_at = time.perf_counter()
        should_refresh = self._force_refresh or self._frame_index == 0
        if FrameInferenceScheduler.should_run(
            self._frame_index, self.mode.pose_interval, should_refresh
        ):
            task_started = time.perf_counter()
            pose_result = self.pose.detect_for_video(image, timestamp_ms)
            task_finished = time.perf_counter()
            self.last_pose_ms = (task_finished - task_started) * 1000.0
            self.pose_detection_fps = 1.0 / max(task_finished - self._last_task_run_at.get("pose", task_started), 1e-6) if "pose" in self._last_task_run_at else 0.0
            self._last_task_run_at["pose"] = task_finished
            pose_data = tracking_results_from_tasks(pose_result, None, None)
            self._cached_pose_landmarks = pose_data.pose_landmarks
            self._cached_poses_landmarks = pose_data.poses_landmarks
        if FrameInferenceScheduler.should_run(
            self._frame_index, self.mode.face_interval, should_refresh
        ):
            task_started = time.perf_counter()
            face_result = self.face.detect_for_video(image, timestamp_ms)
            task_finished = time.perf_counter()
            self.last_face_ms = (task_finished - task_started) * 1000.0
            self.face_detection_fps = 1.0 / max(task_finished - self._last_task_run_at.get("face", task_started), 1e-6) if "face" in self._last_task_run_at else 0.0
            self._last_task_run_at["face"] = task_finished
            face_data = tracking_results_from_tasks(None, None, face_result)
            self._cached_face_landmarks = face_data.face_landmarks
            self._cached_faces_landmarks = face_data.faces_landmarks
        task_started = time.perf_counter()
        hand_result = self.hands.detect_for_video(image, timestamp_ms)
        task_finished = time.perf_counter()
        self.last_hand_ms = (task_finished - task_started) * 1000.0
        self.hand_detection_fps = 1.0 / max(task_finished - self._last_task_run_at.get("hand", task_started), 1e-6) if "hand" in self._last_task_run_at else 0.0
        self._last_task_run_at["hand"] = task_finished
        self._force_refresh = False
        self.last_inference_ms = (time.perf_counter() - started_at) * 1000.0
        hands = tracking_results_from_tasks(None, hand_result, None).hand_landmarks
        return TrackingResults(
            pose_landmarks=self._cached_pose_landmarks,
            hand_landmarks=hands,
            face_landmarks=self._cached_face_landmarks,
            faces_landmarks=self._cached_faces_landmarks,
            poses_landmarks=self._cached_poses_landmarks,
        )

    def encode_face_image(self, image_bgr: np.ndarray) -> list[float] | None:
        """Create a local SFace descriptor from a Telegram enrollment image."""
        if self._face_image_landmarker is None:
            self._face_image_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(
                        model_asset_path=str(self.model_paths["face_landmarker.task"])
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                )
            )
            self.landmarkers.append(self._face_image_landmarker)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_image_landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        faces = getattr(result, "face_landmarks", None) or []
        return self.embed_face(image_bgr, faces[0]) if faces else None

    def embed_face(self, image_bgr: np.ndarray, landmarks: Any) -> list[float] | None:
        if self._face_embedder is None:
            self._face_embedder = LocalFaceEmbedder(self.models_dir)
        return self._face_embedder.embed(image_bgr, landmarks)

    def close(self) -> None:
        for landmarker in reversed(self.landmarkers):
            landmarker.close()
        self.landmarkers.clear()

    def __enter__(self) -> MediaPipeTasks:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class JsonHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[ActionEvent] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.events = [ActionEvent(**item) for item in data]
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            self.events = []

    def add(self, event: ActionEvent) -> None:
        self.events.append(event)
        self.save()

    def clear(self) -> None:
        self.events.clear()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = [asdict(event) for event in self.events]
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)


class VisionApp:
    def __init__(
        self,
        camera_index: int | None = None,
        history_path: Path = Path("actions.json"),
        recordings_dir: Path = Path("recordings"),
        models_dir: Path = Path("models"),
        ui_config_path: Path = Path("motion_vision_ui.json"),
        config: AppConfig | None = None,
        config_path: Path = Path("motion_vision_config.json"),
        faces_path: Path = Path("faces.json"),
        reset_window: bool = False,
    ) -> None:
        supplied_config = config is not None
        self.config = config or AppConfig.from_env()
        self.history = JsonHistory(history_path)
        self.recordings_dir = recordings_dir
        self.models_dir = models_dir
        self.window_state = WindowState(config_path)
        self.window_manager = WindowManager()
        self.reset_window_on_start = reset_window
        if not supplied_config:
            saved_camera = self.window_state.data
            if "CAMERA_RESOLUTION" not in os.environ and saved_camera.get("camera_request_resolution"):
                try:
                    restored_resolution = parse_resolution(str(saved_camera["camera_request_resolution"]))
                    self.config = replace(self.config, camera=CameraRequest(restored_resolution, self.config.camera.fps))
                except ValueError:
                    pass
            if "CAMERA_FPS" not in os.environ and saved_camera.get("camera_request_fps") is not None:
                try:
                    restored_fps = parse_fps(str(saved_camera["camera_request_fps"]))
                    self.config = replace(self.config, camera=CameraRequest(self.config.camera.resolution, restored_fps))
                except ValueError:
                    pass
            if camera_index is None and isinstance(saved_camera.get("camera_index"), int):
                self.config = replace(self.config, camera_index=max(0, saved_camera["camera_index"]))
        self.camera_index = self.config.camera_index if camera_index is None else camera_index
        self.face_database = FaceDatabase(faces_path)
        self.face_tracker = FaceTracker()
        self.face_tracks: list[FaceTrack] = []
        self.watch_mode = False
        self._last_face_alert: dict[str, float] = {}
        self._face_embedding_cache: dict[tuple[float, ...], list[float] | None] = {}
        self._last_snapshot: Path | None = None
        self._latest_frame: np.ndarray | None = None
        self._last_valid_viewport: tuple[int, int] | None = None
        self._restored_window_rect: tuple[int, int, int, int] | None = None
        self._was_maximized_before_fullscreen = False
        self._camera_mode: CameraMode | None = None
        self._camera_change_requested = False
        self.telegram = TelegramBot(
            self.config.telegram_bot_token,
            self.config.telegram_chat_id,
            on_status=self._telegram_status,
        )
        self.panel_layout = HudPanelLayout(ui_config_path)
        self.hud_drag = GestureHudController(self.panel_layout)
        self.recognizer = ActionRecognizer()
        self.recent_actions: deque[ActionEvent] = deque(maxlen=5)
        self.action_counts: Counter[str] = Counter()
        self.metrics = RuntimeMetrics()
        self.processing_mode = PROCESSING_MODES[1]
        saved_mode = self.window_state.data.get("processing_mode")
        self.processing_mode = next((mode for mode in PROCESSING_MODES if mode.name == saved_mode), self.processing_mode)
        self.show_hud = True
        self.show_stats = True
        self.light_theme = False
        self.fullscreen = False
        self.debug_mode = False
        self.video_writer: cv2.VideoWriter | None = None
        self._recording_size: tuple[int, int] | None = None
        self._recording_path: Path | None = None
        self._recording_started_at: float | None = None
        self._recording_fps = 0.0
        self._palm_overlay_buffer: np.ndarray | None = None
        self.status_message = "Ready"
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)
        self._last_capture_at: float | None = None
        self._last_render_at: float | None = None
        self._last_resource_sample_at = 0.0
        self.metrics.requested_camera = self._camera_request_label()

    def _telegram_status(self, message: str) -> None:
        self.status_message = message

    def _camera_request_label(self) -> str:
        resolution = (
            "Auto"
            if self.config.camera.resolution is None
            else f"{self.config.camera.resolution[0]}x{self.config.camera.resolution[1]}"
        )
        fps = "Auto" if self.config.camera.fps is None else str(self.config.camera.fps)
        return f"{resolution} @ {fps} FPS"

    @staticmethod
    def _ema(previous: float, current: float, alpha: float = 0.2) -> float:
        return current if previous <= 0.0 else previous + alpha * (current - previous)

    def _sample_resources(self, now: float) -> None:
        if now - self._last_resource_sample_at < 0.5:
            return
        self._last_resource_sample_at = now
        try:
            logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
            process_cpu = self._process.cpu_percent(interval=None) / logical_cpus
            self.metrics.cpu_percent = min(100.0, max(0.0, process_cpu))
            self.metrics.ram_mb = self._process.memory_info().rss / (1024 * 1024)
        except (psutil.Error, OSError):
            self.metrics.cpu_percent = 0.0
            self.metrics.ram_mb = 0.0

    @staticmethod
    def _draw_face_box(
        frame: Any,
        face_landmarks: Any,
        transform: ViewportTransform | None = None,
    ) -> PixelRect | None:
        if not face_landmarks:
            return None
        height, width = frame.shape[:2]
        face_bounds = (
            transform.bounds(face_landmarks, padding=2)
            if transform is not None
            else ViewportTransform.cover(width, height, width, height).bounds(
                face_landmarks, padding=2
            )
        )
        if face_bounds is None:
            return None
        x1, y1 = face_bounds.x, face_bounds.y
        x2, y2 = face_bounds.right - 1, face_bounds.bottom - 1
        face_color = (255, 220, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2)
        return face_bounds

    @staticmethod
    def _pixel(
        frame: Any,
        landmark: Any,
        transform: ViewportTransform | None = None,
    ) -> tuple[int, int]:
        height, width = frame.shape[:2]
        if transform is not None:
            return transform.point(float(landmark.x), float(landmark.y))
        return (
            max(0, min(width - 1, round(landmark.x * width))),
            max(0, min(height - 1, round(landmark.y * height))),
        )

    @staticmethod
    def _responsive_scale(width: int, height: int) -> float:
        return max(0.32, min(0.72, min(width / 1050, height / 720) * 0.62))

    @staticmethod
    def _wrap_lines(
        lines: list[tuple[str, tuple[int, int, int]]],
        max_width: int,
        scale: float,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        wrapped: list[tuple[str, tuple[int, int, int]]] = []
        for text, color in lines:
            words = text.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                measured = cv2.getTextSize(
                    candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
                )[0][0]
                if current and measured > max_width:
                    wrapped.append((current, color))
                    current = word
                else:
                    current = candidate
                if cv2.getTextSize(
                    current, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
                )[0][0] > max_width:
                    wrapped.append((VisionApp._fit_text(current, max_width, scale), color))
                    current = ""
            if current:
                wrapped.append((current, color))
        return wrapped

    @classmethod
    def _panel_content(
        cls,
        width: int,
        height: int,
        title: str,
        lines: list[tuple[str, tuple[int, int, int]]],
    ) -> tuple[int, int, float, int, int, list[tuple[str, tuple[int, int, int]]]]:
        margin = 20
        max_panel_width = max(0, width - margin * 2)
        max_panel_height = max(0, height - margin * 2)
        if max_panel_width == 0 or max_panel_height == 0:
            return 0, 0, 0.0, 0, 0, []

        scale = cls._responsive_scale(width, height)
        padding = max(7, round(14 * scale / 0.52))
        line_height = max(13, round(22 * scale / 0.52))
        title_height = max(12, round(22 * scale / 0.52))
        max_width = min(max_panel_width, max(120, round(width * 0.42)))
        title_width = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0]
        desired_width = max(
            [title_width]
            + [
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                for text, _ in lines
            ]
        )
        panel_width = min(max_width, max(min(max_width, 180), desired_width + padding * 2))
        content_width = max(1, panel_width - padding * 2)
        wrapped = cls._wrap_lines(lines, content_width, scale)

        while scale > 0.32:
            needed_height = padding * 2 + title_height + 6 + len(wrapped) * line_height
            if needed_height <= max_panel_height:
                break
            scale = max(0.32, scale * 0.9)
            padding = max(7, round(14 * scale / 0.52))
            line_height = max(13, round(22 * scale / 0.52))
            title_height = max(12, round(22 * scale / 0.52))
            wrapped = cls._wrap_lines(lines, content_width, scale)

        max_lines = max(
            0,
            (max_panel_height - padding * 2 - title_height - 6) // line_height,
        )
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            if wrapped:
                wrapped[-1] = (cls._fit_text("...", content_width, scale), wrapped[-1][1])
        panel_height = min(
            max_panel_height,
            padding * 2 + title_height + 6 + len(wrapped) * line_height,
        )
        return panel_width, panel_height, scale, padding, line_height, wrapped

    @staticmethod
    def _landmark_collision_regions(
        landmarks: list[Any],
        connections: tuple[tuple[int, int], ...],
        transform: ViewportTransform,
        radius: int = 5,
    ) -> list[PixelRect]:
        points = [transform.point(item.x, item.y) for item in landmarks]
        regions = [
            PixelRect(x - radius, y - radius, radius * 2 + 1, radius * 2 + 1)
            for x, y in points
        ]
        for start, end in connections:
            if start >= len(points) or end >= len(points):
                continue
            x1, y1 = points[start]
            x2, y2 = points[end]
            left, top = min(x1, x2) - radius, min(y1, y2) - radius
            right, bottom = max(x1, x2) + radius, max(y1, y2) + radius
            regions.append(PixelRect(left, top, right - left + 1, bottom - top + 1))
        return regions

    @staticmethod
    def _fit_to_window(
        frame: np.ndarray,
        window_name: str,
        fallback_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        try:
            _, _, target_width, target_height = cv2.getWindowImageRect(window_name)
        except cv2.error:
            target_width, target_height = fallback_size or (0, 0)
        if target_width < 64 or target_height < 64 or target_width > 16384 or target_height > 16384:
            if fallback_size is None:
                return frame
            target_width, target_height = fallback_size

        source_height, source_width = frame.shape[:2]
        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, round(source_width * scale))
        resized_height = max(target_height, round(source_height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA,
        )
        crop_x = (resized_width - target_width) // 2
        crop_y = (resized_height - target_height) // 2
        return resized[crop_y : crop_y + target_height, crop_x : crop_x + target_width]

    @classmethod
    def _draw_connections(
        cls,
        frame: Any,
        landmarks: list[Any],
        connections: tuple[tuple[int, int], ...],
        color: tuple[int, int, int],
        thickness: int = 2,
        transform: ViewportTransform | None = None,
    ) -> None:
        for start_index, end_index in connections:
            if end_index >= len(landmarks):
                continue
            start = landmarks[start_index]
            end = landmarks[end_index]
            if min(start.visibility, end.visibility) < 0.35:
                continue
            cv2.line(
                frame,
                cls._pixel(frame, start, transform),
                cls._pixel(frame, end, transform),
                color,
                thickness,
                cv2.LINE_AA,
            )
        for landmark in landmarks:
            if landmark.visibility >= 0.35:
                cv2.circle(
                    frame,
                    cls._pixel(frame, landmark, transform),
                    3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

    @classmethod
    def _draw_contour(
        cls,
        frame: Any,
        landmarks: list[Any],
        connections: tuple[tuple[int, int], ...],
        color: tuple[int, int, int],
        thickness: int = 1,
        transform: ViewportTransform | None = None,
    ) -> None:
        for start_index, end_index in connections:
            if start_index >= len(landmarks) or end_index >= len(landmarks):
                continue
            cv2.line(
                frame,
                cls._pixel(frame, landmarks[start_index], transform),
                cls._pixel(frame, landmarks[end_index], transform),
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_landmarks(
        self,
        frame: Any,
        results: Any,
        draw_hand_labels: bool = True,
        transform: ViewportTransform | None = None,
    ) -> None:
        height, width = frame.shape[:2]
        if transform is None:
            transform = ViewportTransform.cover(width, height, width, height)

        panel_obstacles: list[PixelRect] = []
        label_obstacles: list[PixelRect] = []
        self._hand_label_rects: list[PixelRect] = []
        pose_sets = getattr(results, "poses_landmarks", None) or (
            [results.pose_landmarks] if results.pose_landmarks else []
        )
        face_sets = getattr(results, "faces_landmarks", None) or (
            [results.face_landmarks] if results.face_landmarks else []
        )
        body_bounds = [transform.bounds(pose, padding=8) for pose in pose_sets]
        face_bounds = [transform.bounds(face, padding=6) for face in face_sets]
        hand_bounds = {
            side: transform.bounds(landmarks, padding=6)
            for side, landmarks in results.hand_landmarks.items()
        }
        panel_obstacles.extend(
            bounds
            for bounds in (*body_bounds, *face_bounds, *hand_bounds.values())
            if bounds is not None
        )
        for pose_landmarks in pose_sets:
            label_obstacles.extend(
                self._landmark_collision_regions(
                    pose_landmarks, POSE_CONNECTIONS, transform, radius=6
                )
            )
        for face_landmarks in face_sets:
            bounds = transform.bounds(face_landmarks, padding=6)
            if bounds is not None:
                label_obstacles.append(bounds)
            label_obstacles.extend(
                self._landmark_collision_regions(
                    face_landmarks,
                    FACE_OVAL_CONNECTIONS,
                    transform,
                    radius=7,
                )
            )
        for side, landmarks in results.hand_landmarks.items():
            label_obstacles.extend(
                self._landmark_collision_regions(
                    landmarks, HAND_CONNECTIONS, transform, radius=6
                )
            )

        self._overlay_obstacles = panel_obstacles
        for pose_landmarks in pose_sets:
            self._draw_connections(
                frame,
                pose_landmarks,
                POSE_CONNECTIONS,
                (60, 220, 120),
                transform=transform,
            )
        palm_overlay = None
        if results.hand_landmarks:
            if (
                self._palm_overlay_buffer is None
                or self._palm_overlay_buffer.shape != frame.shape
            ):
                self._palm_overlay_buffer = np.empty_like(frame)
            np.copyto(self._palm_overlay_buffer, frame)
            palm_overlay = self._palm_overlay_buffer
        hand_colors = {
            "left": (220, 115, 45),
            "right": (35, 145, 255),
        }
        pending_labels: list[tuple[str, list[Any], PixelRect]] = []
        for side, hand_landmarks in results.hand_landmarks.items():
            color = hand_colors.get(side, (190, 150, 70))
            if palm_overlay is not None and len(hand_landmarks) > max(PALM_BOUNDARY):
                palm = np.array(
                    [
                        self._pixel(frame, hand_landmarks[index], transform)
                        for index in PALM_BOUNDARY
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(palm_overlay, [palm], color, cv2.LINE_AA)
            self._draw_connections(
                frame,
                hand_landmarks,
                HAND_CONNECTIONS,
                color,
                transform=transform,
            )
            current_hand_bounds = hand_bounds.get(side)
            if draw_hand_labels and hand_landmarks and current_hand_bounds is not None:
                pending_labels.append((side, hand_landmarks, current_hand_bounds))
        if palm_overlay is not None:
            cv2.addWeighted(palm_overlay, 0.2, frame, 0.8, 0, frame)
        for side, landmarks, bounds in pending_labels:
            self._draw_hand_label(
                frame,
                side,
                landmarks,
                bounds,
                transform,
                label_obstacles,
                self._hand_label_rects,
            )
        for face_landmarks in face_sets:
            self._draw_face_box(frame, face_landmarks, transform)
            self._draw_contour(
                frame,
                face_landmarks,
                FACE_OVAL_CONNECTIONS,
                (255, 220, 50),
                thickness=2,
                transform=transform,
            )

    def _draw_hand_label(
        self,
        frame: Any,
        side: str,
        landmarks: list[Any],
        bounds: PixelRect,
        transform: ViewportTransform,
        obstacles: list[PixelRect],
        reserved_labels: list[PixelRect],
    ) -> None:
        text = f"{side.upper()} HAND"
        scale = self._responsive_scale(frame.shape[1], frame.shape[0])
        scale = max(0.3, min(0.56, scale * 0.95))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(text, font, scale, 1)
        padding_x = max(6, round(10 * scale / 0.45))
        padding_y = max(4, round(7 * scale / 0.45))
        label_width = text_size[0] + padding_x * 2
        label_height = text_size[1] + baseline + padding_y * 2
        engine = HudLayoutEngine(frame.shape[1], frame.shape[0])
        label_rect = engine.place_hand_label(
            bounds,
            label_width,
            label_height,
            tuple(obstacles),
            tuple(reserved_labels),
        )
        if label_rect is None:
            return

        color = (220, 115, 45) if side == "left" else (35, 145, 255)
        roi = frame[
            label_rect.y : label_rect.bottom,
            label_rect.x : label_rect.right,
        ]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (label_rect.width - 1, label_rect.height - 1), color, -1)
        cv2.addWeighted(overlay, 0.78, roi, 0.22, 0, roi)
        cv2.rectangle(
            frame,
            (label_rect.x, label_rect.y),
            (label_rect.right - 1, label_rect.bottom - 1),
            color,
            1,
        )
        text_color = (25, 35, 45) if self.light_theme else (255, 255, 255)
        text = self._fit_text(text, label_rect.width - padding_x * 2, scale)
        cv2.putText(
            frame,
            text,
            (label_rect.x + padding_x, label_rect.y + padding_y + text_size[1]),
            font,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
        reserved_labels.append(label_rect)

    @staticmethod
    def _draw_panel(
        frame: Any,
        rectangle: PixelRect,
        title: str,
        lines: list[tuple[str, tuple[int, int, int]]],
        light_theme: bool = False,
        font_scale: float = 0.4,
        padding: int = 10,
        line_height: int = 18,
        selected: bool = False,
        owner_hand: str | None = None,
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        if not rectangle.inside(frame_width, frame_height, 20):
            return

        x, y = rectangle.x, rectangle.y
        width, height = rectangle.width, rectangle.height
        region = frame[y : y + height, x : x + width]
        overlay = region.copy()
        panel_color = (235, 240, 242) if light_theme else (23, 29, 34)
        if selected:
            border_color = (35, 125, 225) if light_theme else (65, 205, 255)
        else:
            border_color = (105, 120, 125) if light_theme else (85, 103, 112)
        title_color = (25, 110, 92) if light_theme else (80, 220, 190)
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), panel_color, -1)
        cv2.addWeighted(overlay, 0.84, region, 0.16, 0, region)
        cv2.rectangle(
            frame,
            (x, y),
            (x + width - 1, y + height - 1),
            border_color,
            3 if selected else 1,
        )
        if selected and owner_hand:
            title = f"{title}  /  DRAGGING {owner_hand.upper()} HAND"
        title_scale = font_scale * 1.12
        text_width = max(1, width - padding * 2)
        title = VisionApp._fit_text(title, text_width, title_scale)
        title_baseline = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 1
        )[0][1]
        cv2.putText(
            frame,
            title,
            (x + padding, y + padding + title_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            title_color,
            1,
            cv2.LINE_AA,
        )
        first_baseline = y + padding * 2 + title_baseline + 2
        for index, (text, color) in enumerate(lines):
            text_y = first_baseline + index * line_height
            if text_y + int(line_height * 0.25) >= y + height - padding // 2:
                break
            text = VisionApp._fit_text(text, text_width, font_scale)
            cv2.putText(
                frame,
                text,
                (x + padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def _prepare_panel(
        cls,
        frame: Any,
        title: str,
        lines: list[tuple[str, tuple[int, int, int]]],
    ) -> tuple[PixelRect, float, int, int, list[tuple[str, tuple[int, int, int]]]]:
        height, width = frame.shape[:2]
        panel_width, panel_height, scale, padding, line_height, wrapped = cls._panel_content(
            width, height, title, lines
        )
        if panel_width <= 0 or panel_height <= 0:
            return PixelRect(20, 20, 0, 0), scale, padding, line_height, []
        return (
            PixelRect(20, 20, panel_width, panel_height),
            scale,
            padding,
            line_height,
            wrapped,
        )

    @staticmethod
    def _fit_text(text: str, max_width: int, scale: float) -> str:
        suffix = "..."
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_width:
            return text
        while text and cv2.getTextSize(
            text + suffix, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0] > max_width:
            text = text[:-1]
        return text + suffix if text else suffix

    def _draw_hud(self, frame: Any, results: TrackingResults) -> None:
        if not self.show_hud:
            return

        self._sample_resources(time.perf_counter())
        metrics = self.metrics
        text_color = (32, 38, 42) if self.light_theme else (245, 245, 245)
        muted_color = (80, 91, 96) if self.light_theme else (190, 200, 204)
        found_color = (35, 145, 65) if self.light_theme else (70, 220, 110)
        missing_color = (45, 55, 190) if self.light_theme else (75, 85, 245)
        frame_height, frame_width = frame.shape[:2]
        layout = HudLayoutEngine(frame_width, frame_height)
        obstacles = tuple(
            getattr(self, "_overlay_obstacles", ())
        ) + tuple(getattr(self, "_hand_label_rects", ()))
        reserved_panels: list[PixelRect] = []
        stats_panel: tuple[Any, ...] | None = None

        if self.show_stats:
            face_status = "FOUND" if metrics.face_detected else "LOST"
            body_status = "FOUND" if metrics.body_detected else "LOST"
            left_status = "FOUND" if metrics.left_hand_detected else "LOST"
            right_status = "FOUND" if metrics.right_hand_detected else "LOST"
            face_match_status = (
                f"{sum(track.name != 'UNKNOWN' for track in self.face_tracks)} PERSON"
                if self.config.face_recognition_enabled else "OFF"
            )
            stats_lines = [
                (f"MODE  {self.processing_mode.name.replace(' Mode', '').upper()}", text_color),
                (f"FPS  C {metrics.camera_fps:.0f}  D {metrics.detection_fps:.0f}  R {metrics.render_fps:.0f}", text_color),
                (f"INFER {metrics.inference_ms:.0f}ms  LAT {metrics.frame_latency_ms:.0f}ms", text_color),
                (f"CAMERA {metrics.actual_camera}  REQUEST {metrics.requested_camera}", muted_color),
                (f"INFERENCE {metrics.inference_resolution}", muted_color),
                (f"TASK FPS  FACE {metrics.face_detection_fps:.0f}  ID {metrics.face_recognition_fps:.0f}  HAND {metrics.hand_detection_fps:.0f}  POSE {metrics.pose_detection_fps:.0f}", muted_color),
                (f"CPU {metrics.cpu_percent:.0f}%  RAM {metrics.ram_mb:.0f}MB", text_color),
                (f"FRAMES {metrics.total_frames}  ACTIONS {metrics.total_actions}", text_color),
                (f"FACE {face_status}  BODY {body_status}", found_color if metrics.face_detected and metrics.body_detected else missing_color),
                (f"LEFT HAND {left_status}  RIGHT HAND {right_status}", found_color if metrics.left_hand_detected and metrics.right_hand_detected else missing_color),
                (f"FACE MATCH {face_match_status}  TELEGRAM {'CONNECTED' if self.telegram.connected else 'OFFLINE'}", text_color),
            ]
            if self.video_writer is not None:
                elapsed = max(0.0, time.monotonic() - (self._recording_started_at or time.monotonic()))
                minutes, seconds = divmod(int(elapsed), 60)
                width, height = self._recording_size or (0, 0)
                try:
                    size_mb = self._recording_path.stat().st_size / (1024 * 1024) if self._recording_path else 0.0
                except OSError:
                    size_mb = 0.0
                stats_lines.append((f"REC ● {minutes:02}:{seconds:02}  {width}x{height}  {self._recording_fps:.0f} FPS  {size_mb:.1f}MB", found_color))
            if frame_width < 500 or frame_height < 400:
                stats_lines = [
                    (f"MODE {self.processing_mode.name.replace(' Mode', '').upper()}", text_color),
                    (f"FPS C {metrics.camera_fps:.0f} D {metrics.detection_fps:.0f} R {metrics.render_fps:.0f}", text_color),
                    (f"FACE {face_status} BODY {body_status}", text_color),
                    (f"LEFT HAND {left_status} RIGHT HAND {right_status}", text_color),
                    (f"CPU {metrics.cpu_percent:.0f}% RAM {metrics.ram_mb:.0f}MB", text_color),
                    (f"MODELS {metrics.active_models}", muted_color),
                    (f"RESOLUTION {metrics.resolution}", muted_color),
                ]
            if self.debug_mode:
                if frame_width >= 500 and frame_height >= 400:
                    stats_lines.extend(
                        [
                            (f"MODELS {metrics.active_models}", muted_color),
                            (f"RESOLUTION {metrics.resolution}", muted_color),
                        ]
                    )
            title = "MOTION / DEBUG" if self.debug_mode else "MOTION STATS"
            base_rect, scale, padding, line_height, wrapped = self._prepare_panel(
                frame, title, stats_lines
            )
            stats_rect = layout.place_panel(
                base_rect.width,
                base_rect.height,
                obstacles,
                anchor="left",
            )
            stats_rect = PixelRect(*self.panel_layout.apply(
                "motion_stats", (stats_rect.x, stats_rect.y, stats_rect.width, stats_rect.height),
                frame_width, frame_height, margin=20,
            ))
            stats_panel = (stats_rect, title, wrapped, self.light_theme, scale, padding, line_height)
            if stats_rect.width and stats_rect.height:
                reserved_panels.append(stats_rect)

        history_lines = [
            (
                f"{event.name.replace('_', ' ').upper()}  "
                f"{(event.handedness or '').upper()}  {event.confidence:.0%}",
                text_color,
            )
            for event in reversed(self.recent_actions)
        ] or [("NO GESTURES DETECTED", muted_color)]
        history_title = "GESTURE HISTORY  /  REC" if self.video_writer else "GESTURE HISTORY"
        history_base, scale, padding, line_height, wrapped = self._prepare_panel(
            frame, history_title, history_lines[:6]
        )
        history_rect = layout.place_panel(
            history_base.width,
            history_base.height,
            obstacles,
            tuple(reserved_panels),
            anchor="right",
        )
        history_rect = PixelRect(*self.panel_layout.apply(
            "gesture_history", (history_rect.x, history_rect.y, history_rect.width, history_rect.height),
            frame_width, frame_height, margin=20,
        ))
        history_panel = (history_rect, history_title, wrapped, self.light_theme, scale, padding, line_height)
        panels = {"gesture_history": (history_rect.x, history_rect.y, history_rect.width, history_rect.height)}
        if stats_panel is not None:
            stats_rect = stats_panel[0]
            panels["motion_stats"] = (stats_rect.x, stats_rect.y, stats_rect.width, stats_rect.height)
        drag_state = self.hud_drag.update(results.hand_landmarks, panels, frame_width, frame_height)
        if stats_panel is not None:
            stats_rect = PixelRect(*self.panel_layout.apply(
                "motion_stats", panels["motion_stats"], frame_width, frame_height, margin=20,
            ))
            self._draw_panel(frame, stats_rect, *stats_panel[1:])
        history_rect = PixelRect(*self.panel_layout.apply(
            "gesture_history", panels["gesture_history"], frame_width, frame_height, margin=20,
        ))
        self._draw_panel(frame, history_rect, *history_panel[1:])
        for side, (cursor_x, cursor_y) in drag_state.cursors.items():
            cv2.circle(frame, (cursor_x, cursor_y), max(5, round(frame_width / 280)), (80, 240, 220), 2, cv2.LINE_AA)

    def _handle_key(
        self,
        key: int,
        window_name: str,
        frame: Any,
        landmarkers: MediaPipeTasks | None = None,
    ) -> bool:
        if self.window_manager.reset_shortcut(key):
            self._reset_window_geometry(window_name)
            return True
        normalized_key = key & 0xFF
        if normalized_key in (ord("q"), ord("Q")):
            return False
        if normalized_key in (ord("r"), ord("R")):
            if self.video_writer is None:
                self._start_recording(self._latest_frame if self._latest_frame is not None else frame)
            else:
                self._stop_recording()
        elif normalized_key in (ord("c"), ord("C")):
            self.history.clear()
            self.recent_actions.clear()
            self.action_counts.clear()
            self.metrics.total_frames = 0
            self.metrics.total_actions = 0
            self.recognizer.reset()
            self.status_message = "Action history cleared"
        elif normalized_key in (ord("h"), ord("H")):
            self.show_hud = not self.show_hud
        elif normalized_key in (ord("s"), ord("S")):
            self.show_stats = not self.show_stats
        elif normalized_key in (ord("t"), ord("T")):
            self.light_theme = not self.light_theme
        elif normalized_key in (ord("d"), ord("D")):
            self.debug_mode = not self.debug_mode
        elif normalized_key in (ord("v"), ord("V")):
            self.camera_index = (self.camera_index + 1) % 10
            self._camera_change_requested = True

        if key in (0x720000, 0xFFC0):
            mode_index = PROCESSING_MODES.index(self.processing_mode)
            self.processing_mode = PROCESSING_MODES[
                (mode_index + 1) % len(PROCESSING_MODES)
            ]
            self.metrics.current_mode = self.processing_mode.name
            if landmarkers is not None:
                landmarkers.set_mode(self.processing_mode)

        if key in (0x7A, 0xFFC8, 0xFF0C) or (key >> 16) & 0xFF == 0x7A:
            entering = not self.fullscreen
            if entering:
                self._was_maximized_before_fullscreen = (
                    self.window_manager.state(window_name) == "maximized"
                )
                self._save_window_state(window_name)
            self.fullscreen = entering
            try:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
                )
            except cv2.error:
                self.status_message = "The window manager could not change fullscreen state"
                self.fullscreen = False
            if not self.fullscreen:
                self._restore_window(window_name)
                if self._was_maximized_before_fullscreen:
                    self.window_manager.maximize(window_name)
                self._was_maximized_before_fullscreen = False
        return True

    def _window_work_areas(self, window_name: str) -> tuple[Any, ...]:
        return self.window_manager.client_work_areas(window_name)

    def _safe_window_rect(
        self,
        rect: WindowRect | tuple[int, int, int, int] | None,
        window_name: str,
    ) -> WindowRect:
        return normalize_window_rect(rect, self._window_work_areas(window_name))

    def _apply_window_rect(self, window_name: str, rect: WindowRect) -> None:
        self.window_manager.restore(window_name)
        rect = self._safe_window_rect(rect, window_name)
        try:
            cv2.resizeWindow(window_name, rect.width, rect.height)
            if not self.window_manager.move_to_client_origin(window_name, rect.x, rect.y):
                cv2.moveWindow(window_name, rect.x, rect.y)
        except cv2.error:
            pass
        self._restored_window_rect = (rect.x, rect.y, rect.width, rect.height)
        self._last_valid_viewport = (rect.width, rect.height)

    def _reset_window_geometry(self, window_name: str) -> None:
        if self.fullscreen:
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            except cv2.error:
                pass
            self.fullscreen = False
        rect = centered_window_rect(self._window_work_areas(window_name))
        self._apply_window_rect(window_name, rect)
        self.window_state.update(x=rect.x, y=rect.y, width=rect.width, height=rect.height,
                                 viewport_width=rect.width, viewport_height=rect.height)
        self.window_state.save()
        self.status_message = "Window reset to a safe centered size"

    def _keep_window_on_work_area(self, window_name: str) -> None:
        try:
            current = WindowRect(*(int(value) for value in cv2.getWindowImageRect(window_name)))
        except (cv2.error, TypeError, ValueError):
            return
        if current.width < 64 or current.height < 64:
            return
        safe = self._safe_window_rect(current, window_name)
        if safe != current:
            self._apply_window_rect(window_name, safe)

    def _restore_window(self, window_name: str) -> None:
        rect = self._restored_window_rect
        if rect is None:
            data = self.window_state.data
            try:
                rect = (int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]))
            except (KeyError, TypeError, ValueError):
                rect = None
        if rect is not None:
            self._apply_window_rect(window_name, self._safe_window_rect(rect, window_name))

    def _save_window_state(self, window_name: str) -> None:
        native_state = self.window_manager.state(window_name)
        can_save_geometry = not self.fullscreen and native_state not in ("minimized", "maximized")
        if can_save_geometry:
            try:
                geometry = cv2.getWindowImageRect(window_name)
                candidate = WindowRect(*(int(value) for value in geometry))
            except cv2.error:
                candidate = None
            if candidate is not None and candidate.width >= 64 and candidate.height >= 64:
                rect = self._safe_window_rect(candidate, window_name)
                if rect != candidate:
                    self._apply_window_rect(window_name, rect)
                else:
                    self._restored_window_rect = (rect.x, rect.y, rect.width, rect.height)
                    self._last_valid_viewport = (rect.width, rect.height)
                self.window_state.update(x=rect.x, y=rect.y, width=rect.width, height=rect.height,
                                         viewport_width=rect.width, viewport_height=rect.height)
        frame = self._latest_frame
        if frame is not None:
            self.window_state.update(frame_width=int(frame.shape[1]), frame_height=int(frame.shape[0]))
        self.window_state.update(
            camera_index=self.camera_index,
            camera_resolution=self.metrics.resolution,
            camera_mode=self.metrics.actual_camera,
            processing_mode=self.processing_mode.name,
            camera_request_resolution="Auto" if self.config.camera.resolution is None else f"{self.config.camera.resolution[0]}x{self.config.camera.resolution[1]}",
            camera_request_fps="Auto" if self.config.camera.fps is None else self.config.camera.fps,
        )
        self.window_state.save()

    def _open_camera(self) -> tuple[Any, CameraMode]:
        if (
            self.config.camera.resolution is None
            and self.config.camera.fps is None
            and self.window_state.data.get("camera_index") == self.camera_index
        ):
            saved_mode = str(self.window_state.data.get("camera_mode", ""))
            try:
                resolution_text, fps_text = saved_mode.split("@", 1)
                width, height = (int(value) for value in resolution_text.strip().split("x", 1))
                fps_value = float(fps_text.strip().split()[0])
                fps = min((30, 60, 120), key=lambda candidate: abs(candidate - fps_value))
                cached_request = CameraRequest((width, height), fps)
                camera, mode = open_camera(
                    self.camera_index, cached_request, cv2.VideoCapture,
                    prop_width=cv2.CAP_PROP_FRAME_WIDTH,
                    prop_height=cv2.CAP_PROP_FRAME_HEIGHT,
                    prop_fps=cv2.CAP_PROP_FPS,
                    prop_buffer=cv2.CAP_PROP_BUFFERSIZE,
                )
                if (mode.width, mode.height) == (width, height):
                    return camera, mode
                camera.release()
            except (ValueError, TypeError):
                pass
        return open_camera(
            self.camera_index,
            self.config.camera,
            cv2.VideoCapture,
            prop_width=cv2.CAP_PROP_FRAME_WIDTH,
            prop_height=cv2.CAP_PROP_FRAME_HEIGHT,
            prop_fps=cv2.CAP_PROP_FPS,
            prop_buffer=cv2.CAP_PROP_BUFFERSIZE,
        )

    def _start_recording(self, frame: Any) -> None:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            self.status_message = "Cannot start recording without a camera frame"
            return
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.mp4")
        path = self.recordings_dir / filename
        height, width = frame.shape[:2]
        recording_fps = self.metrics.camera_fps
        if not math.isfinite(recording_fps) or recording_fps < 1.0:
            recording_fps = self._camera_mode.fps if self._camera_mode else 30.0
        try:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), recording_fps, (width, height)
            )
        except (cv2.error, OSError):
            self.status_message = "Could not create a video file; check disk permissions and codec support"
            return
        if not writer.isOpened():
            writer.release()
            self.status_message = "Could not open video writer"
            return
        self.video_writer = writer
        self._recording_size = (width, height)
        self._recording_path = path
        self._recording_started_at = time.monotonic()
        self._recording_fps = recording_fps
        self.status_message = f"Recording to {path}"

    def _stop_recording(self) -> None:
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except (cv2.error, OSError):
                self.status_message = "Video writer failed while closing"
            self.video_writer = None
            self._recording_size = None
            path = self._recording_path
            self._recording_path = None
            self._recording_started_at = None
            try:
                file_size = path.stat().st_size if path is not None and path.is_file() else 0
            except OSError:
                file_size = 0
            if path is not None and file_size > 0:
                size_mb = file_size / (1024 * 1024)
                self.status_message = f"Video saved ({size_mb:.1f} MB): {path}"
                destination = self.telegram.chat_id or self.config.telegram_chat_id
                if destination and self.telegram.enabled:
                    if size_mb > self.config.telegram_max_video_mb:
                        self.telegram.send_message(
                            destination,
                            f"Recording saved locally ({size_mb:.1f} MB), over the configured Telegram upload limit.",
                        )
                    else:
                        # Upload only after release finalized the MP4 container.
                        self.telegram.send_file(
                            destination,
                            "sendVideo",
                            str(path),
                            "Motion Vision recording",
                        )
            else:
                self.status_message = "Recording closed; no playable video data was written"

    def _update_face_tracks(
        self,
        results: TrackingResults,
        width: int,
        height: int,
        frame: np.ndarray | None = None,
        embed_face_fn: Any | None = None,
    ) -> list[FaceTrack]:
        faces = results.faces_landmarks or ([results.face_landmarks] if results.face_landmarks else [])
        candidates: list[tuple[tuple[int, int, int, int], list[float] | None]] = []
        recognition_started = time.perf_counter()
        encoded_any = False
        for landmarks in faces:
            box = face_box(landmarks, width, height)
            if box is not None:
                key = tuple(
                    value
                    for point in landmarks[::12]
                    for value in (float(point.x), float(point.y), float(point.z))
                )
                if key not in self._face_embedding_cache:
                    embedding = None
                    if (self.config.face_recognition_enabled and self.face_database.people
                            and frame is not None and embed_face_fn is not None):
                        try:
                            embedding = embed_face_fn(frame, landmarks)
                        except (RuntimeError, OSError, ValueError, cv2.error) as exc:
                            self.status_message = f"Local face recognition unavailable ({type(exc).__name__})"
                        encoded_any = True
                    self._face_embedding_cache[key] = embedding
                candidates.append((box, self._face_embedding_cache[key]))
        if len(self._face_embedding_cache) > 100:
            active_keys = {
                tuple(value for point in landmarks[::12] for value in (float(point.x), float(point.y), float(point.z)))
                for landmarks in faces
            }
            self._face_embedding_cache = {key: value for key, value in self._face_embedding_cache.items() if key in active_keys}
        if encoded_any:
            elapsed = max(time.perf_counter() - recognition_started, 1e-6)
            self.metrics.face_recognition_fps = self._ema(self.metrics.face_recognition_fps, 1.0 / elapsed)
        ids = self.face_tracker.assign([box for box, _ in candidates])
        updated: list[FaceTrack] = []
        for track_id, (box, embedding) in zip(ids, candidates):
            name, score = (
                self.face_database.match(embedding, self.config.face_match_threshold)
                if self.config.face_recognition_enabled and embedding else ("UNKNOWN", 0.0)
            )
            updated.append(FaceTrack(track_id, box, name, score))
        self.face_tracks = updated
        self.metrics.faces_found = len(updated)
        self.metrics.face_detected = bool(updated)
        return updated

    def _draw_face_matches(self, frame: np.ndarray, transform: ViewportTransform, tracks: list[FaceTrack]) -> None:
        for track in tracks:
            x, y, width, _ = track.box
            anchor_x, anchor_y = transform.point(x / max(transform.source_width, 1), y / max(transform.source_height, 1))
            label = f"PERSON {track.track_id}: {track.name}  MATCH {track.similarity:.0%}"
            cv2.putText(frame, label, (max(4, anchor_x), max(16, anchor_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.36, min(0.65, frame.shape[1] / 1800)),
                        (60, 235, 140) if track.name != "UNKNOWN" else (20, 210, 255), 1, cv2.LINE_AA)

    def _check_face_alerts(self, tracks: list[FaceTrack], frame: np.ndarray) -> None:
        if not self.watch_mode or not self.telegram.enabled:
            return
        now = time.monotonic()
        for track in tracks:
            if track.name == "UNKNOWN" or track.similarity < self.config.face_match_threshold:
                continue
            previous = self._last_face_alert.get(track.name, float("-inf"))
            if now - previous < self.config.face_alert_cooldown:
                continue
            self._last_face_alert[track.name] = now
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            snapshot = self.recordings_dir / f"face_event_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            try:
                cv2.imwrite(str(snapshot), frame)
                self._last_snapshot = snapshot
            except cv2.error:
                pass
            destination = self.telegram.chat_id or self.config.telegram_chat_id
            if destination:
                self.telegram.send_message(
                    destination,
                    f"Person detected:\nName: {track.name}\nMatch: {track.similarity:.0%}\nTime: {datetime.now().astimezone():%H:%M:%S}",
                )

    def _handle_bot_commands(self, landmarkers: MediaPipeTasks) -> None:
        while True:
            try:
                command = self.telegram.commands.get_nowait()
            except queue.Empty:
                return
            if not self.config.face_recognition_enabled and command.command == "/addface_photo":
                self.telegram.send_message(command.chat_id, "Face recognition is disabled in settings.")
                continue
            if command.command == "/status":
                reply = (f"Camera: {self.metrics.actual_camera}\nCamera FPS: {self.metrics.camera_fps:.1f}\n"
                         f"Detection FPS: {self.metrics.detection_fps:.1f}\nRender FPS: {self.metrics.render_fps:.1f}\n"
                         f"Mode: {self.processing_mode.name}\nRecording: {'ON' if self.video_writer else 'OFF'}\n"
                         f"Faces: {self.metrics.faces_found}\nWatch: {'ON' if self.watch_mode else 'OFF'}")
            elif command.command == "/photo":
                if self._latest_frame is None:
                    reply = "Camera is not producing frames."
                else:
                    self.recordings_dir.mkdir(parents=True, exist_ok=True)
                    path = self.recordings_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    if cv2.imwrite(str(path), self._latest_frame):
                        sent = self.telegram.send_file(command.chat_id, "sendPhoto", str(path), "Motion Vision snapshot")
                        reply = "Snapshot sent." if sent else "Snapshot saved locally, but Telegram delivery failed."
                    else:
                        reply = "Could not save snapshot."
            elif command.command == "/record":
                self._start_recording(self._latest_frame)
                reply = "Recording started." if self.video_writer else self.status_message
            elif command.command == "/stop":
                was_recording = self.video_writer is not None
                self._stop_recording()
                reply = "Recording stopped and saved." if was_recording else "No recording is active."
            elif command.command == "/history":
                reply = "\n".join(f"{event.name} {event.handedness or ''} {event.confidence:.0%}" for event in self.recent_actions) or "No recent gestures."
            elif command.command == "/settings":
                reply = f"Camera: {self.metrics.requested_camera}\nActual: {self.metrics.actual_camera}\nMode: {self.processing_mode.name}\nFace recognition: {'ON' if self.config.face_recognition_enabled else 'OFF'}\nFace threshold: {self.config.face_match_threshold:.2f}\nWatch: {'ON' if self.watch_mode else 'OFF'}"
            elif command.command == "/addface_photo":
                image = None
                try:
                    image = cv2.imdecode(np.frombuffer(command.photo or b"", dtype=np.uint8), cv2.IMREAD_COLOR)
                    embedding = landmarkers.encode_face_image(image) if image is not None else None
                    if embedding is None:
                        reply = "No usable face was found in that photo. Try a clear, front facing image."
                    else:
                        person_name = command.args[0] if command.args else f"Person {len(self.face_database.people) + 1}"
                        person_id = self.face_database.add(person_name, embedding)
                        self._face_embedding_cache.clear()
                        reply = f"Added {person_name} ({person_id}). Photo discarded; only the local descriptor was saved."
                except (cv2.error, OSError, ValueError, RuntimeError) as exc:
                    reply = f"Face photo could not be processed ({type(exc).__name__})."
                finally:
                    del image
            elif command.command == "/faces":
                reply = "\n".join(f"{person_id}: {name}" for person_id, (name, _) in self.face_database.people.items()) or "No enrolled faces."
            elif command.command == "/deleteface":
                if command.args and self.face_database.delete(command.args[0]):
                    reply = f"Deleted {command.args[0]}."
                else:
                    reply = "Usage: /deleteface ID (see /faces)."
            elif command.command == "/clearfaces":
                self.face_database.clear()
                reply = "Local face database cleared."
            elif command.command == "/watch":
                self.watch_mode = self.config.face_recognition_enabled and not self.watch_mode
                reply = f"Face match watch {'enabled' if self.watch_mode else 'disabled'}."
            else:
                reply = "Command unavailable. Send /help."
            self.telegram.send_message(command.chat_id, reply)

    def run(self) -> None:
        camera: Any | None = None
        window_name = "Motion Vision"
        self.telegram.start()
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            saved = self.window_state.data
            if self.reset_window_on_start:
                initial_rect = centered_window_rect(self._window_work_areas(window_name))
                self.reset_window_on_start = False
            else:
                try:
                    saved_rect = WindowRect(
                        int(saved.get("x", 80)), int(saved.get("y", 80)),
                        int(saved.get("width", 1280)), int(saved.get("height", 720)),
                    )
                except (TypeError, ValueError):
                    saved_rect = None
                initial_rect = self._safe_window_rect(saved_rect, window_name)
            self._apply_window_rect(window_name, initial_rect)
            width, height = initial_rect.width, initial_rect.height
            if self.window_state.data.get("x") != initial_rect.x or self.window_state.data.get("y") != initial_rect.y or self.window_state.data.get("width") != initial_rect.width or self.window_state.data.get("height") != initial_rect.height:
                self.window_state.update(x=initial_rect.x, y=initial_rect.y, width=initial_rect.width,
                                         height=initial_rect.height, viewport_width=initial_rect.width,
                                         viewport_height=initial_rect.height)
                self.window_state.save()

            previous_capture_at: float | None = None
            previous_render_at: float | None = None
            last_camera_attempt = 0.0
            camera_recovery = CameraReadRecovery()
            camera_switch_pending = False
            last_window_save = 0.0
            measured_capture_times: deque[float] = deque(maxlen=61)
            self.metrics.requested_camera = self._camera_request_label()
            with MediaPipeTasks(self.models_dir) as landmarkers:
                landmarkers.set_mode(self.processing_mode)
                self.metrics.current_mode = self.processing_mode.name
                while True:
                    native_state = self.window_manager.state(window_name)
                    if native_state == "minimized":
                        try:
                            key = cv2.waitKeyEx(100)
                        except cv2.error:
                            key = -1
                        if not self._handle_key(key, window_name, self._latest_frame, landmarkers):
                            break
                        self._handle_bot_commands(landmarkers)
                        continue
                    if native_state == "normal" and not self.fullscreen:
                        self._keep_window_on_work_area(window_name)
                    if camera is None or self._camera_change_requested:
                        camera_switch_pending = camera_switch_pending or self._camera_change_requested
                        switching_camera = camera_switch_pending
                        if camera is not None:
                            camera.release()
                            camera = None
                        now = time.monotonic()
                        if now >= last_camera_attempt:
                            last_camera_attempt = now + 2.0
                            self._camera_change_requested = False
                            try:
                                camera, self._camera_mode = self._open_camera()
                                width, height = self._camera_mode.width, self._camera_mode.height
                                self.metrics.resolution = f"{width}x{height}"
                                self.metrics.actual_camera = self._camera_mode.label
                                self.metrics.camera_fps = self._camera_mode.fps
                                self.status_message = f"Camera {self.camera_index}: {self.metrics.actual_camera}"
                                previous_capture_at = None
                                measured_capture_times.clear()
                                camera_recovery = CameraReadRecovery()
                                if switching_camera:
                                    landmarkers._force_refresh = True
                                    camera_switch_pending = False
                            except Exception as exc:
                                self.status_message = f"Camera {self.camera_index} unavailable ({type(exc).__name__}); retrying"
                        if camera is None:
                            height, width = self._last_valid_viewport[::-1] if self._last_valid_viewport else (720, 1280)
                            waiting = np.zeros((height, width, 3), dtype=np.uint8)
                            cv2.putText(waiting, self.status_message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                            try:
                                cv2.imshow(window_name, waiting)
                            except cv2.error:
                                pass
                            key = cv2.waitKeyEx(100)
                            if not self._handle_key(key, window_name, waiting, landmarkers):
                                break
                            self._handle_bot_commands(landmarkers)
                            continue

                    try:
                        success, frame = camera.read()
                    except Exception:
                        success, frame = False, None
                    captured_at = time.perf_counter()
                    if camera_recovery.observe(success, frame):
                        self.status_message = "Camera frame lost; attempting to reconnect"
                        if camera is not None:
                            camera.release()
                            camera = None
                            last_camera_attempt = 0.0
                        key = cv2.waitKeyEx(30)
                        if not self._handle_key(key, window_name, self._latest_frame, landmarkers):
                            break
                        continue
                    if not success or frame is None or getattr(frame, "ndim", 0) < 2 or min(frame.shape[:2]) < 2:
                        self.status_message = "Camera frame lost; waiting for recovery"
                        key = cv2.waitKeyEx(30)
                        if not self._handle_key(key, window_name, self._latest_frame, landmarkers):
                            break
                        continue
                    if previous_capture_at is not None:
                        measured_capture_times.append(captured_at)
                        if len(measured_capture_times) >= 3:
                            elapsed = measured_capture_times[-1] - measured_capture_times[0]
                            if elapsed > 0:
                                self.metrics.camera_fps = (len(measured_capture_times) - 1) / elapsed
                    else:
                        measured_capture_times.append(captured_at)
                    previous_capture_at = captured_at

                    cv2.flip(frame, 1, frame)
                    self._latest_frame = frame
                    self.metrics.resolution = f"{frame.shape[1]}x{frame.shape[0]}"
                    self.metrics.actual_camera = f"{self.metrics.resolution} @ {self.metrics.camera_fps:.0f} FPS"
                    self.metrics.inference_resolution = f"{max(1, round(frame.shape[1] * landmarkers.mode.inference_scale))}x{max(1, round(frame.shape[0] * landmarkers.mode.inference_scale))}"
                    if self.config.auto_record_on_start and self.video_writer is None and self.metrics.total_frames == 0:
                        self._start_recording(frame)
                    inference_started = time.perf_counter()
                    try:
                        results = landmarkers.detect(frame)
                    except (RuntimeError, ValueError, cv2.error) as exc:
                        self.status_message = f"MediaPipe frame skipped ({type(exc).__name__})"
                        results = TrackingResults(None, {}, None)
                    inference_finished = time.perf_counter()
                    inference_seconds = max(inference_finished - inference_started, 1e-6)
                    self.metrics.inference_ms = self._ema(self.metrics.inference_ms, inference_seconds * 1000.0)
                    self.metrics.detection_fps = self._ema(self.metrics.detection_fps, 1.0 / inference_seconds)
                    self.metrics.face_detection_fps = landmarkers.face_detection_fps
                    self.metrics.hand_detection_fps = landmarkers.hand_detection_fps
                    self.metrics.pose_detection_fps = landmarkers.pose_detection_fps
                    self.metrics.body_detected = bool(results.pose_landmarks)
                    self.metrics.left_hand_detected = any(side.startswith("left") for side in results.hand_landmarks)
                    self.metrics.right_hand_detected = any(side.startswith("right") for side in results.hand_landmarks)
                    tracks = self._update_face_tracks(
                        results, frame.shape[1], frame.shape[0], frame, landmarkers.embed_face
                    )
                    self._check_face_alerts(tracks, frame)

                    for event in self.recognizer.recognize(results):
                        self.history.add(event)
                        self.recent_actions.append(event)
                        self.action_counts[event.name] += 1
                        self.metrics.total_actions += 1
                        self.status_message = f"Detected: {event.name}"

                    self.metrics.total_frames += 1
                    try:
                        _, _, viewport_width, viewport_height = cv2.getWindowImageRect(window_name)
                    except cv2.error:
                        viewport_width, viewport_height = self._last_valid_viewport or (width, height)
                    if viewport_width > 0 and viewport_height > 0:
                        self._last_valid_viewport = (viewport_width, viewport_height)
                    else:
                        viewport_width, viewport_height = self._last_valid_viewport or (width, height)
                    display_frame = self._fit_to_window(frame, window_name, (viewport_width, viewport_height))
                    viewport = ViewportTransform.cover(frame.shape[1], frame.shape[0], display_frame.shape[1], display_frame.shape[0])
                    self._draw_landmarks(display_frame, results, draw_hand_labels=self.show_hud, transform=viewport)
                    self._draw_face_matches(display_frame, viewport, tracks)
                    self._draw_hud(display_frame, results)
                    if self.video_writer is not None:
                        if self._recording_size != (frame.shape[1], frame.shape[0]):
                            self.status_message = "Camera resolution changed; closing recording"
                            self._stop_recording()
                        elif self.video_writer is not None:
                            try:
                                self.video_writer.write(frame)
                            except (cv2.error, OSError):
                                self.status_message = "Video writer failed; recording stopped"
                                self._stop_recording()
                    try:
                        cv2.imshow(window_name, display_frame)
                    except cv2.error:
                        self.status_message = "Window display is temporarily unavailable"
                    self.metrics.frame_latency_ms = (time.perf_counter() - captured_at) * 1000.0

                    try:
                        key = cv2.waitKeyEx(1)
                    except cv2.error:
                        key = -1
                    rendered_at = time.perf_counter()
                    if previous_render_at is not None and rendered_at > previous_render_at:
                        self.metrics.render_fps = self._ema(self.metrics.render_fps, 1.0 / (rendered_at - previous_render_at))
                    previous_render_at = rendered_at
                    self._handle_bot_commands(landmarkers)
                    if rendered_at - last_window_save > 2.0:
                        self._save_window_state(window_name)
                        last_window_save = rendered_at
                    if not self._handle_key(key, window_name, display_frame, landmarkers):
                        break
        except (RuntimeError, OSError, ValueError, cv2.error) as exc:
            self.status_message = f"Motion Vision stopped safely ({type(exc).__name__}): {exc}"
            print(self.status_message)
        finally:
            self._stop_recording()
            if camera is not None:
                camera.release()
            self.telegram.stop()
            self._save_window_state(window_name)
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def main() -> None:
    initialize_frozen_runtime()
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Motion Vision camera gesture and tracking")
    parser.add_argument(
        "--version", action="version",
        version=f"Motion Vision {application_version()}",
    )
    parser.add_argument("--camera", type=int, default=None, help="OpenCV camera index")
    parser.add_argument("--resolution", default=None, help="Auto, 640x480, 1280x720, 1920x1080, 2560x1440, or 3840x2160")
    parser.add_argument("--fps", default=None, help="Auto, 30, 60, or 120")
    parser.add_argument("--list-cameras", action="store_true", help="Probe camera indices 0 through 9")
    parser.add_argument("--reset-window", action="store_true", help="Reset the application window to a centered safe size")
    args = parser.parse_args()
    if args.list_cameras:
        available = []
        for index in range(10):
            capture = cv2.VideoCapture(index)
            try:
                if capture.isOpened():
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        available.append((index, frame.shape[1], frame.shape[0]))
            finally:
                capture.release()
        print("Available cameras:", available or "none detected")
        return
    saved = WindowState(Path("motion_vision_config.json")).data
    camera_index = args.camera
    if camera_index is None:
        if "CAMERA_INDEX" not in os.environ and isinstance(saved.get("camera_index"), int):
            camera_index = max(0, saved["camera_index"])
        else:
            camera_index = config.camera_index
    resolution = config.camera.resolution if args.resolution is None else parse_resolution(args.resolution)
    fps = config.camera.fps if args.fps is None else parse_fps(args.fps)
    if args.resolution is None and "CAMERA_RESOLUTION" not in os.environ and saved.get("camera_request_resolution"):
        try:
            resolution = parse_resolution(str(saved["camera_request_resolution"]))
        except ValueError:
            pass
    if args.fps is None and "CAMERA_FPS" not in os.environ and saved.get("camera_request_fps") is not None:
        try:
            fps = parse_fps(str(saved["camera_request_fps"]))
        except ValueError:
            pass
    request = CameraRequest(
        resolution,
        fps,
    )
    config = replace(config, camera_index=camera_index, camera=request)
    VisionApp(camera_index=camera_index, config=config, reset_window=args.reset_window).run()


if __name__ == "__main__":
    main()
