"""Local OpenCV Zoo SFace descriptor extraction using MediaPipe landmarks."""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np


MODEL_NAME = "face_recognition_sface_2021dec.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
MINIMUM_MODEL_BYTES = 10_000_000


class LocalFaceEmbedder:
    """Run SFace on locally aligned face crops; photos are never persisted."""

    def __init__(self, models_dir: Path) -> None:
        model_path = self._ensure_model(models_dir)
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")
        except (cv2.error, AttributeError) as exc:
            raise RuntimeError("OpenCV SFace support could not load its local model") from exc

    @staticmethod
    def _ensure_model(models_dir: Path) -> Path:
        path = models_dir / MODEL_NAME
        if path.is_file() and path.stat().st_size >= MINIMUM_MODEL_BYTES:
            return path
        models_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urlopen(MODEL_URL, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size < MINIMUM_MODEL_BYTES:
                raise RuntimeError("SFace model download was incomplete")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Could not download the local SFace model from OpenCV Zoo") from None
        return path

    @staticmethod
    def detection_row(landmarks: Any, image_width: int, image_height: int) -> np.ndarray | None:
        if len(landmarks) < 264 or image_width <= 0 or image_height <= 0:
            return None
        try:
            def pixel(index: int) -> tuple[float, float]:
                point = landmarks[index]
                return float(point.x) * image_width, float(point.y) * image_height

            right_eye_outer, right_eye_inner = pixel(33), pixel(133)
            left_eye_outer, left_eye_inner = pixel(362), pixel(263)
            right_eye = ((right_eye_outer[0] + right_eye_inner[0]) / 2, (right_eye_outer[1] + right_eye_inner[1]) / 2)
            left_eye = ((left_eye_outer[0] + left_eye_inner[0]) / 2, (left_eye_outer[1] + left_eye_inner[1]) / 2)
            nose, mouth_right, mouth_left = pixel(1), pixel(61), pixel(291)
            x_values = [float(point.x) * image_width for point in landmarks]
            y_values = [float(point.y) * image_height for point in landmarks]
            left = max(0.0, min(x_values))
            top = max(0.0, min(y_values))
            right = min(float(image_width), max(x_values))
            bottom = min(float(image_height), max(y_values))
            values = [left, top, right - left, bottom - top,
                      *right_eye, *left_eye, *nose, *mouth_right, *mouth_left, 1.0]
            if right <= left or bottom <= top or not all(math.isfinite(value) for value in values):
                return None
            return np.asarray(values, dtype=np.float32).reshape(1, 15)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None

    def embed(self, image_bgr: np.ndarray, landmarks: Any) -> list[float] | None:
        row = self.detection_row(landmarks, image_bgr.shape[1], image_bgr.shape[0])
        if row is None:
            return None
        try:
            aligned = self.recognizer.alignCrop(image_bgr, row)
            feature = self.recognizer.feature(aligned)
        except cv2.error:
            return None
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not math.isfinite(norm) or norm < 1e-12:
            return None
        return (vector / norm).astype(float).tolist()
