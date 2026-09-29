"""Local face database, matching, tracking, and a geometry-only helper.

Production identity descriptors are extracted by the OpenCV SFace backend in
``sface.py``. ``face_embedding`` remains a non-identity geometric utility.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from actions import LandmarkPoint


@dataclass(frozen=True)
class FaceTrack:
    track_id: int
    box: tuple[int, int, int, int]
    name: str
    similarity: float


def face_embedding(landmarks: Iterable[Any]) -> list[float] | None:
    """Build a roll and scale normalized descriptor from facial geometry."""
    points = list(landmarks)
    if len(points) < 264:
        return None
    try:
        def xy(index: int) -> tuple[float, float]:
            point = points[index]
            return float(point.x), float(point.y)

        left_eye, right_eye = xy(33), xy(263)
        eye_width = math.dist(left_eye, right_eye)
        if not math.isfinite(eye_width) or eye_width < 1e-5:
            return None
        center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
        angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        cosine, sine = math.cos(angle), math.sin(angle)
        indices = (10, 152, 234, 127, 162, 21, 54, 103, 67, 109, 33, 133, 362, 263,
                   1, 4, 6, 168, 197, 5, 98, 327, 61, 291, 0, 17, 13, 14, 78, 308,
                   50, 280, 116, 345, 123, 352, 205, 425)
        values: list[float] = []
        for index in indices:
            point = points[index]
            dx = (float(point.x) - center[0]) / eye_width
            dy = (float(point.y) - center[1]) / eye_width
            values.extend((dx * cosine + dy * sine, -dx * sine + dy * cosine))
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm < 1e-9:
            return None
        return [value / norm for value in values]
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def cosine_similarity(first: Iterable[float], second: Iterable[float]) -> float:
    a, b = list(first), list(second)
    if not a or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return -1.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class FaceDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.people: dict[str, tuple[str, list[float]]] = {}
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            rows = payload.get("people", []) if isinstance(payload, dict) else []
            for row in rows:
                person_id, name = str(row["id"]), str(row["name"])
                embedding = [float(value) for value in row["embedding"]]
                if embedding and all(math.isfinite(value) for value in embedding):
                    self.people[person_id] = (name, embedding)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.people = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        rows = [{"id": person_id, "name": name, "embedding": embedding}
                for person_id, (name, embedding) in sorted(self.people.items())]
        temp.write_text(json.dumps({"people": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def add(self, name: str, embedding: Iterable[float], person_id: str | None = None) -> str:
        vector = [float(value) for value in embedding]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("A finite, non-empty face embedding is required")
        if person_id is None:
            sequence = 1
            for existing_id in self.people:
                prefix, separator, suffix = existing_id.rpartition("-")
                if prefix == "person" and separator and suffix.isdigit():
                    sequence = max(sequence, int(suffix) + 1)
            while f"person-{sequence:04d}" in self.people:
                sequence += 1
            person_id = f"person-{sequence:04d}"
        self.people[person_id] = (name.strip() or person_id, vector)
        self.save()
        return person_id

    def delete(self, person_id: str) -> bool:
        existed = self.people.pop(person_id, None) is not None
        if existed:
            self.save()
        return existed

    def clear(self) -> None:
        self.people.clear()
        self.save()

    def match(self, embedding: Iterable[float], threshold: float = 0.6) -> tuple[str, float]:
        best_name, best_score = "UNKNOWN", -1.0
        for name, known in self.people.values():
            score = cosine_similarity(embedding, known)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < threshold:
            return "UNKNOWN", max(0.0, best_score)
        return best_name, max(0.0, best_score)


class FaceTracker:
    """Greedy IoU tracker for short gaps between detections."""
    def __init__(self, max_missing: int = 8, iou_threshold: float = 0.1) -> None:
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: dict[int, tuple[tuple[int, int, int, int], int]] = {}

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def assign(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        candidates = sorted(((self._iou(old_box, box), track_id, index)
                             for track_id, (old_box, _) in self.tracks.items()
                             for index, box in enumerate(boxes)), reverse=True)
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        assigned = [0] * len(boxes)
        for overlap, track_id, index in candidates:
            if overlap < self.iou_threshold:
                break
            if track_id in used_tracks or index in used_boxes:
                continue
            assigned[index] = track_id
            used_tracks.add(track_id)
            used_boxes.add(index)
        for index, box in enumerate(boxes):
            if not assigned[index]:
                assigned[index] = self.next_id
                self.next_id += 1
            self.tracks[assigned[index]] = (box, 0)
        for track_id, (box, missing) in tuple(self.tracks.items()):
            if track_id not in assigned:
                if missing + 1 > self.max_missing:
                    del self.tracks[track_id]
                else:
                    self.tracks[track_id] = (box, missing + 1)
        return assigned


def face_box(landmarks: Iterable[Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    points = list(landmarks)
    if not points or width <= 0 or height <= 0:
        return None
    xs = [float(point.x) * width for point in points]
    ys = [float(point.y) * height for point in points]
    left, top = max(0, int(min(xs))), max(0, int(min(ys)))
    right, bottom = min(width, int(max(xs)) + 1), min(height, int(max(ys)) + 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top
