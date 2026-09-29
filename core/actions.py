from __future__ import annotations

import time
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class TrackingResults:
    pose_landmarks: list[LandmarkPoint] | None
    hand_landmarks: dict[str, list[LandmarkPoint]]
    face_landmarks: list[LandmarkPoint] | None
    faces_landmarks: list[list[LandmarkPoint]] = field(default_factory=list)
    poses_landmarks: list[list[LandmarkPoint]] = field(default_factory=list)


@dataclass(frozen=True)
class ActionEvent:
    name: str
    timestamp: str
    details: dict[str, Any]
    confidence: float = 1.0
    handedness: str | None = None


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _landmark_value(landmark: Any, attribute: str, default: float) -> float:
    return _safe_float(getattr(landmark, attribute, None), default)


def _normalize_landmarks(landmarks: Any) -> list[LandmarkPoint]:
    if landmarks is None:
        return []

    normalized: list[LandmarkPoint] = []
    for landmark in landmarks:
        if landmark is None:
            continue
        normalized.append(
            LandmarkPoint(
                x=_landmark_value(landmark, "x", 0.0),
                y=_landmark_value(landmark, "y", 0.0),
                z=_landmark_value(landmark, "z", 0.0),
                visibility=_landmark_value(landmark, "visibility", 1.0),
                presence=_landmark_value(landmark, "presence", 1.0),
            )
        )
    return normalized


def tracking_results_from_tasks(
    pose_result: Any,
    hand_result: Any,
    face_result: Any,
) -> TrackingResults:
    poses = getattr(pose_result, "pose_landmarks", None)
    faces = getattr(face_result, "face_landmarks", None)
    detected_hands = getattr(hand_result, "hand_landmarks", None) or []
    handedness = getattr(hand_result, "handedness", None) or []

    hands: dict[str, list[LandmarkPoint]] = {}
    side_counts: Counter[str] = Counter()
    for index, landmarks in enumerate(detected_hands):
        categories = handedness[index] if index < len(handedness) else None
        category_name = (
            getattr(categories[0], "category_name", "")
            if categories and categories[0] is not None
            else ""
        )
        side = str(category_name or "").lower()
        if side not in {"left", "right"}:
            side = f"hand_{index + 1}"
        side_counts[side] += 1
        if side_counts[side] > 1:
            side = f"{side}_{side_counts[side]}"
        hands[side] = _normalize_landmarks(landmarks)

    normalized_poses = [_normalize_landmarks(item) for item in (poses or [])]
    normalized_faces = [_normalize_landmarks(item) for item in (faces or [])]
    pose_landmarks = normalized_poses[0] if normalized_poses else None
    face_landmarks = normalized_faces[0] if normalized_faces else None
    return TrackingResults(
        pose_landmarks=pose_landmarks or None,
        hand_landmarks=hands,
        face_landmarks=face_landmarks or None,
        faces_landmarks=normalized_faces,
        poses_landmarks=normalized_poses,
    )


class ActionRecognizer:
    """Recognize gestures from normalized MediaPipe Tasks landmarks."""

    ACTION_COOLDOWN_SECONDS = 1.0
    GESTURE_DEBOUNCE_SECONDS = 0.16
    GESTURE_CONFIRM_FRAMES = 5
    GESTURE_MIN_CONFIDENCE = 0.74
    HAND_RAISE_MIN_CONFIDENCE = 0.68
    SMOOTHING_ALPHA = 0.42
    TRACK_LOST_GRACE_SECONDS = 0.4
    TEMPORAL_VOTE_FRAMES = 7
    TEMPORAL_VOTE_MINIMUM = 3

    def __init__(self) -> None:
        self.hand_positions: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self.head_positions: deque[tuple[float, float]] = deque(maxlen=30)
        self.hand_was_raised = {"left": False, "right": False}
        self.last_emitted: dict[str, float] = {}
        self.pending_gestures: dict[str, tuple[str, int, float, float]] = {}
        self.active_gestures: dict[str, str | None] = {}
        self.pending_raises: dict[str, tuple[int, float, float]] = {}
        self.last_hand_seen: dict[str, float] = {}
        self.smoothed_landmarks: dict[str, list[LandmarkPoint]] = {}
        self.gesture_votes: dict[str, deque[tuple[str | None, float]]] = defaultdict(
            lambda: deque(maxlen=self.TEMPORAL_VOTE_FRAMES)
        )
        self.gesture_misses: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.hand_positions.clear()
        self.head_positions.clear()
        self.hand_was_raised = {"left": False, "right": False}
        self.last_emitted.clear()
        self.pending_gestures.clear()
        self.active_gestures.clear()
        self.pending_raises.clear()
        self.last_hand_seen.clear()
        self.smoothed_landmarks.clear()
        self.gesture_votes.clear()
        self.gesture_misses.clear()

    def _can_emit(self, key: str, now: float) -> bool:
        if now - self.last_emitted.get(key, float("-inf")) < self.ACTION_COOLDOWN_SECONDS:
            return False
        self.last_emitted[key] = now
        return True

    @staticmethod
    def _distance(first: Any, second: Any) -> float:
        dx = _landmark_value(first, "x", 0.0) - _landmark_value(second, "x", 0.0)
        dy = _landmark_value(first, "y", 0.0) - _landmark_value(second, "y", 0.0)
        dz = _landmark_value(first, "z", 0.0) - _landmark_value(second, "z", 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _smooth(self, key: str, landmarks: list[LandmarkPoint]) -> list[LandmarkPoint]:
        previous = self.smoothed_landmarks.get(key)
        if previous is None or len(previous) != len(landmarks):
            self.smoothed_landmarks[key] = list(landmarks)
            return landmarks

        alpha = self.SMOOTHING_ALPHA
        smoothed = [
            LandmarkPoint(
                x=alpha * current.x + (1.0 - alpha) * old.x,
                y=alpha * current.y + (1.0 - alpha) * old.y,
                z=alpha * current.z + (1.0 - alpha) * old.z,
                visibility=current.visibility,
                presence=current.presence,
            )
            for old, current in zip(previous, landmarks)
        ]
        self.smoothed_landmarks[key] = smoothed
        return smoothed

    def _reset_lost_hands(self, active_sides: set[str], now: float) -> None:
        for side, last_seen in tuple(self.last_hand_seen.items()):
            if side in active_sides or now - last_seen < self.TRACK_LOST_GRACE_SECONDS:
                continue
            self.last_hand_seen.pop(side, None)
            self.pending_gestures.pop(side, None)
            self.active_gestures.pop(side, None)
            self.hand_positions.pop(side, None)
            self.smoothed_landmarks.pop(f"hand:{side}", None)
            self.gesture_votes.pop(side, None)
            self.gesture_misses.pop(side, None)

    @classmethod
    def _classify_finger_gesture(cls, landmarks: list[Any]) -> tuple[str | None, float]:
        if len(landmarks) < 21:
            return None, 0.0

        wrist = landmarks[0]
        finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        palm_scale = max(
            cls._distance(wrist, landmarks[index]) for index in (5, 9, 13, 17)
        )
        if palm_scale < 1e-4:
            return None, 0.0
        finger_ratios = [
            cls._distance(wrist, landmarks[tip])
            / max(cls._distance(wrist, landmarks[pip]), 1e-6)
            for tip, pip in finger_pairs
        ]
        def joint_angle(first: int, joint: int, last: int) -> float:
            a = np.array((landmarks[first].x, landmarks[first].y, landmarks[first].z), dtype=float)
            b = np.array((landmarks[joint].x, landmarks[joint].y, landmarks[joint].z), dtype=float)
            c = np.array((landmarks[last].x, landmarks[last].y, landmarks[last].z), dtype=float)
            v1, v2 = a - b, c - b
            denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denominator < 1e-9:
                return 0.0
            cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
            return math.degrees(math.acos(cosine))

        finger_angles = [
            joint_angle(mcp, pip, tip)
            for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
        ]
        fingers_extended = [
            ratio > 1.12 and angle > 110.0
            for ratio, angle in zip(finger_ratios, finger_angles)
        ]

        thumb_reach = cls._distance(wrist, landmarks[4])
        thumb_base_reach = cls._distance(wrist, landmarks[2])
        thumb_ratio = thumb_reach / max(thumb_base_reach, 1e-6)
        thumb_extended = thumb_ratio > 1.15
        thumb_dx = _landmark_value(landmarks[4], "x", 0.0) - _landmark_value(landmarks[2], "x", 0.0)
        thumb_dy = _landmark_value(landmarks[4], "y", 0.0) - _landmark_value(landmarks[2], "y", 0.0)

        ratios = finger_ratios + [thumb_ratio]
        margins = [abs(ratio - threshold) for ratio, threshold in zip(ratios, (1.12,) * 4 + (1.15,))]
        angle_margin = min((max(0.0, angle - 110.0) if extended else max(0.0, 110.0 - angle))
                           for angle, extended in zip(finger_angles, fingers_extended))
        confidence = min(0.99, 0.65 + min(margins) * 0.25 + min(angle_margin / 180.0, 0.2))

        if all(fingers_extended) and thumb_extended:
            return "open_palm", confidence
        if fingers_extended == [True, True, False, False] and not thumb_extended:
            return "peace_sign", confidence
        if fingers_extended == [False, True, False, False] and not thumb_extended:
            return "middle_finger", confidence
        if fingers_extended == [True, False, False, False] and not thumb_extended:
            palm_scale = max(cls._distance(wrist, landmarks[5]), 1e-6)
            index_dx = (_landmark_value(landmarks[8], "x", 0.0) - _landmark_value(landmarks[6], "x", 0.0)) / palm_scale
            index_dy = (_landmark_value(landmarks[8], "y", 0.0) - _landmark_value(landmarks[6], "y", 0.0)) / palm_scale
            if abs(index_dy) >= abs(index_dx) * 0.75 and index_dy < -0.35:
                return "pointing_up", confidence
            if abs(index_dx) > abs(index_dy) * 1.1 and abs(index_dx) > 0.35:
                return ("pointing_right" if index_dx > 0 else "pointing_left"), confidence
        if not any(fingers_extended) and not thumb_extended:
            return "fist", confidence
        if not any(fingers_extended) and thumb_extended:
            thumb_vertical = thumb_dy
            if thumb_vertical < -0.12:
                return "thumbs_up", min(0.99, confidence + abs(thumb_vertical) * 0.2)
            if thumb_vertical > 0.12:
                return "thumbs_down", min(0.99, confidence + abs(thumb_vertical) * 0.2)
            if abs(thumb_dx) > 0.12:
                return ("thumb_side_right" if thumb_dx > 0 else "thumb_side_left"), confidence
        return None, 0.0

    def _vote_gesture(self, side: str, gesture: str | None, confidence: float) -> tuple[str | None, float]:
        votes = self.gesture_votes[side]
        if confidence < self.GESTURE_MIN_CONFIDENCE:
            gesture, confidence = None, 0.0
        votes.append((gesture, confidence))
        counts = Counter(name for name, _ in votes if name is not None)
        if counts:
            winner, count = counts.most_common(1)[0]
            if count >= self.TEMPORAL_VOTE_MINIMUM:
                self.gesture_misses[side] = 0
                mean_confidence = sum(score for name, score in votes if name == winner) / count
                return winner, mean_confidence
        self.gesture_misses[side] += 1
        active = self.active_gestures.get(side)
        if active and self.gesture_misses[side] <= 2:
            return active, confidence
        if self.gesture_misses[side] > 2:
            self.active_gestures[side] = None
        return None, 0.0

    def _debounced_gesture(
        self, side: str, gesture: str | None, confidence: float, now: float
    ) -> ActionEvent | None:
        if gesture is None or confidence < self.GESTURE_MIN_CONFIDENCE:
            self.pending_gestures.pop(side, None)
            if gesture is None:
                self.active_gestures[side] = None
            return None

        pending = self.pending_gestures.get(side)
        if pending is None or pending[0] != gesture:
            self.pending_gestures[side] = (gesture, 1, now, confidence)
            return None
        count = pending[1] + 1
        started_at = pending[2]
        confidence_sum = pending[3] + confidence
        self.pending_gestures[side] = (gesture, count, started_at, confidence_sum)
        if (
            count < self.GESTURE_CONFIRM_FRAMES
            or now - started_at < self.GESTURE_DEBOUNCE_SECONDS
        ):
            return None
        if self.active_gestures.get(side) == gesture:
            return None

        if not self._can_emit(f"{gesture}_{side}", now):
            return None
        self.active_gestures[side] = gesture
        return ActionEvent(
            name=gesture,
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            details={"side": side, "handedness": side},
            confidence=confidence_sum / count,
            handedness=side,
        )

    @staticmethod
    def _oscillation(points: deque[tuple[float, float]], now: float) -> tuple[int, float]:
        recent = [value for timestamp, value in points if now - timestamp <= 1.8]
        directions: list[int] = []
        for previous, current in zip(recent, recent[1:]):
            delta = current - previous
            if abs(delta) >= 0.025:
                direction = 1 if delta > 0 else -1
                if not directions or direction != directions[-1]:
                    directions.append(direction)
        span = max(recent) - min(recent) if recent else 0.0
        return max(0, len(directions) - 1), span

    def recognize(self, results: Any) -> list[ActionEvent]:
        now = time.monotonic()
        events: list[ActionEvent] = []

        raw_pose = getattr(results, "pose_landmarks", None)
        pose = self._smooth("pose", raw_pose) if raw_pose is not None else None
        if pose is not None and len(pose) > 24:
            for side, shoulder_index, wrist_index in (
                ("left", 11, 15),
                ("right", 12, 16),
            ):
                shoulder = pose[shoulder_index]
                wrist = pose[wrist_index]
                visible = min(
                    _landmark_value(shoulder, "visibility", 1.0),
                    _landmark_value(wrist, "visibility", 1.0),
                )
                wrist_y = _landmark_value(wrist, "y", 1.0)
                shoulder_y = _landmark_value(shoulder, "y", 1.0)
                vertical_gap = shoulder_y - wrist_y
                raise_confidence = visible * min(1.0, max(0.0, vertical_gap / 0.18))
                raised = (
                    visible > 0.55
                    and vertical_gap > 0.04
                    and raise_confidence >= self.HAND_RAISE_MIN_CONFIDENCE
                )
                if raised:
                    pending = self.pending_raises.get(side)
                    if pending is None:
                        count, started_at, confidence_sum = 1, now, raise_confidence
                    else:
                        count = pending[0] + 1
                        started_at = pending[1]
                        confidence_sum = pending[2] + raise_confidence
                    self.pending_raises[side] = (count, started_at, confidence_sum)
                    confirmed = (
                        count >= self.GESTURE_CONFIRM_FRAMES
                        and now - started_at >= self.GESTURE_DEBOUNCE_SECONDS
                    )
                    if confirmed and not self.hand_was_raised[side] and self._can_emit(f"raised_{side}", now):
                        events.append(
                            ActionEvent(
                                name="hand_raised",
                                timestamp=datetime.now().astimezone().isoformat(
                                    timespec="seconds"
                                ),
                                details={"side": side, "handedness": side},
                                confidence=min(0.99, confidence_sum / count),
                                handedness=side,
                            )
                        )
                        self.hand_was_raised[side] = True
                else:
                    self.pending_raises.pop(side, None)
                    self.hand_was_raised[side] = False

        hand_landmarks = getattr(results, "hand_landmarks", None) or {}
        self._reset_lost_hands(set(hand_landmarks), now)
        for side, raw_landmarks in hand_landmarks.items():
            if not raw_landmarks:
                continue
            self.last_hand_seen[side] = now
            landmarks = self._smooth(f"hand:{side}", raw_landmarks)
            wrist_x = _landmark_value(raw_landmarks[0], "x", 0.0)
            points = self.hand_positions[side]
            points.append((now, wrist_x))
            reversals, span = self._oscillation(points, now)
            key = f"wave_{side}"
            if reversals >= 2 and span >= 0.16 and self._can_emit(key, now):
                events.append(
                    ActionEvent(
                        name="hand_wave",
                        timestamp=datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        details={"side": side, "handedness": side},
                        confidence=min(0.99, 0.6 + span),
                        handedness=side,
                    )
                )

            gesture, confidence = self._classify_finger_gesture(landmarks)
            gesture, confidence = self._vote_gesture(side, gesture, confidence)
            gesture_event = self._debounced_gesture(side, gesture, confidence, now)
            if gesture_event is not None:
                events.append(gesture_event)

        raw_face_landmarks = getattr(results, "face_landmarks", None)
        face_landmarks = (
            self._smooth("face", raw_face_landmarks)
            if raw_face_landmarks is not None
            else None
        )
        if face_landmarks is not None and pose is not None and len(pose) > 12 and len(face_landmarks) > 1:
            left_shoulder, right_shoulder = pose[11], pose[12]
            left_x = _landmark_value(left_shoulder, "x", 0.0)
            right_x = _landmark_value(right_shoulder, "x", 0.0)
            shoulder_width = abs(right_x - left_x)
            if shoulder_width > 0.05:
                nose_x = _landmark_value(face_landmarks[1], "x", 0.0)
                shoulder_center = (left_x + right_x) / 2
                relative_x = (nose_x - shoulder_center) / shoulder_width
                self.head_positions.append((now, relative_x))
                reversals, span = self._oscillation(self.head_positions, now)
                if reversals >= 2 and span >= 0.35 and self._can_emit("head_shake", now):
                    events.append(
                        ActionEvent(
                            name="head_shake_left_right",
                            timestamp=datetime.now().astimezone().isoformat(
                                timespec="seconds"
                            ),
                            details={},
                            confidence=min(0.99, 0.6 + span * 0.4),
                        )
                    )

        return events
