from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PanelBox = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(frozen=True)
class DragFrameState:
    cursors: dict[str, tuple[int, int]]
    pinching: dict[str, bool]
    owner_hand: str | None
    panel_id: str | None
    released_panel: str | None = None


class HudPanelLayout:
    DEFAULT_PINCH_THRESHOLD = 0.055
    DEFAULT_CURSOR_SMOOTHING = 0.35

    def __init__(self, path: Path) -> None:
        self.path = path
        self.positions: dict[str, Point] = {}
        self.pinch_threshold = self.DEFAULT_PINCH_THRESHOLD
        self.cursor_smoothing = self.DEFAULT_CURSOR_SMOOTHING
        self.load()

    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number if math.isfinite(number) else default

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            content = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(content, dict):
            return

        raw_positions = content.get("panel_positions", {})
        if isinstance(raw_positions, dict):
            for panel_id, position in raw_positions.items():
                if not isinstance(panel_id, str) or not isinstance(position, dict):
                    continue
                x = self._finite_float(position.get("x"), 0.0)
                y = self._finite_float(position.get("y"), 0.0)
                self.positions[panel_id] = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

        self.pinch_threshold = min(
            0.2,
            max(
                0.01,
                self._finite_float(
                    content.get("pinch_threshold"), self.DEFAULT_PINCH_THRESHOLD
                ),
            ),
        )
        self.cursor_smoothing = min(
            1.0,
            max(
                0.05,
                self._finite_float(
                    content.get("cursor_smoothing"), self.DEFAULT_CURSOR_SMOOTHING
                ),
            ),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "panel_positions": {
                panel_id: {"x": x, "y": y}
                for panel_id, (x, y) in self.positions.items()
            },
            "pinch_threshold": self.pinch_threshold,
            "cursor_smoothing": self.cursor_smoothing,
        }
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    def apply(self, panel_id: str, box: PanelBox, width: int, height: int, margin: int = 10) -> PanelBox:
        position = self.positions.get(panel_id)
        if position is None:
            return self.clamp(box, width, height, margin)
        _, _, panel_width, panel_height = box
        max_x = max(margin, width - margin - panel_width)
        max_y = max(margin, height - margin - panel_height)
        x = margin + round(position[0] * (max_x - margin))
        y = margin + round(position[1] * (max_y - margin))
        return self.clamp((x, y, panel_width, panel_height), width, height, margin)

    def store(self, panel_id: str, box: PanelBox, width: int, height: int, margin: int = 10) -> None:
        x, y, panel_width, panel_height = self.clamp(box, width, height, margin)
        max_x = max(margin, width - margin - panel_width)
        max_y = max(margin, height - margin - panel_height)
        range_x = max_x - margin
        range_y = max_y - margin
        self.positions[panel_id] = (
            0.0 if range_x <= 0 else min(1.0, max(0.0, (x - margin) / range_x)),
            0.0 if range_y <= 0 else min(1.0, max(0.0, (y - margin) / range_y)),
        )

    @staticmethod
    def clamp(box: PanelBox, width: int, height: int, margin: int = 10) -> PanelBox:
        x, y, panel_width, panel_height = box
        panel_width = min(max(0, panel_width), max(0, width - margin * 2))
        panel_height = min(max(0, panel_height), max(0, height - margin * 2))
        max_x = max(margin, width - margin - panel_width)
        max_y = max(margin, height - margin - panel_height)
        x = min(max(margin, x), max_x)
        y = min(max(margin, y), max_y)
        return x, y, panel_width, panel_height

    def reset(self) -> None:
        self.positions.clear()
        self.save()


class GestureHudController:
    def __init__(self, layout: HudPanelLayout) -> None:
        self.layout = layout
        self.cursor_positions: dict[str, Point] = {}
        self.pinching: dict[str, bool] = {}
        self.owner_hand: str | None = None
        self.panel_id: str | None = None
        self.grab_offset: Point = (0.0, 0.0)

    def reset_drag(self) -> None:
        self.cursor_positions.clear()
        self.pinching.clear()
        self.owner_hand = None
        self.panel_id = None
        self.grab_offset = (0.0, 0.0)

    def update(
        self,
        hands: Mapping[str, list[Any]],
        panels: Mapping[str, PanelBox],
        viewport_width: int,
        viewport_height: int,
        margin: int = 10,
    ) -> DragFrameState:
        smoothed_normalized: dict[str, Point] = {}
        cursor_pixels: dict[str, tuple[int, int]] = {}
        pinch_states: dict[str, bool] = {}
        pinch_started: set[str] = set()
        alpha = self.layout.cursor_smoothing

        for side, landmarks in hands.items():
            if len(landmarks) <= 8:
                continue
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            point = (
                min(1.0, max(0.0, self._coordinate(index_tip, "x"))),
                min(1.0, max(0.0, self._coordinate(index_tip, "y"))),
            )
            previous = self.cursor_positions.get(side)
            smoothed = (
                point
                if previous is None
                else (
                    previous[0] + (point[0] - previous[0]) * alpha,
                    previous[1] + (point[1] - previous[1]) * alpha,
                )
            )
            self.cursor_positions[side] = smoothed
            smoothed_normalized[side] = smoothed
            cursor_pixels[side] = (
                min(viewport_width - 1, max(0, round(smoothed[0] * viewport_width))),
                min(viewport_height - 1, max(0, round(smoothed[1] * viewport_height))),
            )
            distance = math.hypot(
                self._coordinate(index_tip, "x") - self._coordinate(thumb_tip, "x"),
                self._coordinate(index_tip, "y") - self._coordinate(thumb_tip, "y"),
            )
            is_pinching = distance < self.layout.pinch_threshold
            pinch_states[side] = is_pinching
            if is_pinching and not self.pinching.get(side, False):
                pinch_started.add(side)

        released_panel = None
        if self.owner_hand is not None:
            if self.owner_hand not in pinch_states or not pinch_states[self.owner_hand]:
                released_panel = self.panel_id
                self.owner_hand = None
                self.panel_id = None
                self.grab_offset = (0.0, 0.0)
                self.layout.save()

        if self.owner_hand is None:
            for side in sorted(pinch_started):
                cursor = cursor_pixels[side]
                for panel_id, (x, y, panel_width, panel_height) in panels.items():
                    if x <= cursor[0] <= x + panel_width and y <= cursor[1] <= y + panel_height:
                        self.owner_hand = side
                        self.panel_id = panel_id
                        self.grab_offset = (cursor[0] - x, cursor[1] - y)
                        break
                if self.owner_hand is not None:
                    break

        if self.owner_hand is not None and self.panel_id in panels:
            side = self.owner_hand
            cursor_x, cursor_y = cursor_pixels[side]
            _, _, panel_width, panel_height = panels[self.panel_id]
            new_x = round(cursor_x - self.grab_offset[0])
            new_y = round(cursor_y - self.grab_offset[1])
            moved_box = self.layout.clamp(
                (new_x, new_y, panel_width, panel_height),
                viewport_width,
                viewport_height,
                margin,
            )
            self.layout.store(
                self.panel_id,
                moved_box,
                viewport_width,
                viewport_height,
                margin,
            )

        for side in tuple(self.pinching):
            if side not in pinch_states:
                self.pinching[side] = False
        self.pinching.update(pinch_states)
        return DragFrameState(
            cursors=cursor_pixels,
            pinching=pinch_states,
            owner_hand=self.owner_hand,
            panel_id=self.panel_id,
            released_panel=released_panel,
        )

    @staticmethod
    def _coordinate(landmark: Any, name: str) -> float:
        try:
            value = float(getattr(landmark, name))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0.0
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
