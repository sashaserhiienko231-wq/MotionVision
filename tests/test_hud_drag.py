from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hud_drag import GestureHudController, HudPanelLayout


def make_hand(index_tip: tuple[float, float], thumb_tip: tuple[float, float]):
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    landmarks[8] = SimpleNamespace(x=index_tip[0], y=index_tip[1])
    landmarks[4] = SimpleNamespace(x=thumb_tip[0], y=thumb_tip[1])
    return landmarks


class HudDragControllerTests(unittest.TestCase):
    def make_layout(self, directory: str) -> HudPanelLayout:
        return HudPanelLayout(Path(directory) / "motion_vision_ui.json")

    def test_pinch_detector_open_and_pinch_states(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.cursor_smoothing = 1.0
            controller = GestureHudController(layout)
            panels = {"motion_stats": (100, 100, 200, 120)}

            open_state = controller.update(
                {"left": make_hand((0.2, 0.2), (0.35, 0.2))}, panels, 1000, 800
            )
            pinch_state = controller.update(
                {"left": make_hand((0.2, 0.2), (0.22, 0.2))}, panels, 1000, 800
            )

            self.assertFalse(open_state.pinching["left"])
            self.assertTrue(pinch_state.pinching["left"])
            self.assertEqual(pinch_state.owner_hand, "left")

    def test_drag_start_move_release_and_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.cursor_smoothing = 1.0
            controller = GestureHudController(layout)
            panels = {"motion_stats": (100, 100, 200, 120)}

            started = controller.update(
                {"left": make_hand((0.2, 0.2), (0.22, 0.2))}, panels, 1000, 800
            )
            self.assertEqual(started.panel_id, "motion_stats")
            self.assertEqual(started.owner_hand, "left")

            moved = controller.update(
                {"left": make_hand((0.5, 0.5), (0.51, 0.5))}, panels, 1000, 800
            )
            self.assertEqual(moved.owner_hand, "left")
            self.assertEqual(layout.apply("motion_stats", panels["motion_stats"], 1000, 800), (400, 340, 200, 120))

            released = controller.update(
                {"left": make_hand((0.5, 0.5), (0.7, 0.5))}, panels, 1000, 800
            )
            self.assertIsNone(released.owner_hand)
            self.assertEqual(released.released_panel, "motion_stats")
            self.assertTrue(layout.path.is_file())

    def test_drag_is_clamped_to_viewport_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.cursor_smoothing = 1.0
            controller = GestureHudController(layout)
            panels = {"gesture_history": (800, 650, 180, 120)}

            controller.update(
                {"right": make_hand((0.95, 0.95), (0.96, 0.95))},
                panels,
                1000,
                800,
            )
            controller.update(
                {"right": make_hand((1.0, 1.0), (0.99, 1.0))},
                panels,
                1000,
                800,
            )
            clamped = layout.apply("gesture_history", panels["gesture_history"], 1000, 800)

            self.assertEqual(clamped, (810, 670, 180, 120))

    def test_layout_save_load_and_relative_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.store("motion_stats", (400, 300, 200, 100), 1000, 800)
            layout.pinch_threshold = 0.07
            layout.cursor_smoothing = 0.6
            layout.save()

            restored = self.make_layout(directory)
            resized = restored.apply("motion_stats", (10, 10, 300, 150), 1600, 900)

            self.assertAlmostEqual(restored.pinch_threshold, 0.07)
            self.assertAlmostEqual(restored.cursor_smoothing, 0.6)
            self.assertAlmostEqual(
                (resized[0] - 10) / (1600 - 20 - resized[2]),
                layout.positions["motion_stats"][0],
                places=2,
            )
            self.assertAlmostEqual(
                (resized[1] - 10) / (900 - 20 - resized[3]),
                layout.positions["motion_stats"][1],
                places=2,
            )

    def test_only_one_hand_owns_drag_and_other_hand_can_grab_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.cursor_smoothing = 1.0
            controller = GestureHudController(layout)
            panels = {"motion_stats": (100, 100, 220, 140)}
            left_pinching = make_hand((0.2, 0.2), (0.21, 0.2))
            right_pinching = make_hand((0.2, 0.2), (0.21, 0.2))

            first = controller.update(
                {"right": right_pinching, "left": left_pinching},
                panels,
                1000,
                800,
            )
            self.assertEqual(first.owner_hand, "left")
            self.assertEqual(first.panel_id, "motion_stats")

            released = controller.update(
                {
                    "left": make_hand((0.2, 0.2), (0.5, 0.2)),
                    "right": make_hand((0.2, 0.2), (0.5, 0.2)),
                },
                panels,
                1000,
                800,
            )
            self.assertIsNone(released.owner_hand)

            right_grab = controller.update(
                {"right": right_pinching}, panels, 1000, 800
            )
            self.assertEqual(right_grab.owner_hand, "right")
            self.assertEqual(right_grab.panel_id, "motion_stats")

    def test_cursor_smoothing_is_per_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.cursor_smoothing = 0.25
            controller = GestureHudController(layout)
            panels = {}

            first = controller.update(
                {
                    "left": make_hand((0.2, 0.2), (0.4, 0.2)),
                    "right": make_hand((0.8, 0.8), (0.6, 0.8)),
                },
                panels,
                1000,
                800,
            )
            second = controller.update(
                {
                    "left": make_hand((0.6, 0.6), (0.8, 0.6)),
                    "right": make_hand((0.4, 0.4), (0.2, 0.4)),
                },
                panels,
                1000,
                800,
            )

            self.assertEqual(first.cursors["left"], (200, 160))
            self.assertEqual(first.cursors["right"], (800, 640))
            self.assertEqual(second.cursors["left"], (300, 240))
            self.assertEqual(second.cursors["right"], (700, 560))

    def test_reset_layout_clears_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = self.make_layout(directory)
            layout.store("motion_stats", (100, 100, 200, 100), 800, 600)
            layout.save()

            layout.reset()

            self.assertEqual(layout.positions, {})
            restored = self.make_layout(directory)
            self.assertEqual(restored.positions, {})


if __name__ == "__main__":
    unittest.main()
