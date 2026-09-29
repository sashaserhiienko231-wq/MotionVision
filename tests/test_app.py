from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from actions import LandmarkPoint, TrackingResults
from app import (
    PROCESSING_MODES,
    HudLayoutEngine,
    MediaPipeTasks,
    PixelRect,
    RuntimeMetrics,
    ViewportTransform,
    VisionApp,
)


class FakeLandmarker:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect_for_video(self, image, timestamp_ms):
        self.calls += 1
        return self.result


class MotionVisionUiTests(unittest.TestCase):
    def make_app(self) -> VisionApp:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        return VisionApp(
            history_path=root / "actions.json",
            recordings_dir=root / "recordings",
            models_dir=root / "models",
        )

    def test_cover_resize_fills_target_viewport(self):
        frame = np.full((480, 640, 3), (15, 30, 45), dtype=np.uint8)
        with patch("app.cv2.getWindowImageRect", return_value=(0, 0, 800, 450)):
            display = VisionApp._fit_to_window(frame, "test-window")

        self.assertEqual(display.shape, (450, 800, 3))
        self.assertTrue(np.all(display == (15, 30, 45)))

    def test_window_resize_and_fullscreen_dimensions_are_refit_each_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        viewport_sizes = ((1280, 720), (960, 540), (1920, 1080), (720, 1280))

        for width, height in viewport_sizes:
            with self.subTest(viewport=(width, height)):
                with patch(
                    "app.cv2.getWindowImageRect",
                    return_value=(0, 0, width, height),
                ):
                    display = VisionApp._fit_to_window(frame, "test-window")
                self.assertEqual(display.shape[:2], (height, width))

    def test_viewport_transform_maps_landmarks_after_cover_crop(self):
        for width, height in ((320, 240), (800, 450), (1920, 1080), (1080, 1920)):
            with self.subTest(viewport=(width, height)):
                transform = ViewportTransform.cover(640, 480, width, height)
                center = transform.point(0.5, 0.5)
                self.assertLessEqual(abs(center[0] - width // 2), 1)
                self.assertLessEqual(abs(center[1] - height // 2), 1)
                self.assertGreaterEqual(center[0], 0)
                self.assertLess(center[0], width)
                self.assertGreaterEqual(center[1], 0)
                self.assertLess(center[1], height)

        transform = ViewportTransform.cover(640, 480, 800, 450)
        self.assertLess(transform.point(0.5, 0.0)[1], 0)
        self.assertIsNone(
            transform.bounds([LandmarkPoint(0.5, 0.0)], padding=2)
        )

    def test_panel_placement_avoids_tracks_and_other_panels(self):
        engine = HudLayoutEngine(800, 600, margin=20)
        subject = PixelRect(20, 20, 260, 520)
        first = engine.place_panel(230, 150, obstacles=(subject,), anchor="left")
        second = engine.place_panel(
            230, 150, obstacles=(subject,), reserved=(first,), anchor="right"
        )

        self.assertTrue(first.inside(800, 600, margin=20))
        self.assertTrue(second.inside(800, 600, margin=20))
        self.assertEqual(first.intersection_area(subject), 0)
        self.assertEqual(second.intersection_area(subject), 0)
        self.assertEqual(first.intersection_area(second), 0)

    def test_hand_labels_are_above_hand_bounds_and_collision_free(self):
        app = self.make_app()
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        hands = {
            "left": [LandmarkPoint(0.5, 0.48) for _ in range(21)],
            "right": [LandmarkPoint(0.5, 0.48) for _ in range(21)],
        }
        transform = ViewportTransform.cover(800, 600, 800, 600)

        app._draw_landmarks(frame, TrackingResults(None, hands, None), transform=transform)

        self.assertEqual(len(app._hand_label_rects), 2)
        labels = app._hand_label_rects
        self.assertTrue(all(label.inside(800, 600, margin=20) for label in labels))
        self.assertEqual(labels[0].intersection_area(labels[1]), 0)
        for side, label in zip(("left", "right"), labels):
            bounds = transform.bounds(hands[side], padding=6)
            self.assertLessEqual(label.bottom, bounds.y - 6)

    def test_hand_label_moves_around_obstacle_but_stays_above_hand(self):
        engine = HudLayoutEngine(800, 600, margin=20)
        hand_bounds = PixelRect(330, 260, 140, 200)
        obstacle = PixelRect(335, 222, 130, 30)
        label = engine.place_hand_label(
            hand_bounds,
            label_width=110,
            label_height=28,
            obstacles=(obstacle,),
        )

        self.assertIsNotNone(label)
        self.assertTrue(label.inside(800, 600, margin=20))
        self.assertLessEqual(label.bottom, hand_bounds.y - 6)
        self.assertEqual(label.intersection_area(obstacle), 0)

    def test_face_box_uses_landmark_extrema_in_viewport_coordinates(self):
        frame = np.zeros((450, 800, 3), dtype=np.uint8)
        landmarks = [
            LandmarkPoint(0.25, 0.3),
            LandmarkPoint(0.75, 0.8),
            LandmarkPoint(0.5, 0.5),
        ]
        transform = ViewportTransform.cover(640, 480, 800, 450)

        bounds = VisionApp._draw_face_box(frame, landmarks, transform)

        self.assertEqual(bounds, transform.bounds(landmarks, padding=2))

    def test_panel_content_wraps_inside_small_viewports(self):
        lines = [("A LONG GESTURE NAME WITH HAND AND CONFIDENCE 99%", (255, 255, 255))]
        for width, height in ((160, 120), (320, 240), (1024, 768)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            rectangle, scale, padding, line_height, wrapped = VisionApp._prepare_panel(
                frame, "GESTURE HISTORY", lines
            )
            self.assertLessEqual(rectangle.width, width - 40)
            self.assertLessEqual(rectangle.height, height - 40)
            self.assertGreater(scale, 0)
            self.assertGreater(padding, 0)
            self.assertGreater(line_height, 0)
            self.assertTrue(wrapped)

    def test_stats_and_debug_rows_fit_adaptive_panel_layout(self):
        app = self.make_app()
        app.debug_mode = True
        app.metrics = RuntimeMetrics(
            camera_fps=29.8,
            detection_fps=24.1,
            render_fps=23.7,
            inference_ms=41.5,
            frame_latency_ms=48.0,
            total_frames=120,
            total_actions=7,
            face_detected=True,
            body_detected=True,
            left_hand_detected=True,
            right_hand_detected=False,
            cpu_percent=32.0,
            ram_mb=512.0,
            resolution="640x480",
        )
        results = TrackingResults(
            pose_landmarks=None,
            hand_landmarks={"left": [LandmarkPoint(0.3, 0.4)]},
            face_landmarks=None,
        )

        for width, height in ((320, 240), (800, 600), (1920, 1080)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            with patch.object(app, "_draw_panel", wraps=app._draw_panel) as draw_panel:
                app._draw_hud(frame, results)

            self.assertGreaterEqual(draw_panel.call_count, 2)
            stats_call, history_call = draw_panel.call_args_list[:2]
            stats_rect = stats_call.args[1]
            history_rect = history_call.args[1]
            self.assertTrue(stats_rect.inside(width, height, margin=20))
            self.assertTrue(history_rect.inside(width, height, margin=20))
            self.assertIn("MOTION / DEBUG", stats_call.args[2])
            stat_lines = [text for text, _ in stats_call.args[3]]
            joined_stats = " ".join(stat_lines)
            self.assertIn("FACE FOUND BODY FOUND", joined_stats)
            self.assertIn("LEFT HAND FOUND RIGHT HAND LOST", joined_stats)
            self.assertTrue(any("CPU" in text for text in stat_lines))
            self.assertTrue(any("RAM" in text for text in stat_lines))
            self.assertTrue(any("MODELS" in text for text in stat_lines))
            self.assertTrue(any("RESOLUTION" in text for text in stat_lines))
            self.assertIn("GESTURE HISTORY", history_call.args[2])
            self.assertEqual(stats_rect.intersection_area(history_rect), 0)
            self.assertGreater(int(frame.sum()), 0)

    def test_hidden_hud_skips_panels_without_affecting_stats_state(self):
        app = self.make_app()
        app.show_hud = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(app, "_draw_panel") as draw_panel:
            app._draw_hud(frame, TrackingResults(None, {}, None))

        draw_panel.assert_not_called()
        self.assertTrue(app.show_stats)

    def test_keyboard_toggles_and_quit_are_testable_without_camera(self):
        app = self.make_app()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.assertTrue(app._handle_key(ord("h"), "test-window", frame))
        self.assertFalse(app.show_hud)
        self.assertTrue(app._handle_key(ord("s"), "test-window", frame))
        self.assertFalse(app.show_stats)
        self.assertTrue(app._handle_key(ord("t"), "test-window", frame))
        self.assertTrue(app.light_theme)
        self.assertTrue(app._handle_key(ord("d"), "test-window", frame))
        self.assertTrue(app.debug_mode)

        with patch("app.cv2.setWindowProperty") as set_property:
            self.assertTrue(app._handle_key(0x7A, "test-window", frame))
        self.assertTrue(app.fullscreen)
        set_property.assert_called_once()
        self.assertFalse(app._handle_key(ord("q"), "test-window", frame))

    def test_clear_key_resets_event_counters(self):
        app = self.make_app()
        app.metrics.total_frames = 123
        app.metrics.total_actions = 5
        app.action_counts["fist"] = 5

        self.assertTrue(app._handle_key(ord("c"), "test-window", np.zeros((8, 8, 3))))

        self.assertEqual(app.metrics.total_frames, 0)
        self.assertEqual(app.metrics.total_actions, 0)
        self.assertEqual(app.action_counts, {})

    def test_inference_scheduler_modes_preserve_hand_detection_every_frame(self):
        pose_landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
        hand_landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        face_landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(2)]
        test_results = SimpleNamespace(
            pose_landmarks=[pose_landmarks],
            hand_landmarks=[hand_landmarks],
            handedness=[[SimpleNamespace(category_name="Left")]],
            face_landmarks=[face_landmarks],
        )
        expected_calls = {
            "Quality Mode": (8, 8, 8),
            "Balanced Mode": (4, 3, 8),
            "Performance Mode": (2, 2, 8),
        }

        for mode in PROCESSING_MODES:
            with self.subTest(mode=mode.name):
                detector = MediaPipeTasks.__new__(MediaPipeTasks)
                detector.last_timestamp_ms = -1
                detector._rgb_buffer = None
                detector._inference_bgr_buffer = None
                detector._input_shape = None
                detector._frame_index = -1
                detector._force_refresh = True
                detector.mode = mode
                detector._cached_pose_landmarks = None
                detector._cached_face_landmarks = None
                detector.last_inference_ms = 0.0
                detector.pose = FakeLandmarker(test_results)
                detector.hands = FakeLandmarker(test_results)
                detector.face = FakeLandmarker(test_results)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                with patch("app.mp.Image", return_value=object()):
                    results = [detector.detect(frame) for _ in range(8)]

                pose_count, face_count, hand_count = expected_calls[mode.name]
                self.assertEqual(detector.pose.calls, pose_count)
                self.assertEqual(detector.face.calls, face_count)
                self.assertEqual(detector.hands.calls, hand_count)
                self.assertTrue(all(result.pose_landmarks for result in results))
                self.assertTrue(all(result.face_landmarks for result in results))
                self.assertTrue(all("left" in result.hand_landmarks for result in results))
                if mode.pose_interval > 1:
                    self.assertIs(results[0].pose_landmarks, results[1].pose_landmarks)
                if mode.face_interval > 1:
                    self.assertIs(results[0].face_landmarks, results[1].face_landmarks)

                if mode.name == "Balanced Mode":
                    detector.set_mode(PROCESSING_MODES[2])
                    with patch("app.mp.Image", return_value=object()):
                        detector.detect(frame)
                    self.assertEqual(detector.pose.calls, pose_count + 1)
                    self.assertEqual(detector.face.calls, face_count + 1)
                    self.assertEqual(detector.hands.calls, hand_count + 1)

    def test_performance_mode_resizes_inference_input_to_half(self):
        detector = MediaPipeTasks.__new__(MediaPipeTasks)
        detector.last_timestamp_ms = -1
        detector._rgb_buffer = None
        detector._inference_bgr_buffer = None
        detector._input_shape = None
        detector._frame_index = -1
        detector._force_refresh = True
        detector.mode = PROCESSING_MODES[2]
        detector._cached_pose_landmarks = None
        detector._cached_face_landmarks = None
        detector.last_inference_ms = 0.0
        empty_result = SimpleNamespace(pose_landmarks=None)
        detector.pose = FakeLandmarker(empty_result)
        detector.hands = FakeLandmarker(
            SimpleNamespace(hand_landmarks=[], handedness=[])
        )
        detector.face = FakeLandmarker(SimpleNamespace(face_landmarks=[]))
        image_shapes = []

        def make_image(**kwargs):
            image_shapes.append(kwargs["data"].shape)
            return object()

        with patch("app.mp.Image", side_effect=make_image):
            detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(image_shapes, [(240, 320, 3)])

    def test_media_pipe_input_dimensions_preserve_aspect_ratio(self):
        cases = ((1920, 1080), (640, 480), (1080, 1920))
        for width, height in cases:
            with self.subTest(source=(width, height)):
                detector = MediaPipeTasks.__new__(MediaPipeTasks)
                detector.last_timestamp_ms = -1
                detector._rgb_buffer = None
                detector._inference_bgr_buffer = None
                detector._input_shape = None
                detector._frame_index = -1
                detector._force_refresh = True
                detector.mode = PROCESSING_MODES[2]
                detector._cached_pose_landmarks = None
                detector._cached_poses_landmarks = []
                detector._cached_face_landmarks = None
                detector._cached_faces_landmarks = []
                detector.last_inference_ms = 0.0
                empty_pose = SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)
                detector.pose = FakeLandmarker(empty_pose)
                detector.hands = FakeLandmarker(SimpleNamespace(hand_landmarks=[], handedness=[]))
                detector.face = FakeLandmarker(SimpleNamespace(face_landmarks=[]))
                image_dimensions = []

                def make_image(*, dimensions=image_dimensions, **kwargs):
                    pixels = kwargs["data"]
                    dimensions.append((pixels.shape[1], pixels.shape[0]))
                    return SimpleNamespace(width=pixels.shape[1], height=pixels.shape[0])

                with patch("app.mp.Image", side_effect=make_image):
                    detector.detect(np.zeros((height, width, 3), dtype=np.uint8))

                expected = (width // 2, height // 2)
                self.assertEqual(image_dimensions, [expected])
                self.assertAlmostEqual(expected[0] / expected[1], width / height, places=5)

    def test_hud_and_landmark_transform_cover_landscape_standard_and_portrait_inputs(self):
        app = self.make_app()
        for source, target in (
            ((1920, 1080), (1280, 720)),
            ((640, 480), (1024, 768)),
            ((1080, 1920), (720, 1280)),
        ):
            with self.subTest(source=source, target=target):
                source_width, source_height = source
                target_width, target_height = target
                frame = np.zeros((source_height, source_width, 3), np.uint8)
                with patch("app.cv2.getWindowImageRect", return_value=(0, 0, target_width, target_height)):
                    display = VisionApp._fit_to_window(frame, "window")
                transform = ViewportTransform.cover(source_width, source_height, target_width, target_height)
                self.assertLessEqual(abs(transform.point(0.5, 0.5)[0] - target_width // 2), 1)
                self.assertLessEqual(abs(transform.point(0.5, 0.5)[1] - target_height // 2), 1)
                app._draw_hud(display, TrackingResults(None, {}, None))

    def test_f3_cycles_modes_and_refreshes_tasks(self):
        app = self.make_app()
        tasks = SimpleNamespace(set_mode=unittest.mock.Mock())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for expected_mode in (PROCESSING_MODES[2], PROCESSING_MODES[0], PROCESSING_MODES[1]):
            self.assertTrue(app._handle_key(0x720000, "test-window", frame, tasks))
            self.assertEqual(app.processing_mode, expected_mode)
            self.assertEqual(app.metrics.current_mode, expected_mode.name)

        self.assertEqual(tasks.set_mode.call_count, 3)


if __name__ == "__main__":
    unittest.main()
