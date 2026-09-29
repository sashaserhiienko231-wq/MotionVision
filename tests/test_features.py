from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from actions import ActionRecognizer, LandmarkPoint, TrackingResults
from app import VisionApp
from app_config import AppConfig
from camera import (
    CameraMode,
    CameraReadRecovery,
    CameraRequest,
    open_camera,
    parse_fps,
    parse_resolution,
)
from face_identity import FaceDatabase, FaceTracker, face_embedding
from telegram_bot import BotCommand, TelegramBot


def hand_gesture(gesture: str, scale: float = 1.0, noise: float = 0.0) -> list[LandmarkPoint]:
    points = [LandmarkPoint(0.5, 0.8) for _ in range(21)]
    points[0] = LandmarkPoint(0.5, 0.8)
    finger_ids = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    mcp_x = (0.40, 0.47, 0.54, 0.61)
    extended = {
        "middle_finger": (False, True, False, False),
        "pointing_up": (True, False, False, False),
        "pointing_left": (True, False, False, False),
        "pointing_right": (True, False, False, False),
        "thumb_side_left": (False, False, False, False),
        "thumb_side_right": (False, False, False, False),
        "thumb_up": (False, False, False, False),
        "thumb_down": (False, False, False, False),
    }.get(gesture, (False, False, False, False))
    for index, ((mcp, pip, tip), x, is_extended) in enumerate(zip(finger_ids, mcp_x, extended)):
        points[mcp] = LandmarkPoint(x, 0.58)
        points[pip] = LandmarkPoint(x, 0.42 if is_extended else 0.55)
        if is_extended and index == 0 and gesture == "pointing_right":
            points[mcp] = LandmarkPoint(0.40, 0.58)
            points[pip] = LandmarkPoint(0.58, 0.58)
            points[tip] = LandmarkPoint(0.78, 0.58)
        elif is_extended and index == 0 and gesture == "pointing_left":
            points[mcp] = LandmarkPoint(0.60, 0.58)
            points[pip] = LandmarkPoint(0.42, 0.58)
            points[tip] = LandmarkPoint(0.22, 0.58)
        else:
            points[tip] = LandmarkPoint(x, 0.22 if is_extended else 0.63)
    points[2] = LandmarkPoint(0.53, 0.68)
    points[3] = LandmarkPoint(0.54, 0.70)
    points[4] = LandmarkPoint(0.54, 0.72)
    if gesture.startswith("thumb_") or gesture in {"thumb_up", "thumb_down"}:
        points[3] = LandmarkPoint(0.63, 0.66)
        if gesture in {"thumb_up", "thumbs_up"}:
            points[4] = LandmarkPoint(0.57, 0.40)
        elif gesture in {"thumb_down", "thumbs_down"}:
            points[4] = LandmarkPoint(0.57, 0.98)
        elif gesture == "thumb_side_right":
            points[4] = LandmarkPoint(0.73, 0.67)
        elif gesture == "thumb_side_left":
            points[4] = LandmarkPoint(0.32, 0.67)
    # Keep the wrist as the scale origin so the ratios remain unchanged.
    origin = points[0]
    noisy = []
    rng = np.random.default_rng(73)
    for point in points:
        x = origin.x + (point.x - origin.x) * scale + rng.normal(0, noise)
        y = origin.y + (point.y - origin.y) * scale + rng.normal(0, noise)
        noisy.append(LandmarkPoint(float(x), float(y)))
    return noisy


def synthetic_face(offset: float = 0.0) -> list[LandmarkPoint]:
    points = [LandmarkPoint(0.5 + ((i * 17) % 83) / 500 + offset, 0.5 + ((i * 29) % 79) / 500) for i in range(468)]
    points[33] = LandmarkPoint(0.36 + offset, 0.43)
    points[263] = LandmarkPoint(0.64 + offset, 0.43)
    points[152] = LandmarkPoint(0.50 + offset, 0.85)
    if offset:
        points[10] = LandmarkPoint(points[10].x + offset, points[10].y - offset)
        points[152] = LandmarkPoint(points[152].x, points[152].y + offset)
    return points


class FakeCapture:
    def __init__(self, supported: dict[tuple[int, int], float]) -> None:
        self.supported = supported
        self.width, self.height, self.fps = 640, 480, 30.0
        self.requested_width, self.requested_height, self.requested_fps = self.width, self.height, 30.0
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        if prop == 3:
            self.requested_width = int(value)
        elif prop == 4:
            self.requested_height = int(value)
        elif prop == 5:
            self.requested_fps = float(value)
            actual_size = (self.requested_width, self.requested_height)
            self.fps = min(float(value), self.supported.get(actual_size, 30.0))
        return True

    def get(self, prop):
        return self.fps if prop == 5 else 0

    def read(self):
        actual_size = (self.requested_width, self.requested_height)
        if actual_size not in self.supported:
            actual_size = (640, 480)
            self.fps = 30.0
        return True, np.zeros((actual_size[1], actual_size[0], 3), np.uint8)

    def release(self):
        self.released = True
        self.opened = False


class FakeTransport:
    def __init__(self) -> None:
        self.calls = []
        self.files = []
        self.bytes = b"local photo bytes"

    def call(self, method, payload):
        self.calls.append((method, payload))
        if method == "getFile":
            return {"file_path": "incoming/photo.jpg"}
        return {"ok": True}

    def get_updates(self, offset, timeout=20):
        return []

    def get_file(self, file_id):
        return {"file_path": "incoming/photo.jpg"}

    def download_file(self, path):
        return self.bytes

    def send_file(self, method, chat_id, path, caption=""):
        self.files.append((method, chat_id, Path(path).read_bytes(), caption))


class NewGestureTests(unittest.TestCase):
    def test_new_gestures_use_geometry_and_survive_scale_and_small_noise(self):
        expected = {
            "thumb_side_left": "thumb_side_left",
            "thumb_side_right": "thumb_side_right",
            "middle_finger": "middle_finger",
            "pointing_up": "pointing_up",
            "pointing_left": "pointing_left",
            "pointing_right": "pointing_right",
        }
        for gesture, output in expected.items():
            for scale, noise in ((0.8, 0.0), (1.1, 0.0015)):
                with self.subTest(gesture=gesture, scale=scale, noise=noise):
                    name, confidence = ActionRecognizer._classify_finger_gesture(
                        hand_gesture(gesture, scale=scale, noise=noise)
                    )
                    self.assertEqual(name, output)
                    self.assertGreaterEqual(confidence, ActionRecognizer.GESTURE_MIN_CONFIDENCE)
                    self.assertLessEqual(confidence, 1.0)

    def test_thumb_up_down_preserve_legacy_event_names_and_direction(self):
        for source, expected in (("thumb_up", "thumbs_up"), ("thumb_down", "thumbs_down")):
            with self.subTest(source=source):
                name, confidence = ActionRecognizer._classify_finger_gesture(hand_gesture(source))
                self.assertEqual(name, expected)
                self.assertGreaterEqual(confidence, 0.74)

    def test_temporal_vote_rejects_single_frame_outlier_and_confirms_stable_gesture(self):
        recognizer = ActionRecognizer()
        emitted = []
        for index in range(14):
            name = "fist" if index == 7 else "pointing_right"
            result = TrackingResults(None, {"left": hand_gesture(name)}, None)
            with patch("actions.time.monotonic", return_value=20.0 + index * 0.05):
                emitted.extend(recognizer.recognize(result))
        self.assertEqual([event.name for event in emitted], ["pointing_right"])
        self.assertEqual(emitted[0].handedness, "left")

    def test_classifier_rejects_incomplete_and_degenerate_landmarks(self):
        self.assertEqual(ActionRecognizer._classify_finger_gesture([]), (None, 0.0))
        degenerate = [LandmarkPoint(0.5, 0.5) for _ in range(21)]
        self.assertEqual(ActionRecognizer._classify_finger_gesture(degenerate), (None, 0.0))


class CameraNegotiationTests(unittest.TestCase):
    def test_temporary_read_failure_recovers_without_reconnect_or_state_reset(self):
        recovery = CameraReadRecovery(reconnect_after=3)
        frame = np.zeros((240, 320, 3), np.uint8)
        self.assertFalse(recovery.observe(False, None))
        self.assertEqual(recovery.consecutive_failures, 1)
        self.assertFalse(recovery.observe(True, frame))
        self.assertEqual(recovery.consecutive_failures, 0)
        self.assertFalse(recovery.observe(False, None))
        self.assertFalse(recovery.observe(False, None))
        self.assertTrue(recovery.observe(False, None))


class CameraRecoveryAndModeTests(unittest.TestCase):
    def test_one_bad_read_keeps_capture_tracker_and_writer_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(history_path=root / "actions.json", recordings_dir=root / "recordings",
                            models_dir=root / "models", config_path=root / "config.json",
                            ui_config_path=root / "ui.json", faces_path=root / "faces.json")
            frame = np.zeros((240, 320, 3), np.uint8)

            class Capture:
                released = False
                reads = 0

                def read(self):
                    self.reads += 1
                    if self.reads == 1:
                        return False, None
                    return True, frame.copy()

                def release(self):
                    self.released = True

            class Writer:
                writes = 0
                released = False

                def write(self, _frame):
                    self.writes += 1

                def release(self):
                    self.released = True

            class Tasks:
                def __init__(self):
                    self.detect_calls = 0
                    self._force_refresh = True
                    self.mode = app.processing_mode
                    self.face_detection_fps = self.hand_detection_fps = self.pose_detection_fps = 0.0

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    pass

                def set_mode(self, _mode):
                    pass

                def detect(self, _frame):
                    self.detect_calls += 1
                    return TrackingResults(None, {}, None)

                def embed_face(self, *_args):
                    return None

            capture, writer, tasks = Capture(), Writer(), Tasks()
            app.video_writer = writer
            app._recording_size = (320, 240)
            app._latest_frame = frame.copy()
            with patch.object(app.telegram, "start"), patch.object(app.telegram, "stop"), \
                 patch.object(app, "_handle_bot_commands"), \
                 patch.object(app, "_open_camera", return_value=(capture, CameraMode(320, 240, 30))), \
                 patch("app.MediaPipeTasks", return_value=tasks), \
                 patch.object(app.window_manager, "state", return_value=None), \
                 patch.object(app.window_manager, "work_areas", return_value=(
                     __import__("windowing").WorkArea(0, 0, 1920, 1080, True),
                 )), patch.object(app.window_manager, "restore"), \
                 patch("app.cv2.namedWindow"), patch("app.cv2.resizeWindow"), patch("app.cv2.moveWindow"), \
                 patch("app.cv2.getWindowImageRect", return_value=(0, 0, 800, 600)), \
                 patch("app.cv2.imshow"), patch("app.cv2.destroyAllWindows"), \
                 patch("app.cv2.waitKeyEx", side_effect=(-1, -1, ord("q"))), \
                 patch.object(app, "_draw_landmarks"), patch.object(app, "_draw_face_matches"), \
                 patch.object(app, "_draw_hud"):
                app.run()

            self.assertEqual(capture.reads, 3)
            self.assertTrue(capture.released)
            self.assertEqual(tasks.detect_calls, 2)
            self.assertEqual(writer.writes, 2)
            self.assertTrue(writer.released)

    def test_unavailable_camera_shows_retry_status_and_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(history_path=root / "actions.json", recordings_dir=root / "recordings",
                            models_dir=root / "models", config_path=root / "config.json",
                            ui_config_path=root / "ui.json", faces_path=root / "faces.json")

            class Tasks:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    pass

                def set_mode(self, _mode):
                    pass

            with patch.object(app.telegram, "start"), patch.object(app.telegram, "stop"), \
                 patch.object(app, "_handle_bot_commands"), \
                 patch.object(app, "_open_camera", side_effect=RuntimeError("no device")), \
                 patch("app.MediaPipeTasks", return_value=Tasks()), \
                 patch.object(app.window_manager, "state", return_value=None), \
                 patch.object(app.window_manager, "work_areas", return_value=(
                     __import__("windowing").WorkArea(0, 0, 1920, 1080, True),
                 )), patch.object(app.window_manager, "restore"), \
                 patch("app.cv2.namedWindow"), patch("app.cv2.resizeWindow"), patch("app.cv2.moveWindow"), \
                 patch("app.cv2.imshow"), patch("app.cv2.destroyAllWindows"), \
                 patch("app.cv2.waitKeyEx", return_value=ord("q")):
                app.run()

            self.assertIn("Camera 0 unavailable", app.status_message)

    def test_parse_capture_requests(self):
        self.assertEqual(parse_resolution("1920x1080"), (1920, 1080))
        self.assertIsNone(parse_resolution("Auto"))
        self.assertEqual(parse_fps("120"), 120)
        self.assertIsNone(parse_fps("Auto"))
        with self.assertRaises(ValueError):
            parse_resolution("9999x9999")
        with self.assertRaises(ValueError):
            parse_fps("144")

    def test_resolution_and_fps_fallback_reports_observed_frame_shape(self):
        supported = {(640, 480): 30.0, (1280, 720): 30.0, (1920, 1080): 60.0}
        made = []

        def factory(index):
            capture = FakeCapture(supported)
            made.append(capture)
            return capture

        capture, mode = open_camera(
            0, CameraRequest((3840, 2160), 120), factory,
            prop_width=3, prop_height=4, prop_fps=5, prop_buffer=6, sample_frames=3,
        )
        self.assertEqual((mode.width, mode.height), (1920, 1080))
        self.assertEqual(mode.fps, 60.0)
        self.assertFalse(capture.released)
        self.assertTrue(all(item.released for item in made if item is not capture))

    def test_auto_mode_probes_modes_and_selects_highest_delivered_pixel_rate(self):
        supported = {(640, 480): 30.0, (1280, 720): 60.0, (1920, 1080): 60.0}
        capture, mode = open_camera(
            0, CameraRequest(), lambda _: FakeCapture(supported),
            prop_width=3, prop_height=4, prop_fps=5, prop_buffer=6, sample_frames=3,
        )
        self.assertEqual((mode.width, mode.height, mode.fps), (1920, 1080, 60.0))
        capture.release()

    def test_auto_does_not_return_default_when_last_probe_fps_is_noisy(self):
        class NoisyFinalProbe(FakeCapture):
            def __init__(self, supported):
                super().__init__(supported)
                self.read_count = 0

            def read(self):
                self.read_count += 1
                result = super().read()
                # Make the final Auto fallback report a slightly different
                # rate from the earlier 640x480@30 probe so exact-key
                # deduplication cannot hide an incorrect early return.
                if self.read_count > 60:
                    self.fps = 27.4
                return result

        capture, mode = open_camera(
            0, CameraRequest(), lambda _: NoisyFinalProbe({(640, 480): 30.0, (1280, 720): 30.0, (1920, 1080): 60.0}),
            prop_width=3, prop_height=4, prop_fps=5, prop_buffer=6, sample_frames=4,
        )
        self.assertEqual((mode.width, mode.height), (1920, 1080))
        capture.release()

    def test_disconnected_camera_fails_cleanly(self):
        capture = FakeCapture({})
        capture.opened = False
        with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
            open_camera(0, CameraRequest(), lambda _: capture,
                        prop_width=3, prop_height=4, prop_fps=5, prop_buffer=6)


class FaceIdentityTests(unittest.TestCase):
    def test_face_recognition_can_be_disabled_from_environment(self):
        with patch.dict(os.environ, {"FACE_RECOGNITION_ENABLED": "false"}):
            self.assertFalse(AppConfig.from_env().face_recognition_enabled)

    def test_local_face_embedding_and_database_matching(self):
        first = face_embedding(synthetic_face())
        self.assertIsNotNone(first)
        self.assertEqual(len(first), 76)
        with tempfile.TemporaryDirectory() as directory:
            database = FaceDatabase(Path(directory) / "faces.json")
            person_id = database.add("Alice", first)
            self.assertEqual(database.match(first, 0.6), ("Alice", 1.0))
            self.assertEqual(FaceDatabase(Path(directory) / "faces.json").people[person_id][0], "Alice")
            self.assertEqual(database.match([1.0] + [0.0] * (len(first) - 1), 0.99)[0], "UNKNOWN")
            self.assertTrue(database.delete(person_id))
            database.clear()

    def test_new_face_id_does_not_overwrite_existing_record_after_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            database = FaceDatabase(Path(directory) / "faces.json")
            alice = database.add("Alice", [1.0, 0.0])
            bob = database.add("Bob", [0.0, 1.0])
            charlie = database.add("Charlie", [0.5, 0.5])
            self.assertTrue(database.delete(bob))
            diana = database.add("Diana", [-1.0, 0.0])
            self.assertEqual((alice, charlie, diana), ("person-0001", "person-0003", "person-0004"))
            self.assertEqual(database.people[charlie][0], "Charlie")

    def test_face_tracker_keeps_ids_across_order_changes_and_short_gaps(self):
        tracker = FaceTracker()
        first = tracker.assign([(10, 10, 50, 60), (150, 10, 50, 60)])
        second = tracker.assign([(151, 10, 50, 60), (11, 10, 50, 60)])
        self.assertEqual(second, [first[1], first[0]])
        tracker.assign([])
        third = tracker.assign([(12, 10, 50, 60)])
        self.assertEqual(third[0], first[0])

    def test_multi_person_pipeline_and_unknown_face(self):
        with tempfile.TemporaryDirectory() as directory:
            app = VisionApp(
                config=AppConfig(face_match_threshold=0.99),
                history_path=Path(directory) / "actions.json",
                recordings_dir=Path(directory) / "recordings",
                models_dir=Path(directory) / "models",
                faces_path=Path(directory) / "faces.json",
                config_path=Path(directory) / "config.json",
            )
            first, second = synthetic_face(), synthetic_face(0.1)
            app.face_database.add("Alice", [1.0, 0.0])
            results = TrackingResults(None, {}, first, [first, second])
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            embed = lambda _frame, landmarks: [1.0, 0.0] if landmarks is first else [0.0, 1.0]
            one = app._update_face_tracks(results, 640, 480, frame, embed)
            two = app._update_face_tracks(results, 640, 480, frame, embed)
            self.assertEqual(len(one), 2)
            self.assertEqual([item.track_id for item in one], [item.track_id for item in two])
            self.assertEqual(one[0].name, "Alice")
            self.assertEqual(one[1].name, "UNKNOWN")
            self.assertEqual(app.metrics.faces_found, 2)

    def test_face_tracking_skips_sface_until_a_person_is_enrolled(self):
        with tempfile.TemporaryDirectory() as directory:
            app = VisionApp(
                history_path=Path(directory) / "actions.json",
                recordings_dir=Path(directory) / "recordings",
                models_dir=Path(directory) / "models",
                config_path=Path(directory) / "config.json",
                faces_path=Path(directory) / "faces.json",
            )
            face = synthetic_face()
            results = TrackingResults(None, {}, face, [face])
            calls = []
            app._update_face_tracks(results, 640, 480, np.zeros((480, 640, 3), dtype=np.uint8),
                                    lambda *_: calls.append(True) or [1.0])
            self.assertEqual(calls, [])

    def test_disabled_face_recognition_keeps_tracking_but_skips_embedding_and_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(
                config=AppConfig(face_recognition_enabled=False),
                history_path=root / "actions.json",
                recordings_dir=root / "recordings",
                models_dir=root / "models",
                config_path=root / "config.json",
                faces_path=root / "faces.json",
            )
            app.face_database.add("Alice", [1.0, 0.0])
            face = synthetic_face()
            calls = []
            tracks = app._update_face_tracks(
                TrackingResults(None, {}, face, [face]), 640, 480,
                np.zeros((480, 640), dtype=np.uint8),
                lambda *_: calls.append(True) or [1.0, 0.0],
            )
            self.assertEqual(calls, [])
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].name, "UNKNOWN")
            self.assertTrue(app.metrics.face_detected)

    def test_disabled_face_recognition_blocks_telegram_enrollment_and_watch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(
                config=AppConfig(face_recognition_enabled=False),
                history_path=root / "actions.json",
                recordings_dir=root / "recordings",
                models_dir=root / "models",
                config_path=root / "config.json",
                faces_path=root / "faces.json",
            )
            transport = FakeTransport()
            app.telegram = TelegramBot("mock-token", transport=transport)
            app.telegram.commands.put(BotCommand("42", "/watch"))
            app.telegram.commands.put(BotCommand("42", "/addface_photo", photo=b"unused"))
            landmarkers = type("Landmarkers", (), {
                "encode_face_image": lambda *_: self.fail("face embedding must stay disabled")
            })()

            app._handle_bot_commands(landmarkers)

            texts = [payload["text"] for method, payload in transport.calls if method == "sendMessage"]
            self.assertEqual(texts, [
                "Face match watch disabled.",
                "Face recognition is disabled in settings.",
            ])


class SFaceTests(unittest.TestCase):
    def test_mediapipe_landmarks_convert_to_yunet_alignment_row(self):
        from sface import LocalFaceEmbedder

        row = LocalFaceEmbedder.detection_row(synthetic_face(), 640, 480)
        self.assertIsNotNone(row)
        self.assertEqual(row.shape, (1, 15))
        self.assertTrue(np.isfinite(row).all())
        self.assertEqual(float(row[0, -1]), 1.0)

    def test_sface_embedding_is_normalized_and_rejects_bad_alignment(self):
        from sface import LocalFaceEmbedder

        embedder = object.__new__(LocalFaceEmbedder)

        class FakeRecognizer:
            def alignCrop(self, _frame, row):
                self.row = row
                return np.zeros((112, 112, 3), dtype=np.uint8)

            def feature(self, _aligned):
                return np.asarray([[3.0, 4.0]], dtype=np.float32)

        embedder.recognizer = FakeRecognizer()
        vector = embedder.embed(np.zeros((480, 640, 3), np.uint8), synthetic_face())
        np.testing.assert_allclose(vector, [0.6, 0.8], atol=1e-7)
        self.assertIsNone(embedder.embed(np.zeros((0, 0, 3), np.uint8), []))

    def test_downloaded_sface_model_produces_a_real_128d_descriptor(self):
        from sface import MODEL_NAME, LocalFaceEmbedder

        model_dir = Path(__file__).resolve().parents[1] / "models"
        if not (model_dir / MODEL_NAME).is_file():
            self.skipTest("The optional SFace model has not been downloaded")
        embedder = LocalFaceEmbedder(model_dir)
        vector = embedder.embed(np.zeros((480, 640, 3), np.uint8), synthetic_face())
        self.assertIsNotNone(vector)
        self.assertEqual(len(vector), 128)
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


class WindowStateTests(unittest.TestCase):
    def make_app(self, root: Path, **kwargs) -> VisionApp:
        return VisionApp(
            history_path=root / "actions.json", recordings_dir=root / "recordings",
            models_dir=root / "models", ui_config_path=root / "ui.json",
            config_path=root / "config.json", faces_path=root / "faces.json", **kwargs,
        )

    def test_window_size_position_viewport_and_camera_mode_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = self.make_app(root, camera_index=2)
            app._latest_frame = np.zeros((720, 1280, 3), np.uint8)
            app.metrics.resolution = "1280x720"
            with patch("app.cv2.getWindowImageRect", return_value=(33, 44, 1000, 700)), \
                 patch.object(app.window_manager, "client_work_areas", return_value=(
                     __import__("windowing").WorkArea(0, 0, 1920, 1080, True),
                 )):
                app._save_window_state("window")
            stored = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual((stored["x"], stored["y"], stored["width"], stored["height"]), (33, 44, 1000, 700))
            self.assertEqual((stored["viewport_width"], stored["viewport_height"]), (1000, 700))
            self.assertEqual((stored["frame_width"], stored["frame_height"]), (1280, 720))
            self.assertEqual(stored["camera_index"], 2)
            self.assertNotIn("TELEGRAM_BOT_TOKEN", stored)

    def test_fullscreen_restore_reapplies_saved_window_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            app.window_state.update(x=12, y=23, width=900, height=600)
            frame = np.zeros((480, 640, 3), np.uint8)
            with patch("app.cv2.getWindowImageRect", return_value=(12, 23, 900, 600)), \
                 patch("app.cv2.setWindowProperty") as set_property, \
                 patch("app.cv2.resizeWindow") as resize, patch("app.cv2.moveWindow") as move:
                app._handle_key(0x7A, "window", frame)
                self.assertTrue(app.fullscreen)
                app._handle_key(0x7A, "window", frame)
            self.assertFalse(app.fullscreen)
            self.assertEqual(set_property.call_count, 2)
            resize.assert_called_with("window", 900, 600)
            move.assert_called_with("window", 12, 23)

    def test_minimized_bad_viewport_keeps_last_valid_display_size(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch("app.cv2.getWindowImageRect", return_value=(0, 0, 0, 0)):
            shown = VisionApp._fit_to_window(frame, "window", (1024, 768))
        self.assertEqual(shown.shape[:2], (768, 1024))

    def test_minimized_window_save_preserves_last_valid_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            app.window_state.update(x=33, y=44, width=1000, height=700)
            app._restored_window_rect = (33, 44, 1000, 700)
            app._latest_frame = np.zeros((480, 640, 3), np.uint8)
            with patch("app.cv2.getWindowImageRect", return_value=(0, 0, 0, 0)):
                app._save_window_state("window")
            self.assertEqual(
                tuple(app.window_state.data[key] for key in ("x", "y", "width", "height")),
                (33, 44, 1000, 700),
            )

    def test_maximized_and_fullscreen_states_do_not_overwrite_normal_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            saved_rect = (33, 44, 1000, 700)
            app.window_state.update(x=33, y=44, width=1000, height=700)
            app._restored_window_rect = saved_rect
            app._latest_frame = np.zeros((480, 640, 3), np.uint8)
            for state, fullscreen in (("maximized", False), ("minimized", False), ("normal", True)):
                with self.subTest(state=state, fullscreen=fullscreen):
                    app.fullscreen = fullscreen
                    with patch.object(app.window_manager, "state", return_value=state), \
                         patch("app.cv2.getWindowImageRect", return_value=(0, 0, 1920, 1080)):
                        app._save_window_state("window")
                    self.assertEqual(
                        tuple(app.window_state.data[key] for key in ("x", "y", "width", "height")),
                        saved_rect,
                    )

    def test_monitor_change_clamps_restored_window_and_reset_recenters_it(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            app.window_state.update(x=1500, y=700, width=1000, height=600)
            with patch.object(app.window_manager, "work_areas", return_value=(
                __import__("windowing").WorkArea(0, 0, 1280, 720, True),
            )), patch("app.cv2.resizeWindow") as resize, patch("app.cv2.moveWindow") as move:
                app._restore_window("window")
                resize.assert_called_with("window", 1000, 600)
                move.assert_called_with("window", 280, 120)
                app._reset_window_geometry("window")
            self.assertEqual(app._restored_window_rect, (0, 0, 1280, 720))

    def test_ctrl_shift_r_hotkey_resets_geometry_without_starting_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            frame = np.zeros((480, 640, 3), np.uint8)
            with patch.object(app.window_manager, "reset_shortcut", return_value=True), \
                 patch.object(app.window_manager, "work_areas", return_value=(
                     __import__("windowing").WorkArea(0, 0, 1600, 900, True),
                 )), patch("app.cv2.resizeWindow"), patch("app.cv2.moveWindow"):
                self.assertTrue(app._handle_key(ord("r"), "window", frame))
            self.assertEqual(app._restored_window_rect, (160, 90, 1280, 720))
            self.assertIsNone(app.video_writer)

    def test_reset_window_cli_flag_is_passed_to_application(self):
        import sys

        from app import main

        with patch.object(sys, "argv", ["app.py", "--reset-window"]), \
             patch("app.WindowState") as window_state, patch("app.VisionApp") as app_class:
            window_state.return_value.data = {}
            main()
        self.assertTrue(app_class.call_args.kwargs["reset_window"])

    def test_hud_panels_drag_and_save_positions_from_inside_app(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            frame = np.zeros((720, 1280, 3), np.uint8)
            hands = [LandmarkPoint(0.5, 0.5) for _ in range(21)]
            hands[8] = LandmarkPoint(0.025, 0.04)
            hands[4] = LandmarkPoint(0.035, 0.04)
            result = TrackingResults(None, {"left": hands}, None)
            app._draw_hud(frame, result)
            self.assertIn("motion_stats", app.panel_layout.positions)
            hands[8] = LandmarkPoint(0.4, 0.3)
            hands[4] = LandmarkPoint(0.41, 0.3)
            app._draw_hud(frame, result)
            hands[4] = LandmarkPoint(0.7, 0.3)
            app._draw_hud(frame, result)
            self.assertTrue((Path(directory) / "ui.json").is_file())


class RecordingAndTelegramTests(unittest.TestCase):
    def test_missing_telegram_token_keeps_local_recording_available(self):
        statuses = []
        bot = TelegramBot("", on_status=statuses.append)
        self.assertFalse(bot.start())
        self.assertIn("not configured", statuses[-1])

    def test_bot_photo_enrollment_download_is_queued_as_bytes(self):
        transport = FakeTransport()
        bot = TelegramBot("mock-token", transport=transport)
        bot.handle_update({"update_id": 1, "message": {"chat": {"id": 42, "type": "private"}, "text": "/addface Alice"}})
        bot.handle_update({"update_id": 2, "message": {"chat": {"id": 42, "type": "private"}, "photo": [{"file_id": "photo-id"}]}})
        queued = bot.commands.get_nowait()
        self.assertEqual(queued, BotCommand("42", "/addface_photo", ("Alice",), transport.bytes))

    def test_photo_enrollment_runs_local_embedding_and_invalidates_face_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(history_path=root / "actions.json", recordings_dir=root / "recordings",
                            models_dir=root / "models", config_path=root / "config.json",
                            faces_path=root / "faces.json")
            transport = FakeTransport()
            app.telegram = TelegramBot("", transport=transport)
            cached_landmarks = synthetic_face()
            result = TrackingResults(None, {}, cached_landmarks, [cached_landmarks])
            frame = np.zeros((480, 640, 3), np.uint8)
            app._update_face_tracks(result, 640, 480, frame, lambda *_: [1.0, 0.0])
            self.assertTrue(app._face_embedding_cache)

            class FakeLandmarkers:
                def encode_face_image(self, _image):
                    return [1.0, 0.0]

            app.telegram.commands.put(BotCommand("42", "/addface_photo", ("Alice",), b"photo"))
            with patch("app.cv2.imdecode", return_value=frame.copy()):
                app._handle_bot_commands(FakeLandmarkers())
            self.assertEqual(app.face_database.match([1.0, 0.0]), ("Alice", 1.0))
            self.assertFalse(app._face_embedding_cache)
            self.assertIn("Added Alice", transport.calls[-1][1]["text"])

    def test_mock_telegram_upload_happens_after_writer_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(telegram_bot_token="mock", telegram_chat_id="123")
            app = VisionApp(config=config, history_path=root / "actions.json", recordings_dir=root / "recordings",
                            models_dir=root / "models", config_path=root / "config.json", faces_path=root / "faces.json")
            transport = FakeTransport()
            app.telegram = TelegramBot("mock", "123", transport=transport)

            class Writer:
                released = False
                def __init__(self, path): self.path = Path(path)
                def isOpened(self): return True
                def write(self, frame): self.pending = frame.copy()
                def release(self):
                    self.released = True
                    self.path.write_bytes(b"finalized mock video")

            writer = Writer.__new__(Writer)
            def make_writer(path, *_):
                writer.path = Path(path)
                writer.released = False
                return writer
            original_send = app.telegram.send_file
            def send_file(chat_id, method, path, caption=""):
                self.assertTrue(writer.released)
                original_send(chat_id, method, path, caption)
            app.telegram.send_file = send_file
            app.metrics.camera_fps = 60.0
            frame = np.zeros((240, 320, 3), np.uint8)
            with patch("app.cv2.VideoWriter", side_effect=make_writer):
                app._start_recording(frame)
                app.video_writer.write(frame)
                self.assertEqual(app._recording_size, (320, 240))
                self.assertEqual(app._recording_fps, 60.0)
                app._stop_recording()
            self.assertTrue(transport.files)
            self.assertEqual(transport.files[0][0], "sendVideo")
            self.assertEqual(transport.files[0][2], b"finalized mock video")

    def test_opencv_video_writer_saves_decodable_camera_sized_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(config=AppConfig(), history_path=root / "actions.json",
                            recordings_dir=root / "recordings", models_dir=root / "models",
                            config_path=root / "config.json", faces_path=root / "faces.json")
            app.metrics.camera_fps = 25.0
            frame = np.zeros((240, 320, 3), np.uint8)
            app._start_recording(frame)
            self.assertIsNotNone(app.video_writer)
            path = app._recording_path
            for value in range(8):
                frame[:] = (value * 20, 50, 100)
                app.video_writer.write(frame)
            app._stop_recording()
            self.assertGreater(path.stat().st_size, 0)
            reader = cv2.VideoCapture(str(path))
            try:
                ok, saved = reader.read()
            finally:
                reader.release()
            self.assertTrue(ok)
            self.assertEqual(saved.shape[:2], (240, 320))


class MediaPipeSmokeTests(unittest.TestCase):
    def test_media_pipe_tasks_models_process_a_synthetic_frame(self):
        from mediapipe.tasks.python import BaseOptions
        from app import PROCESSING_MODES, MediaPipeTasks
        model_dir = Path(__file__).resolve().parents[1] / "models"
        if not all((model_dir / name).exists() or (model_dir / f"{name}.download").exists()
                   for name in ("pose_landmarker_lite.task", "hand_landmarker.task", "face_landmarker.task")):
            self.skipTest("Bundled MediaPipe task assets are missing")
        # Exercise actual Tasks inference through the CPU path on headless CI runners.
        with MediaPipeTasks(model_dir, delegate=BaseOptions.Delegate.CPU) as tasks:
            tasks.set_mode(PROCESSING_MODES[2])
            result = tasks.detect(np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertEqual(result.hand_landmarks, {})
            self.assertIsNone(result.face_landmarks)
            self.assertIsNone(result.pose_landmarks)
            self.assertGreater(tasks.last_inference_ms, 0.0)

    def test_watch_alert_obeys_per_person_cooldown_and_saves_snapshot(self):
        from face_identity import FaceTrack
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = VisionApp(config=AppConfig(telegram_bot_token="mock", telegram_chat_id="123", face_alert_cooldown=30),
                            history_path=root / "actions.json", recordings_dir=root / "recordings",
                            models_dir=root / "models", config_path=root / "config.json", faces_path=root / "faces.json")
            transport = FakeTransport()
            app.telegram = TelegramBot("mock", "123", transport=transport)
            app.watch_mode = True
            app.face_tracks = [FaceTrack(1, (10, 10, 20, 20), "Alice", 0.94)]
            frame = np.zeros((80, 80, 3), np.uint8)
            with patch("app.time.monotonic", side_effect=(100.0, 120.0, 131.0)):
                for _ in range(3):
                    app._check_face_alerts(app.face_tracks, frame)
            messages = [call for call in transport.calls if call[0] == "sendMessage"]
            self.assertEqual(len(messages), 2)
            self.assertTrue(any(root.joinpath("recordings").glob("face_event_*.jpg")))


if __name__ == "__main__":
    unittest.main()
