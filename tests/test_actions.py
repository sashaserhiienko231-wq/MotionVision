import unittest
from types import SimpleNamespace
from unittest.mock import patch

from actions import (
    ActionRecognizer,
    LandmarkPoint,
    TrackingResults,
    tracking_results_from_tasks,
)


def make_hand_landmarks(
    extended_fingers: tuple[bool, bool, bool, bool],
    thumb_extended: bool,
    thumb_y: float = 0.45,
) -> list[LandmarkPoint]:
    landmarks = [LandmarkPoint(0.5, 0.8) for _ in range(21)]
    landmarks[2] = LandmarkPoint(0.54, 0.68)
    landmarks[5] = LandmarkPoint(0.6, 0.58)
    for index, (tip, pip, x_offset, is_extended) in enumerate(
        zip((8, 12, 16, 20), (6, 10, 14, 18), (0.54, 0.57, 0.60, 0.63), extended_fingers)
    ):
        landmarks[pip] = LandmarkPoint(x_offset, 0.55)
        landmarks[tip] = LandmarkPoint(
            x_offset,
            0.25 if is_extended else 0.63,
        )

    if thumb_extended:
        landmarks[3] = LandmarkPoint(0.45, 0.62)
        landmarks[4] = LandmarkPoint(0.30, thumb_y)
    else:
        landmarks[3] = LandmarkPoint(0.55, 0.62)
        landmarks[4] = LandmarkPoint(0.55, 0.72)
    return landmarks


class ActionRecognizerTests(unittest.TestCase):
    def test_emits_hand_raised_once_per_raise(self):
        recognizer = ActionRecognizer()
        pose = [LandmarkPoint(0.5, 0.6) for _ in range(33)]
        pose[11] = LandmarkPoint(0.4, 0.5)
        pose[12] = LandmarkPoint(0.6, 0.5)
        pose[15] = LandmarkPoint(0.3, 0.3)
        pose[16] = LandmarkPoint(0.7, 0.7)
        results = TrackingResults(
            pose_landmarks=pose,
            hand_landmarks={},
            face_landmarks=None,
        )

        first_events = []
        for index in range(8):
            with patch("actions.time.monotonic", return_value=10.0 + index * 0.05):
                first_events.extend(recognizer.recognize(results))
        with patch("actions.time.monotonic", return_value=10.5):
            second_events = recognizer.recognize(results)

        self.assertEqual([event.name for event in first_events], ["hand_raised"])
        self.assertEqual(first_events[0].details["side"], "left")
        self.assertEqual(second_events, [])

    def test_emits_hand_wave_after_horizontal_reversals(self):
        recognizer = ActionRecognizer()
        results = TrackingResults(
            pose_landmarks=None,
            hand_landmarks={"left": [LandmarkPoint(0.3, 0.5)]},
            face_landmarks=None,
        )
        positions = [0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5]
        events = []

        for index, position in enumerate(positions):
            results.hand_landmarks["left"][0] = LandmarkPoint(position, 0.5)
            with patch("actions.time.monotonic", return_value=10.0 + index * 0.1):
                events.extend(recognizer.recognize(results))

        self.assertEqual([event.name for event in events], ["hand_wave"])
        self.assertEqual(events[0].details["side"], "left")

    def test_normalizes_tasks_result_arrays_and_handedness(self):
        pose = [SimpleNamespace(x=0.5, y=0.6, z=0.1, visibility=0.9)]
        hand = [SimpleNamespace(x=0.3, y=0.4, z=0.0)]
        right_hand = [SimpleNamespace(x=0.7, y=0.4, z=0.0)]
        face = [SimpleNamespace(x=0.5, y=0.5, z=-0.1)]
        results = tracking_results_from_tasks(
            SimpleNamespace(pose_landmarks=[pose]),
            SimpleNamespace(
                hand_landmarks=[hand, right_hand],
                handedness=[
                    [SimpleNamespace(category_name="Left")],
                    [SimpleNamespace(category_name="Right")],
                ],
            ),
            SimpleNamespace(face_landmarks=[face]),
        )

        self.assertEqual(results.pose_landmarks[0].visibility, 0.9)
        self.assertEqual(results.hand_landmarks["left"][0].x, 0.3)
        self.assertEqual(results.hand_landmarks["right"][0].x, 0.7)
        self.assertEqual(results.face_landmarks[0].z, -0.1)

    def test_none_and_missing_optional_landmark_fields_use_safe_defaults(self):
        pose = [
            SimpleNamespace(
                x=0.5,
                y=0.6,
                z=None,
                visibility=None,
                presence=None,
            )
            for _ in range(33)
        ]
        pose[11] = SimpleNamespace(x=0.4, y=0.5, visibility=None, presence=None)
        pose[12] = SimpleNamespace(x=0.6, y=0.5, visibility=None, presence=None)
        pose[15] = SimpleNamespace(x=0.3, y=0.3, visibility=None, presence=None)
        hand = [
            SimpleNamespace(x=0.3, y=0.4, z=None, visibility=None, presence=None)
        ]
        face = [
            SimpleNamespace(x=0.5, y=0.5, z=None, visibility=None, presence=None),
            SimpleNamespace(x=0.5, y=0.4),
        ]
        results = tracking_results_from_tasks(
            SimpleNamespace(pose_landmarks=[pose]),
            SimpleNamespace(
                hand_landmarks=[hand],
                handedness=[[SimpleNamespace(category_name="Left", score=None)]],
            ),
            SimpleNamespace(face_landmarks=[face]),
        )

        for landmark in (
            results.pose_landmarks[0],
            results.hand_landmarks["left"][0],
            results.face_landmarks[0],
            results.face_landmarks[1],
        ):
            self.assertEqual(landmark.z, 0.0)
            self.assertEqual(landmark.visibility, 1.0)
            self.assertEqual(landmark.presence, 1.0)

        self.assertEqual(results.face_landmarks[1].x, 0.5)
        self.assertEqual(ActionRecognizer().recognize(results), [])

    def test_emits_head_shake_from_face_and_pose_landmarks(self):
        recognizer = ActionRecognizer()
        pose = [LandmarkPoint(0.5, 0.8) for _ in range(33)]
        pose[11] = LandmarkPoint(0.4, 0.4)
        pose[12] = LandmarkPoint(0.6, 0.4)
        pose[15] = LandmarkPoint(0.3, 0.8)
        pose[16] = LandmarkPoint(0.7, 0.8)
        face = [LandmarkPoint(0.5, 0.3), LandmarkPoint(0.5, 0.3)]
        results = TrackingResults(pose, {}, face)
        relative_positions = [-0.3, 0.1, 0.4, 0.1, -0.3, 0.1, 0.4]
        events = []

        for index, relative_x in enumerate(relative_positions):
            face[1] = LandmarkPoint(0.5 + relative_x * 0.2, 0.3)
            with patch("actions.time.monotonic", return_value=10.0 + index * 0.1):
                events.extend(recognizer.recognize(results))

        self.assertEqual([event.name for event in events], ["head_shake_left_right"])

    def test_recognizes_all_five_finger_gestures(self):
        cases = (
            ("thumbs_up", (False, False, False, False), True, 0.4),
            ("thumbs_down", (False, False, False, False), True, 0.98),
            ("peace_sign", (True, True, False, False), False, 0.45),
            ("open_palm", (True, True, True, True), True, 0.45),
            ("fist", (False, False, False, False), False, 0.45),
        )

        for expected, fingers, thumb, thumb_y in cases:
            with self.subTest(gesture=expected):
                recognizer = ActionRecognizer()
                results = TrackingResults(
                    pose_landmarks=None,
                    hand_landmarks={
                        "right": make_hand_landmarks(fingers, thumb, thumb_y)
                    },
                    face_landmarks=None,
                )
                emitted = []
                for index in range(12):
                    timestamp = 10.0 + index * 0.05
                    with patch("actions.time.monotonic", return_value=timestamp):
                        emitted.extend(recognizer.recognize(results))

                self.assertEqual([event.name for event in emitted], [expected])
                self.assertEqual(emitted[0].handedness, "right")
                self.assertEqual(emitted[0].details["handedness"], "right")
                self.assertGreater(emitted[0].confidence, 0.0)
                self.assertLessEqual(emitted[0].confidence, 1.0)

    def test_same_gesture_is_recognized_independently_for_each_hand(self):
        recognizer = ActionRecognizer()
        peace_sign = make_hand_landmarks((True, True, False, False), False)
        results = TrackingResults(
            pose_landmarks=None,
            hand_landmarks={"left": peace_sign.copy(), "right": peace_sign.copy()},
            face_landmarks=None,
        )
        events = []

        for index in range(12):
            timestamp = 10.0 + index * 0.05
            with patch("actions.time.monotonic", return_value=timestamp):
                events.extend(recognizer.recognize(results))

        self.assertEqual([event.handedness for event in events], ["left", "right"])

    def test_stable_finger_gesture_is_debounced_until_hand_changes(self):
        recognizer = ActionRecognizer()
        results = TrackingResults(
            pose_landmarks=None,
            hand_landmarks={
                "left": make_hand_landmarks((True, True, False, False), False)
            },
            face_landmarks=None,
        )
        emitted = []

        for index in range(16):
            timestamp = 10.0 + index * 0.05
            with patch("actions.time.monotonic", return_value=timestamp):
                emitted.extend(recognizer.recognize(results))

        self.assertEqual([event.name for event in emitted], ["peace_sign"])

    def test_low_confidence_gesture_is_rejected(self):
        recognizer = ActionRecognizer()
        results = TrackingResults(
            pose_landmarks=None,
            hand_landmarks={"left": make_hand_landmarks((False,) * 4, False)},
            face_landmarks=None,
        )
        with patch.object(
            recognizer, "_classify_finger_gesture", return_value=("fist", 0.73)
        ):
            emitted = []
            for index in range(12):
                with patch(
                    "actions.time.monotonic", return_value=10.0 + index * 0.05
                ):
                    emitted.extend(recognizer.recognize(results))

        self.assertEqual(emitted, [])

    def test_landmark_smoothing_reduces_single_frame_jitter(self):
        recognizer = ActionRecognizer()
        recognizer._smooth("hand:left", [LandmarkPoint(0.5, 0.5)])

        smoothed = recognizer._smooth("hand:left", [LandmarkPoint(0.9, 0.9)])

        self.assertAlmostEqual(smoothed[0].x, 0.5 + 0.42 * 0.4)
        self.assertLess(smoothed[0].x, 0.9)

    def test_actions_respect_one_second_cooldown(self):
        recognizer = ActionRecognizer()
        pose = [LandmarkPoint(0.5, 0.7) for _ in range(33)]
        pose[11] = LandmarkPoint(0.4, 0.5)
        pose[12] = LandmarkPoint(0.6, 0.5)
        pose[15] = LandmarkPoint(0.3, 0.3)
        pose[16] = LandmarkPoint(0.7, 0.7)
        results = TrackingResults(pose, {}, None)
        event_times = []

        timeline = [
            (10.0 + index * 0.05, True) for index in range(10)
        ] + [
            (10.5 + index * 0.05, False) for index in range(10)
        ] + [
            (11.0 + index * 0.05, True) for index in range(12)
        ]
        for timestamp, raised in timeline:
            pose[15] = LandmarkPoint(0.3, 0.3 if raised else 0.7)
            pose[16] = LandmarkPoint(0.7, 0.3 if raised else 0.7)
            with patch("actions.time.monotonic", return_value=timestamp):
                if recognizer.recognize(results):
                    event_times.append(timestamp)

        self.assertEqual(len(event_times), 2)
        self.assertGreaterEqual(event_times[1] - event_times[0], 1.0)


if __name__ == "__main__":
    unittest.main()
