import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from basketball_cropper import (
    _create_crop_smoother,
    _smooth_crop_center,
    crop_left_for_target,
    detect_shot_boundaries,
    render_vertical,
    suggest_player,
    track_target_path,
)


class CropGeometryTests(unittest.TestCase):
    def test_crop_is_clamped_at_left_edge(self):
        self.assertEqual(crop_left_for_target(15, (5, 10, 30, 80), 1920, 608), 0)

    def test_crop_keeps_target_inside_when_possible(self):
        left = crop_left_for_target(1000, (960, 100, 80, 200), 1920, 608)
        self.assertLessEqual(left, 960)
        self.assertGreaterEqual(left + 608, 1040)

    def test_crop_left_is_an_integer_for_tracked_float_boxes(self):
        left = crop_left_for_target(1000.5, (960.25, 100.0, 80.5, 200.0), 1920, 608)

        self.assertIsInstance(left, int)
        self.assertLessEqual(left, 960.25)
        self.assertGreaterEqual(left + 608, 1040.75)

    def test_crop_is_clamped_at_right_edge(self):
        self.assertEqual(
            crop_left_for_target(1900, (1850, 10, 50, 100), 1920, 608), 1312
        )


class CropSmoothingTests(unittest.TestCase):
    def test_kalman_filter_reduces_crop_center_jitter(self):
        measurements = [400.0 + (-1) ** index * 15.0 for index in range(60)]
        smoother = _create_crop_smoother(30.0, measurements[0])
        smoothed = [
            _smooth_crop_center(smoother, value) for value in measurements
        ]

        self.assertLess(
            float(np.std(smoothed[10:])),
            float(np.std(measurements[10:])),
        )

    def test_kalman_filter_follows_steady_target_motion(self):
        measurements = [200.0 + index * 4.0 for index in range(30)]
        smoother = _create_crop_smoother(30.0, measurements[0])
        smoothed = [
            _smooth_crop_center(smoother, value) for value in measurements
        ]

        self.assertGreater(smoothed[-1], measurements[-1] - 12.0)


class CandidateSuggestionTests(unittest.TestCase):
    @patch("basketball_cropper.detect_orange_ball", return_value=(55, 70))
    @patch(
        "basketball_cropper.detect_players",
        return_value=[
            {"box": [250, 20, 40, 100], "confidence": 0.8},
            {"box": [30, 20, 40, 100], "confidence": 0.7},
        ],
    )
    def test_candidates_are_ranked_near_suspected_ball(self, detect_players, detect_ball):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        players, ball_cue = suggest_player(frame)

        self.assertTrue(ball_cue)
        self.assertEqual(players[0]["box"], [30, 20, 40, 100])
        detect_players.assert_called_once_with(frame)
        detect_ball.assert_called_once_with(frame)


class ShotDetectionTests(unittest.TestCase):
    def test_detects_a_hard_cut_between_distinct_scenes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "two_scenes.avi")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90)
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                color = (255, 0, 0) if index < 10 else (0, 255, 255)
                writer.write(np.full((90, 160, 3), color, dtype=np.uint8))
            writer.release()

            boundaries = detect_shot_boundaries(source, sample_interval=0.1)

            self.assertEqual(len(boundaries), 1)
            self.assertAlmostEqual(boundaries[0], 1.0, delta=0.1)


class TargetTrackingTests(unittest.TestCase):
    def test_selected_target_is_tracked_across_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "moving_target.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                x = 40 + index * 2
                cv2.rectangle(frame, (x, 30), (x + 55, 145), (220, 220, 220), -1)
                for row in range(35, 140, 12):
                    cv2.line(frame, (x + 5, row), (x + 50, row), (20, 20, 20), 2)
                cv2.rectangle(frame, (x + 10, 55), (x + 45, 95), (0, 0, 255), -1)
                writer.write(frame)
            writer.release()

            tracking = track_target_path(source, 0, [40, 30, 55, 115], 2.0)

            self.assertEqual(len(tracking["boxes"]), 20)
            self.assertEqual(tracking["boxes"][0], [40.0, 30.0, 55.0, 115.0])
            self.assertGreater(tracking["boxes"][-1][0], 55)
            self.assertEqual(len(tracking["confidences"]), 20)


class VideoExportTests(unittest.TestCase):
    def test_render_reuses_external_tracking_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "input.mp4")
            output = os.path.join(folder, "output.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for index in range(10):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.rectangle(
                    frame,
                    (80 + index, 30),
                    (130 + index, 145),
                    (255, 255, 255),
                    -1,
                )
                writer.write(frame)
            writer.release()

            boxes = [[80.0 + index, 30.0, 50.0, 115.0] for index in range(10)]
            tracking = {
                "boxes": boxes,
                "found": [True] * 4 + [False] * 6,
                "confidences": [0.9] * 4 + [0.0] * 6,
            }
            with patch("basketball_cropper._track_box", side_effect=AssertionError):
                result = render_vertical(
                    source,
                    output,
                    0,
                    boxes[0],
                    tracking=tracking,
                )

            self.assertEqual(result["frames"], 10)
            self.assertIn("ByteTrack", result["tracking_warning"])
            self.assertEqual(result["low_confidence_frames"], 6)

    def test_render_produces_portrait_video_from_selected_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "input.mp4")
            output = os.path.join(folder, "output.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for index in range(12):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                x = 90 + index * 2
                cv2.rectangle(frame, (x, 35), (x + 35, 150), (255, 255, 255), -1)
                cv2.line(frame, (x, 35), (x + 35, 150), (0, 0, 255), 3)
                writer.write(frame)
            writer.release()

            result = render_vertical(source, output, 0, [90, 35, 35, 115])

            capture = cv2.VideoCapture(output)
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(
                    (
                        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ),
                    (720, 1280),
                )
                self.assertEqual(result["frames"], 12)
                self.assertIn("tracking_warning", result)
                self.assertIn("low_confidence_frames", result)
                self.assertGreater(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0)
            finally:
                capture.release()

    def test_render_stops_at_selected_shot_end(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "input.mp4")
            output = os.path.join(folder, "segment.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
            writer.release()

            result = render_vertical(
                source, output, 0, [90, 35, 35, 115], end_seconds=0.8
            )

            self.assertEqual(result["frames"], 8)
            self.assertAlmostEqual(result["duration"], 0.8)


if __name__ == "__main__":
    unittest.main()
