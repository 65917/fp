import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import basketball_web
from basketball_yolo_worker import _box_iou, _select_track_id, _track


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, detections):
        self.xyxy = FakeTensor(
            [
                [x, y, x + width, y + height]
                for x, y, width, height, _, _ in detections
            ]
        )
        self.conf = FakeTensor([confidence for _, _, _, _, confidence, _ in detections])
        identifiers = [identifier for _, _, _, _, _, identifier in detections]
        self.id = FakeTensor(identifiers) if identifiers else None

    def __len__(self):
        return len(self.conf.values)


class FakeModel:
    def __init__(self, detections_by_frame):
        self.detections_by_frame = detections_by_frame
        self.index = 0

    def track(self, frame, **kwargs):
        detections = self.detections_by_frame[self.index]
        self.index += 1
        return [type("Result", (), {"boxes": FakeBoxes(detections)})()]


class ByteTrackAssociationTests(unittest.TestCase):
    def test_box_iou_returns_intersection_over_union(self):
        self.assertAlmostEqual(
            _box_iou([0, 0, 10, 10], [5, 0, 10, 10]),
            1.0 / 3.0,
        )

    def test_initial_box_selects_overlapping_byte_track_id(self):
        detections = [
            {"box": [100, 100, 40, 80], "track_id": 3},
            {"box": [12, 12, 48, 78], "track_id": 8},
        ]

        selected_id = _select_track_id([10, 10, 50, 80], detections)

        self.assertEqual(selected_id, 8)

    def test_rejects_unmatched_initial_box(self):
        detections = [{"box": [100, 100, 40, 80], "track_id": 3}]

        with self.assertRaisesRegex(RuntimeError, "不匹配"):
            _select_track_id([10, 10, 50, 80], detections)

    def test_tracking_keeps_selected_id_and_marks_missing_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "synthetic.avi")
            writer = cv2.VideoWriter(
                video, cv2.VideoWriter_fourcc(*"MJPG"), 5, (120, 80)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((80, 120, 3), dtype=np.uint8))
            writer.release()

            model = FakeModel(
                [
                    [(10, 10, 30, 50, 0.9, 4), (70, 10, 25, 45, 0.8, 9)],
                    [(12, 10, 30, 50, 0.85, 4), (69, 10, 25, 45, 0.8, 9)],
                    [],
                    [(16, 10, 30, 50, 0.88, 4), (67, 10, 25, 45, 0.8, 9)],
                ]
            )
            payload = {
                "video_path": video,
                "start_seconds": 0,
                "end_seconds": 0.8,
                "box": [10, 10, 30, 50],
            }
            progress = []
            with patch("basketball_yolo_worker._load_model", return_value=model):
                with patch("basketball_yolo_worker._emit"):
                    tracking = _track(payload, progress=progress.append)

            self.assertEqual(tracking["tracker_id"], 4)
            self.assertEqual(len(tracking["boxes"]), 4)
            self.assertEqual(tracking["found"], [True, True, False, True])
            self.assertEqual(tracking["boxes"][0], [10.0, 10.0, 30.0, 50.0])
            self.assertEqual(tracking["boxes"][2], tracking["boxes"][1])
            self.assertEqual(tracking["low_confidence_frames"], 1)
            self.assertEqual(progress, [0.25, 1.0])


class YoloWorkerProcessTests(unittest.TestCase):
    def test_worker_runner_reads_result_and_progress_events(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = os.path.join(folder, "fake_worker.py")
            with open(worker, "w") as output:
                output.write(
                    "import json, sys\n"
                    "payload = json.load(sys.stdin)\n"
                    "print(json.dumps({'event':'progress','progress':0.5}))\n"
                    "print(json.dumps({'event':'result','result':payload}))\n"
                )
            progress = []
            with patch("basketball_web._yolo_interpreter", return_value=sys.executable):
                with patch("basketball_web.YOLO_WORKER", worker):
                    result = basketball_web._run_yolo_worker(
                        {"mode": "test"}, progress=progress.append
                    )

        self.assertEqual(result, {"mode": "test"})
        self.assertEqual(progress, [0.5])


if __name__ == "__main__":
    unittest.main()
