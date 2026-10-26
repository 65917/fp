import os
import tempfile
import unittest
from unittest.mock import patch

from basketball_video_cropper.crop_basketball_video import (
    _default_output_path,
    _parse_args,
    _reserve_unique_output_path,
)


class OutputPathTests(unittest.TestCase):
    def test_default_output_groups_by_source_video(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "basketball_video_cropper.crop_basketball_video._tool_directory",
                return_value=folder,
            ):
                output = _default_output_path(r"D:\games\final.mp4")

            self.assertEqual(
                output,
                os.path.join(
                    folder,
                    "cropped_videos",
                    "final",
                    "final_vertical.mp4",
                ),
            )
            self.assertTrue(os.path.isdir(os.path.dirname(output)))

    def test_reserves_requested_name_without_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            requested = os.path.join(folder, "clip.mp4")
            with open(requested, "wb") as output_file:
                output_file.write(b"original")

            reserved = _reserve_unique_output_path(requested)

            self.assertEqual(reserved, os.path.join(folder, "clip_2.mp4"))
            self.assertTrue(os.path.isfile(reserved))
            with open(requested, "rb") as output_file:
                self.assertEqual(output_file.read(), b"original")

    def test_reserves_next_available_number(self):
        with tempfile.TemporaryDirectory() as folder:
            requested = os.path.join(folder, "clip.mp4")
            for name in ("clip.mp4", "clip_2.mp4"):
                with open(os.path.join(folder, name), "wb"):
                    pass

            reserved = _reserve_unique_output_path(requested)

            self.assertEqual(reserved, os.path.join(folder, "clip_3.mp4"))
            self.assertTrue(os.path.isfile(reserved))


class TrackerArgumentTests(unittest.TestCase):
    def test_default_tracker_matches_flow_executable(self):
        with patch(
            "basketball_video_cropper.crop_basketball_video.sys.executable",
            r"C:\tools\BasketballVideoCropper-Flow.exe",
        ):
            args = _parse_args(["game.mp4"])

        self.assertEqual(args.tracker, "flow")

    def test_default_tracker_is_yolo_for_standard_executable(self):
        with patch(
            "basketball_video_cropper.crop_basketball_video.sys.executable",
            r"C:\tools\BasketballVideoCropper-YOLO.exe",
        ):
            args = _parse_args(["game.mp4"])

        self.assertEqual(args.tracker, "yolo")


if __name__ == "__main__":
    unittest.main()
