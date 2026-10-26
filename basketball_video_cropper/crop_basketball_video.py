"""Crop a local basketball video to a tracked 9:16 MP4."""

import argparse
import math
import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from basketball_cropper import get_ffmpeg_exe, open_video, read_frame, render_vertical
from basketball_yolo_worker import track_target


def _tool_directory():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


def _default_output_path(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_directory = os.path.join(
        _tool_directory(), "cropped_videos", video_name
    )
    os.makedirs(output_directory, exist_ok=True)
    filename = video_name + "_vertical.mp4"
    return os.path.join(output_directory, filename)


def _reserve_unique_output_path(output_path):
    directory, filename = os.path.split(output_path)
    name, extension = os.path.splitext(filename)
    candidate = output_path
    index = 2
    while True:
        try:
            descriptor = os.open(
                candidate,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            candidate = os.path.join(
                directory, "{}_{}{}".format(name, index, extension)
            )
            index += 1
        else:
            os.close(descriptor)
            return candidate


def _parse_args(argv=None):
    default_tracker = (
        "flow"
        if "flow" in os.path.basename(sys.executable).lower()
        else "yolo"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Choose a player in a video frame, then export a tracked 9:16 crop."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="path to the input video; omit it to choose a video in a file dialog",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="selection time in seconds; export starts from this frame (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        help="end time in seconds; defaults to the end of the video",
    )
    parser.add_argument(
        "--output",
        help="output MP4 path; defaults to a per-video folder in cropped_videos",
    )
    parser.add_argument(
        "--tracker",
        choices=("yolo", "flow"),
        default=default_tracker,
        help="tracking backend (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _choose_video_and_output():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "无法打开文件选择窗口；请将视频文件拖到程序上，或从命令行传入路径。"
        ) from error

    root = tk.Tk()
    root.withdraw()
    try:
        video_path = filedialog.askopenfilename(
            title="选择要裁剪的篮球视频",
            filetypes=[
                ("视频文件", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if not video_path:
            return None, None
        default_output = _default_output_path(video_path)
        output_path = filedialog.asksaveasfilename(
            title="选择竖屏视频的保存位置",
            initialfile=os.path.basename(default_output),
            initialdir=os.path.dirname(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4")],
            confirmoverwrite=False,
        )
        if not output_path:
            return None, None
        return video_path, output_path
    finally:
        root.destroy()


def main(argv=None):
    args = _parse_args(argv)
    if args.video is None:
        video_path, selected_output = _choose_video_and_output()
        if video_path is None:
            print("已取消。")
            return
        if args.output is None:
            args.output = selected_output
    else:
        video_path = args.video

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise ValueError("找不到输入视频：{}".format(video_path))
    if not math.isfinite(args.start) or args.start < 0:
        raise ValueError("--start 必须是大于或等于 0 的有效秒数。")
    if args.end is not None and (
        not math.isfinite(args.end) or args.end <= args.start
    ):
        raise ValueError("--end 必须是大于 --start 的有效秒数。")

    capture, info = open_video(video_path)
    try:
        if args.start >= info["duration"]:
            raise ValueError(
                "--start 超出视频时长（{:.2f} 秒）。".format(info["duration"])
            )
        if args.end is not None and args.end > info["duration"]:
            raise ValueError(
                "--end 超出视频时长（{:.2f} 秒）。".format(info["duration"])
            )
    finally:
        capture.release()

    output_path = os.path.abspath(
        args.output
        or _default_output_path(video_path)
    )
    if os.path.normcase(output_path) == os.path.normcase(video_path):
        raise ValueError("输出文件不能覆盖输入视频。")
    if not os.path.isdir(os.path.dirname(output_path)):
        raise ValueError("输出目录不存在：{}".format(os.path.dirname(output_path)))

    get_ffmpeg_exe()

    frame, _ = read_frame(video_path, args.start)
    try:
        roi = cv2.selectROI(
            "框选要跟踪的球员，然后按 Enter/Space 确认，按 C 取消",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
    except cv2.error as error:
        raise RuntimeError(
            "无法打开视频选框窗口；请在带图形界面的本机终端运行此脚本。"
        ) from error
    finally:
        cv2.destroyAllWindows()

    x, y, width, height = [int(value) for value in roi]
    if width < 8 or height < 8:
        raise ValueError("没有选择有效的球员区域，已取消导出。")

    output_path = _reserve_unique_output_path(output_path)
    print(
        "开始跟踪并导出（{}）：{}".format(
            "YOLOv8 + ByteTrack" if args.tracker == "yolo" else "光流",
            output_path,
        )
    )

    def report_tracking_progress(value):
        print(
            "\rYOLO 跟踪进度：{:5.1f}%".format(value * 100),
            end="",
            flush=True,
        )

    def report_render_progress(value):
        print(
            "\r视频渲染进度：{:5.1f}%".format(value * 100),
            end="",
            flush=True,
        )

    succeeded = False
    try:
        tracking = None
        if args.tracker == "yolo":
            tracking = track_target(
                video_path,
                args.start,
                args.end if args.end is not None else info["duration"],
                [x, y, width, height],
                progress=report_tracking_progress,
            )
            print("\n开始使用 YOLOv8 + ByteTrack 渲染竖屏视频。")
        else:
            print("\n开始使用光流渲染竖屏视频。")
        result = render_vertical(
            video_path,
            output_path,
            args.start,
            [x, y, width, height],
            progress=report_render_progress,
            end_seconds=args.end,
            tracking=tracking,
        )
        succeeded = True
    finally:
        if not succeeded and os.path.exists(output_path):
            os.unlink(output_path)

    print("\n完成：{}（{} 帧，{:.2f} 秒）".format(
        output_path, result["frames"], result["duration"]
    ))
    if result["tracking_warning"]:
        print("跟踪提醒：{}".format(result["tracking_warning"]), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print("错误：{}".format(error), file=sys.stderr)
        sys.exit(1)
