"""Basketball-focused horizontal-to-vertical video cropping."""

import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


OUTPUT_SIZE = (720, 1280)
MAX_TRACKING_DIMENSION = 960


def get_ffmpeg_exe():
    if getattr(sys, "frozen", False):
        executable = os.path.join(
            sys._MEIPASS,
            "imageio_ffmpeg",
            "binaries",
            "ffmpeg-win64-v4.2.2.exe",
        )
        if not os.path.isfile(executable):
            raise RuntimeError("打包的 FFmpeg 文件不存在：{}".format(executable))
        return executable

    try:
        imageio_ffmpeg = __import__("imageio_ffmpeg")
    except ModuleNotFoundError as error:
        if error.name == "imageio_ffmpeg":
            raise RuntimeError(
                "缺少 imageio-ffmpeg；请安装 requirements-product.txt 中的依赖。"
            ) from error
        raise RuntimeError("imageio-ffmpeg 的依赖未能加载。") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def open_video(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法读取视频文件，请确认视频格式有效。")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("无法读取视频的帧率或画面尺寸。")
    if width <= height:
        capture.release()
        raise ValueError("目前仅支持横屏视频，请上传宽度大于高度的视频。")
    return capture, {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def read_frame(path, seconds):
    capture, info = open_video(path)
    try:
        seconds = min(max(float(seconds), 0.0), info["duration"])
        if not math.isfinite(seconds):
            raise ValueError("预览时间点必须是有效数字。")
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("无法读取所选时间点的画面。")
        return frame, info
    finally:
        capture.release()


def detect_shot_boundaries(path, sample_interval=0.25, threshold=0.5):
    capture, info = open_video(path)
    stride = max(int(round(info["fps"] * sample_interval)), 1)
    boundaries = []
    previous_histogram = None
    last_boundary_frame = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                histogram = cv2.calcHist(
                    [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
                )
                cv2.normalize(
                    histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1
                )
                if previous_histogram is not None:
                    distance = cv2.compareHist(
                        previous_histogram,
                        histogram,
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                    minimum_gap = max(int(round(info["fps"] * 0.75)), 1)
                    if (
                        distance >= threshold
                        and frame_index - last_boundary_frame >= minimum_gap
                        and info["frame_count"] - frame_index >= minimum_gap
                    ):
                        boundaries.append(frame_index / info["fps"])
                        last_boundary_frame = frame_index
                previous_histogram = histogram
            frame_index += 1
    finally:
        capture.release()
    return boundaries


def detect_players(frame):
    """Return HOG person proposals; this is a lightweight, non-basketball-specific detector."""
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = detector.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    results = []
    for box, weight in zip(boxes, weights):
        x, y, width, height = [int(value) for value in box]
        if height >= 48 and width >= 18:
            results.append(
                {"box": [x, y, width, height], "confidence": float(weight)}
            )
    return results


def detect_orange_ball(frame):
    """Find a small, roughly round orange region as a weak basketball cue."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([3, 100, 90]), np.array([28, 255, 255]))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12 or area > frame_area * 0.002:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity >= 0.38:
            moments = cv2.moments(contour)
            if moments["m00"]:
                candidates.append(
                    (
                        circularity * area,
                        (
                            int(moments["m10"] / moments["m00"]),
                            int(moments["m01"] / moments["m00"]),
                        ),
                    )
                )
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def suggest_player(frame):
    players = detect_players(frame)
    if not players:
        raise ValueError(
            "未能在该画面中检测到清晰球员。请换一个时间点，或使用手动框选。"
        )

    ball_center = detect_orange_ball(frame)
    frame_center_x = frame.shape[1] / 2.0
    frame_area = frame.shape[0] * frame.shape[1]
    for player in players:
        x, y, width, height = player["box"]
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        area = width * height
        if ball_center is not None:
            dx = (center_x - ball_center[0]) / frame.shape[1]
            dy = (center_y - ball_center[1]) / frame.shape[0]
            player["rank"] = -(dx * dx + dy * dy)
        else:
            center_score = 1.0 - abs(center_x - frame_center_x) / frame_center_x
            area_score = min(area / frame_area * 180.0, 1.0)
            player["rank"] = center_score * 0.65 + area_score * 0.35

    players.sort(key=lambda player: player["rank"], reverse=True)
    return players, ball_center is not None


def crop_left_for_target(center_x, box, frame_width, crop_width):
    left = int(round(center_x - crop_width / 2.0))
    left = min(max(left, 0), frame_width - crop_width)

    x, _, target_width, _ = box
    if target_width <= crop_width:
        left = min(
            max(left, int(math.ceil(x + target_width - crop_width))),
            int(math.floor(x)),
        )
        left = min(max(left, 0), frame_width - crop_width)
    return int(left)


def _create_crop_smoother(fps, initial_center):
    frame_duration = 1.0 / fps
    smoother = cv2.KalmanFilter(2, 1, 0, cv2.CV_32F)
    smoother.transitionMatrix = np.array(
        [[1.0, frame_duration], [0.0, 1.0]], dtype=np.float32
    )
    smoother.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
    acceleration_variance = 62500.0
    smoother.processNoiseCov = acceleration_variance * np.array(
        [
            [frame_duration ** 4 / 4.0, frame_duration ** 3 / 2.0],
            [frame_duration ** 3 / 2.0, frame_duration ** 2],
        ],
        dtype=np.float32,
    )
    smoother.measurementNoiseCov = np.array([[36.0]], dtype=np.float32)
    smoother.errorCovPost = np.array(
        [[36.0, 0.0], [0.0, 10000.0]], dtype=np.float32
    )
    smoother.statePost = np.array(
        [[float(initial_center)], [0.0]], dtype=np.float32
    )
    return smoother


def _smooth_crop_center(smoother, measured_center):
    smoother.predict()
    measurement = np.array([[float(measured_center)]], dtype=np.float32)
    return float(smoother.correct(measurement)[0, 0])


def _inside_box_features(gray, box, maximum=80):
    x, y, width, height = [int(value) for value in box]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(gray.shape[1], x + width)
    y1 = min(gray.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maximum,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=7,
    )


def _tracking_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(
        1.0,
        MAX_TRACKING_DIMENSION / float(max(frame.shape[1], frame.shape[0])),
    )
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (
                max(int(round(frame.shape[1] * scale)), 1),
                max(int(round(frame.shape[0] * scale)), 1),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _track_box(previous_gray, current_gray, points, box):
    if points is None or len(points) < 3:
        return box, _inside_box_features(current_gray, box), 0.0

    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return box, _inside_box_features(current_gray, box), 0.0

    status = status.reshape(-1).astype(bool)
    if errors is not None:
        status &= errors.reshape(-1) < 40
    old = points.reshape(-1, 2)[status]
    new = next_points.reshape(-1, 2)[status]
    confidence = len(new) / float(len(points))
    if len(new) >= 3:
        displacement = np.median(new - old, axis=0)
        x, y, width, height = box
        box = (
            x + float(displacement[0]),
            y + float(displacement[1]),
            width,
            height,
        )
        points = new.reshape(-1, 1, 2).astype(np.float32)
        refreshed = _inside_box_features(current_gray, box)
        if refreshed is not None:
            points = np.concatenate((points, refreshed), axis=0)
            points = points[:80]
    else:
        points = _inside_box_features(current_gray, box)
    return box, points, confidence


def track_target_path(
    input_path, start_seconds, target_box, end_seconds, progress=None
):
    capture, info = open_video(input_path)
    try:
        start_frame = min(
            max(int(round(float(start_seconds) * info["fps"])), 0),
            info["frame_count"] - 1,
        )
        end_frame = min(
            max(int(round(float(end_seconds) * info["fps"])), 0),
            info["frame_count"],
        )
        if end_frame <= start_frame:
            raise ValueError("片段结束时间必须晚于目标开始时间。")

        frame_height, frame_width = info["height"], info["width"]
        coordinates = [float(value) for value in target_box]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("主体框坐标必须是有效数字。")
        x, y, box_width, box_height = [
            int(round(value)) for value in coordinates
        ]
        if box_width < 8 or box_height < 8:
            raise ValueError("请框选有效的目标区域后再开始跟踪。")
        if (
            x < 0
            or y < 0
            or x + box_width > frame_width
            or y + box_height > frame_height
        ):
            raise ValueError("主体框超出画面范围，请重新框选。")

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("无法读取目标起始画面。")

        previous_gray, scale = _tracking_gray(frame)
        tracking_box = tuple(
            float(value) * scale
            for value in (x, y, box_width, box_height)
        )
        tracking_width = previous_gray.shape[1]
        tracking_height = previous_gray.shape[0]
        points = _inside_box_features(previous_gray, tracking_box)
        boxes = [[float(x), float(y), float(box_width), float(box_height)]]
        confidences = [1.0]
        low_confidence_frames = 0
        low_confidence_run = 0
        longest_low_confidence_run = 0

        total_frames = end_frame - start_frame
        for frame_index in range(start_frame + 1, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            current_gray, _ = _tracking_gray(frame)
            tracking_box, points, confidence = _track_box(
                previous_gray, current_gray, points, tracking_box
            )
            box_x, box_y, box_width, box_height = tracking_box
            tracking_box = (
                min(max(box_x, 0.0), tracking_width - box_width),
                min(max(box_y, 0.0), tracking_height - box_height),
                box_width,
                box_height,
            )
            boxes.append([float(value) / scale for value in tracking_box])
            confidences.append(float(confidence))
            if confidence < 0.25:
                low_confidence_frames += 1
                low_confidence_run += 1
                longest_low_confidence_run = max(
                    longest_low_confidence_run, low_confidence_run
                )
            else:
                low_confidence_run = 0
            previous_gray = current_gray
            if progress is not None and (
                len(boxes) % 10 == 0 or len(boxes) == total_frames
            ):
                progress(len(boxes) / float(total_frames))

        if len(boxes) < 2:
            raise ValueError("片段中没有足够帧可供跟踪。")

        warning = (
            "连续多帧跟踪特征不足，目标框可能已偏离。"
            if longest_low_confidence_run >= max(int(round(info["fps"] * 0.5)), 1)
            else ""
        )
        return {
            "start_frame": start_frame,
            "fps": info["fps"],
            "boxes": boxes,
            "confidences": confidences,
            "low_confidence_frames": low_confidence_frames,
            "longest_low_confidence_frames": longest_low_confidence_run,
            "tracking_warning": warning,
        }
    finally:
        capture.release()


def render_vertical(
    input_path,
    output_path,
    start_seconds,
    target_box,
    progress=None,
    end_seconds=None,
    tracking=None,
):
    """Track a selected player and render a 9:16 crop within the requested time range."""
    capture, info = open_video(input_path)
    start_frame = min(
        max(int(round(float(start_seconds) * info["fps"])), 0),
        info["frame_count"] - 1,
    )
    if end_seconds is None:
        end_frame = info["frame_count"]
    else:
        end_frame = min(
            max(int(round(float(end_seconds) * info["fps"])), 0),
            info["frame_count"],
        )
    if end_frame <= start_frame:
        capture.release()
        raise ValueError("片段结束时间必须晚于开始时间。")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError("无法读取主体选定位置的画面。")

    frame_height, frame_width = first_frame.shape[:2]
    coordinates = [float(value) for value in target_box]
    if not all(math.isfinite(value) for value in coordinates):
        capture.release()
        raise ValueError("主体框坐标必须是有效数字。")
    x, y, box_width, box_height = [int(round(value)) for value in coordinates]
    if box_width < 8 or box_height < 8:
        capture.release()
        raise ValueError("请在球员周围拖动鼠标，框选一个有效的主体区域。")
    if x < 0 or y < 0 or x + box_width > frame_width or y + box_height > frame_height:
        capture.release()
        raise ValueError("主体框超出画面范围，请重新框选。")

    crop_width = int(round(frame_height * 9.0 / 16.0))
    if crop_width > frame_width:
        capture.release()
        raise ValueError("视频画面过窄，无法生成 9:16 裁剪。")

    output_width, output_height = OUTPUT_SIZE
    temporary_video = tempfile.NamedTemporaryFile(
        prefix="basketball_video_", suffix=".mp4", delete=False
    )
    temporary_video.close()
    writer = cv2.VideoWriter(
        temporary_video.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        os.unlink(temporary_video.name)
        raise RuntimeError("无法创建 MP4 视频，请检查 OpenCV 视频编码器。")

    previous_gray, tracking_scale = _tracking_gray(first_frame)
    box = (float(x), float(y), float(box_width), float(box_height))
    tracking_box = tuple(value * tracking_scale for value in box)
    tracking_width = previous_gray.shape[1]
    tracking_height = previous_gray.shape[0]
    points = _inside_box_features(previous_gray, tracking_box)
    crop_smoother = _create_crop_smoother(
        info["fps"], x + box_width / 2.0
    )
    remaining = end_frame - start_frame
    written = 0
    low_confidence_frames = 0
    low_confidence_run = 0
    longest_low_confidence_run = 0
    try:
        frame = first_frame
        while written < remaining:
            current_gray, _ = _tracking_gray(frame)
            if tracking is not None:
                tracked_boxes = tracking.get("boxes", [])
                if written < len(tracked_boxes) and tracked_boxes[written] is not None:
                    box = tuple(float(value) for value in tracked_boxes[written])
                found = tracking.get("found")
                if found is not None and (
                    written >= len(found) or not found[written]
                ):
                    low_confidence_frames += 1
                    low_confidence_run += 1
                    longest_low_confidence_run = max(
                        longest_low_confidence_run, low_confidence_run
                    )
                else:
                    low_confidence_run = 0
            elif written:
                tracking_box, points, confidence = _track_box(
                    previous_gray, current_gray, points, tracking_box
                )
                track_x, track_y, track_width, track_height = tracking_box
                tracking_box = (
                    min(max(track_x, 0.0), tracking_width - track_width),
                    min(max(track_y, 0.0), tracking_height - track_height),
                    track_width,
                    track_height,
                )
                box = tuple(value / tracking_scale for value in tracking_box)
                if confidence < 0.25:
                    low_confidence_frames += 1
                    low_confidence_run += 1
                    longest_low_confidence_run = max(
                        longest_low_confidence_run, low_confidence_run
                    )
                else:
                    low_confidence_run = 0
            box_x, box_y, target_width, target_height = box
            box_x = min(max(box_x, 0.0), frame_width - target_width)
            box_y = min(max(box_y, 0.0), frame_height - target_height)
            box = (box_x, box_y, target_width, target_height)

            target_center_x = box_x + target_width / 2.0
            smoothed_center_x = _smooth_crop_center(
                crop_smoother, target_center_x
            )
            crop_left = crop_left_for_target(
                smoothed_center_x, box, frame_width, crop_width
            )
            crop = frame[:, crop_left : crop_left + crop_width]
            writer.write(cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_AREA))

            written += 1
            if progress is not None and (written % 10 == 0 or written == remaining):
                progress(written / float(remaining))
            previous_gray = current_gray
            if written < remaining:
                ok, frame = capture.read()
                if not ok:
                    break
    except Exception:
        capture.release()
        writer.release()
        if os.path.exists(temporary_video.name):
            os.unlink(temporary_video.name)
        raise
    else:
        capture.release()
        writer.release()

    if written == 0:
        os.unlink(temporary_video.name)
        raise RuntimeError("没有生成任何视频帧。")

    try:
        ffmpeg_path = get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            temporary_video.name,
            "-ss",
            "{:.6f}".format(start_frame / info["fps"]),
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError("视频音频合并失败：{}".format(message))
    finally:
        if os.path.exists(temporary_video.name):
            os.unlink(temporary_video.name)

    return {
        "frames": written,
        "duration": written / info["fps"],
        "width": output_width,
        "height": output_height,
        "tracking_warning": (
            (
                "ByteTrack 连续多帧未检测到所选球员，裁剪位置可能偏离目标。"
                if tracking is not None
                else "连续多帧跟踪特征不足，裁剪位置可能偏离目标。"
            )
            if longest_low_confidence_run >= max(int(round(info["fps"] * 0.5)), 1)
            else ""
        ),
        "low_confidence_frames": low_confidence_frames,
    }
