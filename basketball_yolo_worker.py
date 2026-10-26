"""Optional YOLOv8 person detection and ByteTrack worker."""

import json
import math
import os
import sys


def _emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _load_model():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO backend dependencies are missing; install requirements-yolo.txt "
            "in the configured Python environment."
        ) from error
    model_path = os.environ.get("BASKETBALL_YOLO_MODEL")
    if not model_path:
        if getattr(sys, "frozen", False):
            model_path = os.path.join(sys._MEIPASS, "models", "yolov8n.pt")
        else:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"
            )
    return YOLO(model_path)


def _frame_boxes(result, include_ids=False):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    identifiers = None
    if include_ids and boxes.id is not None:
        identifiers = boxes.id.int().cpu().tolist()
    output = []
    for index, values in enumerate(coordinates):
        x1, y1, x2, y2 = values
        entry = {
            "box": [
                float(x1),
                float(y1),
                float(max(x2 - x1, 1)),
                float(max(y2 - y1, 1)),
            ],
            "confidence": float(confidences[index]),
        }
        if identifiers is not None:
            entry["track_id"] = int(identifiers[index])
        output.append(entry)
    return output


def _detect(payload):
    import cv2

    capture = cv2.VideoCapture(payload["video_path"])
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("无法读取视频以执行 YOLOv8 检测。")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_index = max(int(round(float(payload["time"]) * fps)), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("无法读取 YOLOv8 检测所需的当前帧。")
        model = _load_model()
        results = model.predict(
            frame, classes=[0], conf=0.25, verbose=False
        )
        players = _frame_boxes(results[0])
        players.sort(key=lambda player: player["confidence"], reverse=True)
        if not players:
            raise RuntimeError(
                "YOLOv8 未在当前帧检测到球员，请换一个更清晰的时间点。"
            )
        return {"players": players, "detector": "YOLOv8"}
    finally:
        capture.release()


def _box_iou(first, second):
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def _select_track_id(initial_box, detections, minimum_iou=0.1):
    associated = [
        (_box_iou(initial_box, detection["box"]), detection)
        for detection in detections
        if "track_id" in detection
    ]
    if not associated:
        raise RuntimeError(
            "ByteTrack 未能在起始帧为球员分配 ID。请使用 YOLO 候选框重新选择目标。"
        )
    overlap, match = max(associated, key=lambda item: item[0])
    if overlap < minimum_iou:
        raise RuntimeError(
            "所选框与起始帧的 YOLO 检测不匹配。请先使用 YOLO 候选框选择球员。"
        )
    return match["track_id"]


def _track(payload, progress=None):
    import cv2

    capture = cv2.VideoCapture(payload["video_path"])
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("无法读取视频以执行 YOLOv8 跟踪。")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("无法读取视频帧率或帧数。")

    start_frame = max(int(round(float(payload["start_seconds"]) * fps)), 0)
    end_frame = min(
        max(int(round(float(payload["end_seconds"]) * fps)), 0), frame_count
    )
    if end_frame <= start_frame:
        capture.release()
        raise RuntimeError("镜头片段中没有可跟踪的帧。")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    model = _load_model()
    initial_box = [float(value) for value in payload["box"]]
    if not all(math.isfinite(value) for value in initial_box):
        capture.release()
        raise RuntimeError("初始目标框包含无效坐标。")

    selected_id = None
    boxes = []
    confidences = []
    found = []
    last_box = initial_box
    low_confidence_run = 0
    longest_low_confidence_run = 0
    missing_frames = 0
    total_frames = end_frame - start_frame
    try:
        for offset in range(total_frames):
            ok, frame = capture.read()
            if not ok:
                break
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.15,
                verbose=False,
            )
            detections = _frame_boxes(results[0], include_ids=True)
            if selected_id is None:
                selected_id = _select_track_id(initial_box, detections)

            match = next(
                (
                    detection
                    for detection in detections
                    if detection.get("track_id") == selected_id
                ),
                None,
            )
            if match is None:
                boxes.append(list(last_box))
                confidences.append(0.0)
                found.append(False)
                missing_frames += 1
                low_confidence_run += 1
                longest_low_confidence_run = max(
                    longest_low_confidence_run, low_confidence_run
                )
            else:
                last_box = match["box"]
                boxes.append(list(last_box))
                confidences.append(match["confidence"])
                found.append(True)
                low_confidence_run = 0
            if offset % 10 == 0 or offset + 1 == total_frames:
                value = (offset + 1) / float(total_frames)
                if progress is None:
                    _emit({"event": "progress", "progress": value})
                else:
                    progress(value)
    finally:
        capture.release()

    if not boxes:
        raise RuntimeError("YOLOv8/ByteTrack 没有生成跟踪帧。")
    warning = ""
    if longest_low_confidence_run >= max(int(round(fps * 0.5)), 1):
        warning = "ByteTrack 连续多帧未检测到所选球员，目标位置可能已失效。"
    return {
        "start_frame": start_frame,
        "fps": fps,
        "boxes": boxes,
        "confidences": confidences,
        "found": found,
        "low_confidence_frames": missing_frames,
        "longest_low_confidence_frames": longest_low_confidence_run,
        "tracking_warning": warning,
        "tracker_id": selected_id,
        "tracker_name": "YOLOv8 + ByteTrack",
    }


def track_target(
    video_path, start_seconds, end_seconds, box, progress=None
):
    return _track(
        {
            "video_path": video_path,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "box": box,
        },
        progress=progress,
    )


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        try:
            from ultralytics import YOLO  # noqa: F401
            import torch

            cuda_available = torch.cuda.is_available()
            model_path = os.environ.get("BASKETBALL_YOLO_MODEL", "yolov8n.pt")
            _emit(
                {
                    "event": "ready",
                    "device": (
                        torch.cuda.get_device_name(0) if cuda_available else "CPU"
                    ),
                    "cuda_available": cuda_available,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "model_ready": os.path.isfile(model_path),
                }
            )
            return 0
        except Exception as error:
            _emit({"event": "error", "error": str(error)})
            return 1

    try:
        payload = json.load(sys.stdin)
        if payload.get("mode") == "detect":
            result = _detect(payload)
        elif payload.get("mode") == "track":
            result = _track(payload)
        else:
            raise RuntimeError("未知 YOLO worker 操作。")
        _emit({"event": "result", "result": result})
        return 0
    except Exception as error:
        _emit({"event": "error", "error": str(error)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
