"""Re-run SmartVidCrop on selected RetargetVid videos and score its crop windows."""

import argparse
import json
import os
import re
import statistics
import time

import smartVidCrop


ROOT = os.path.abspath(os.path.dirname(__file__))
ANNOTATORS = range(1, 7)
RATIO_NAMES = {"1:3": "1-3", "3:1": "3-1"}


def read_annotations(video_id, ratio):
    annotations = []
    filename = "{}_{}.txt".format(video_id, RATIO_NAMES[ratio])
    for annotator in ANNOTATORS:
        path = os.path.join(
            ROOT, "annotations", "annotator_{}".format(annotator), filename
        )
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing RetargetVid annotation: {}".format(path))
        with open(path, "r") as source:
            annotations.append(
                [[int(value) for value in line.split(",")] for line in source if line.strip()]
            )
    frame_counts = {len(items) for items in annotations}
    if len(frame_counts) != 1:
        raise ValueError("Annotators have different frame counts for {}.".format(filename))
    return annotations


def evaluate_boxes(boxes, annotations, ratio):
    user_scores = []
    for user_annotations in annotations:
        if len(user_annotations) != len(boxes):
            raise ValueError(
                "Frame-count mismatch: algorithm produced {}, annotation contains {}."
                .format(len(boxes), len(user_annotations))
            )
        frame_scores = []
        for box, annotation in zip(boxes, user_annotations):
            predicted = [int(value) for value in box]
            if ratio == "1:3":
                displacement = annotation[0]
                ground_truth = [displacement, 0, displacement + 120, 360]
                predicted[2] = predicted[0] + 120
                predicted[3] = 360
            else:
                displacement = annotation[1]
                ground_truth = [0, displacement, 640, displacement + 214]
                predicted[2] = 640
                predicted[3] = predicted[1] + 214
            frame_scores.append(
                smartVidCrop.bb_intersection_over_union(ground_truth, predicted)
            )
        user_scores.append(statistics.mean(frame_scores))
    return {
        "per_annotator_iou": user_scores,
        "mean_annotator_iou": statistics.mean(user_scores),
        "worst_annotator_iou": min(user_scores),
        "best_annotator_iou": max(user_scores),
    }


def load_reference_boxes(video_id, ratio, result_set):
    path = os.path.join(
        ROOT,
        "results",
        result_set,
        "{}_{}.txt".format(video_id, RATIO_NAMES[ratio]),
    )
    if not os.path.isfile(path):
        return None
    with open(path, "r") as source:
        return [[int(value) for value in line.split(",")] for line in source if line.strip()]


def compare_box_sequences(boxes, reference):
    if reference is None:
        return None
    if len(boxes) != len(reference):
        return {
            "matching_frames": 0,
            "total_frames": len(boxes),
            "exact_match": False,
        }
    matching_frames = sum(
        actual == expected for actual, expected in zip(boxes, reference)
    )
    return {
        "matching_frames": matching_frames,
        "total_frames": len(boxes),
        "exact_match": matching_frames == len(boxes),
    }


def run_video(video_id, ratios, output_dir, overwrite):
    video_path = os.path.join(ROOT, "DHF1k", video_id + ".AVI")
    if not os.path.isfile(video_path):
        raise FileNotFoundError("Missing RetargetVid video: {}".format(video_path))
    if not re.match(r"^\d{3}$", video_id):
        raise ValueError("Video IDs must be three digits, for example 001.")
    os.makedirs(output_dir, exist_ok=True)

    records = {}
    for ratio in ratios:
        suffix = "{}_{}.txt".format(video_id, RATIO_NAMES[ratio])
        output_path = os.path.join(output_dir, suffix)
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                "{} exists; choose another --output-dir or pass --overwrite."
                .format(output_path)
            )

        parameters = smartVidCrop.sc_init_crop_params(use_best_settings=False)
        parameters["out_ratio"] = ratio
        annotations = read_annotations(video_id, ratio)
        start_time = time.time()
        video_data, method_info = smartVidCrop.smart_vid_crop(
            video_path,
            CP=parameters,
            save_vid=False,
            verbose=False,
        )
        elapsed_seconds = time.time() - start_time
        boxes = [[int(value) for value in box] for box in video_data["bbs"]]
        if not boxes:
            raise RuntimeError(
                "SmartVidCrop produced no crop windows for {} {}."
                .format(video_id, ratio)
            )

        with open(output_path, "w") as destination:
            for box in boxes:
                destination.write(",".join(str(value) for value in box) + "\n")

        default_boxes = load_reference_boxes(video_id, ratio, "default_config")
        paper_boxes = load_reference_boxes(video_id, ratio, "smartvidcrop")
        default_comparison = compare_box_sequences(boxes, default_boxes)
        paper_comparison = compare_box_sequences(boxes, paper_boxes)
        records[ratio] = {
            "video_id": video_id,
            "ratio": ratio,
            "frames": len(boxes),
            "elapsed_seconds": elapsed_seconds,
            "parameters": parameters,
            "scores": evaluate_boxes(boxes, annotations, ratio),
            "comparison_to_bundled_default_config": default_comparison,
            "comparison_to_bundled_paper_result": paper_comparison,
            "result_file": suffix,
            "method_summary": {
                key: value
                for key, value in method_info.items()
                if key in ("result", "mean_sal_score", "coverage_score", "t_total")
            },
        }
        print(
            "{} {}: frames={}, mean IoU={:.4f}, "
            "default_config exact match={}, paper-output exact match={}"
            .format(
                video_id,
                ratio,
                len(boxes),
                records[ratio]["scores"]["mean_annotator_iou"],
                default_comparison["exact_match"] if default_comparison else None,
                paper_comparison["exact_match"] if paper_comparison else None,
            )
        )
        del video_data

    summary = {
        "experiment": "A Fast Smart-Cropping Method and Dataset for Video Retargeting",
        "implementation": "Repository smartVidCrop.py, default ICIP settings",
        "video_id": video_id,
        "runs": records,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path) and not overwrite:
        raise FileExistsError(
            "{} exists; choose another --output-dir or pass --overwrite."
            .format(summary_path)
        )
    with open(summary_path, "w") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
    print("Saved fresh crop windows and scores to {}".format(output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="001", help="RetargetVid video ID (default: 001)")
    parser.add_argument(
        "--ratios",
        nargs="+",
        choices=sorted(RATIO_NAMES.keys()),
        default=["1:3", "3:1"],
        help="Target aspect ratios to reproduce (default: both paper ratios)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output folder (default: reproduction/smartvidcrop_<video-id>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace crop files and summary from a previous run.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(
        ROOT, "reproduction", "smartvidcrop_{}".format(args.video_id)
    )
    run_video(args.video_id, args.ratios, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
