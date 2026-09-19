"""Extract detector-derived pseudo counts, not crowd-risk ground truth."""

import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ml.detection import PersonDetector

ROOT = Path(__file__).resolve().parents[1]


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def default_weights():
    trained = ROOT / "models/detector.pt"
    return trained if trained.is_file() else ROOT / "models/yolov8n.pt"


def run(args):
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    if not 0 <= args.scene_threshold <= 1:
        raise ValueError("--scene-threshold must be between 0 and 1")
    video_path = Path(args.video).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {weights_path}")

    torch.set_num_threads(args.threads)
    detector = PersonDetector(weights_path, confidence=args.confidence)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Video cannot be opened: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        video.release()
        raise RuntimeError("Invalid video frame rate")
    sample_every = max(1, round(fps / args.sample_rate))
    effective_sample_rate = fps / sample_every
    rows = []
    frame_id = 0
    segment = 0
    previous = None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame_id % sample_every == 0:
                hist = cv2.calcHist(
                    [frame], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3
                )
                cv2.normalize(hist, hist)
                if (
                    previous is not None
                    and cv2.compareHist(
                        previous, hist, cv2.HISTCMP_BHATTACHARYYA
                    )
                    > args.scene_threshold
                ):
                    segment += 1
                previous = hist
                items = detector.detect(frame, tracking=False)
                rows.append(
                    {
                        "sequence": f"{video_path.stem}-segment-{segment}",
                        "timestamp": round(frame_id / fps, 3),
                        "count": len(items),
                    }
                )
                if len(rows) % 30 == 0:
                    print(f"Extracted {len(rows)} samples", flush=True)
            frame_id += 1
    finally:
        video.release()
    if not rows:
        raise ValueError("Video contains no readable frames")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "timestamp", "count"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    metadata = {
        "source_video": display_path(video_path),
        "detector_weights": display_path(weights_path),
        "label_type": "YOLO-derived pseudo counts; not manual ground truth",
        "sample_rate_hz": effective_sample_rate,
        "requested_sample_rate_hz": args.sample_rate,
        "source_fps": fps,
        "samples": len(rows),
        "segments": segment + 1,
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".part")
    metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_temporary.replace(metadata_path)
    print(f"Saved {len(rows)} pseudo-labelled temporal samples to {output}")
    return metadata


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "data/raw/umn-demo.avi"))
    parser.add_argument(
        "--weights",
        default=str(default_weights()),
        help="detector checkpoint (prefers models/detector.pt when present)",
    )
    parser.add_argument("--output", default=str(ROOT / "data/processed/counts.csv"))
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--scene-threshold", type=float, default=0.55)
    parser.add_argument("--threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
