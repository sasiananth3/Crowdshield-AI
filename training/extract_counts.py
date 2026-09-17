"""Extract detector-derived pseudo counts at 1 Hz, not crowd-risk ground truth."""

import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ml.detection import PersonDetector

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "data/raw/umn-demo.avi"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    detector = PersonDetector(ROOT / "models/yolov8n.pt")
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise RuntimeError("Video cannot be opened")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Invalid video frame rate")
    rows = []
    frame_id = 0
    segment = 0
    previous = None
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_id % max(1, round(fps)) == 0:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)
            cv2.normalize(hist, hist)
            if (
                previous is not None
                and cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.55
            ):
                segment += 1
            previous = hist
            items = detector.detect(frame, tracking=False)
            rows.append(
                {
                    "sequence": f"umn-demo-segment-{segment}",
                    "timestamp": round(frame_id / fps, 3),
                    "count": len(items),
                }
            )
            if len(rows) % 30 == 0:
                print(f"Extracted {len(rows)} samples", flush=True)
        frame_id += 1
    video.release()
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "timestamp", "count"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "counts.metadata.json").write_text(
        json.dumps(
            {
                "source": "UMN university demonstration AVI",
                "label_type": "YOLOv8n pseudo counts; not manual ground truth",
                "sample_rate_hz": 1,
                "samples": len(rows),
                "segments": segment + 1,
            },
            indent=2,
        )
    )
    print(f"Saved {len(rows)} pseudo-labelled temporal samples")
