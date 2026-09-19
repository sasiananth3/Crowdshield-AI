"""Extract 1 Hz temporal features for forecasting training.

Labels remain detector-derived pseudo-counts for this prototype. The script now
stores the five model inputs: count, count delta, density estimate, peak zone
proxy and kinematic signal. Replace the count column with independently labelled
counts before making a ground-truth forecasting claim.
"""

import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ml.detection import PersonDetector
from ml.density import DensityEstimator
from ml.kinematics import MotionAnalyzer

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "data/raw/umn-demo.avi"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    detector = PersonDetector(ROOT / "models/yolov8n.pt")
    density = DensityEstimator(ROOT / "models/density.pt")
    motion = MotionAnalyzer()
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise RuntimeError("Video cannot be opened")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Invalid video frame rate")
    rows, frame_id, segment, previous, previous_count = [], 0, 0, None, 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_id % max(1, round(fps)) == 0:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)
            cv2.normalize(hist, hist)
            if previous is not None and cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.55:
                segment += 1
                motion = MotionAnalyzer()
                previous_count = 0
            previous = hist
            timestamp = frame_id / fps
            items = detector.detect(frame, tracking=True)
            current_count = len(items)
            motion_result = motion.update(items, timestamp)
            estimated = density.predict(frame)
            density_count = float(estimated["count"]) if estimated else float(current_count)
            rows.append({
                "sequence": f"umn-demo-segment-{segment}",
                "timestamp": round(timestamp, 3),
                "count": current_count,
                "count_delta": current_count - previous_count,
                "density_mean": density_count,
                "density_max_zone": density_count,
                "kinematic_score": round(float(np.clip((motion_result.get("reversal_fraction") or 0) * 0.6 + (motion_result.get("sudden_motion_fraction") or 0) * 0.3 + (motion_result.get("stalled_fraction") or 0) * 0.1, 0, 1)), 4),
            })
            previous_count = current_count
            if len(rows) % 30 == 0:
                print(f"Extracted {len(rows)} samples", flush=True)
        frame_id += 1
    video.release()
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "timestamp", "count", "count_delta", "density_mean", "density_max_zone", "kinematic_score"]
    with (output / "counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output / "counts.metadata.json").write_text(json.dumps({
        "source": "UMN university demonstration AVI",
        "label_type": "YOLOv8n pseudo counts; not manual ground truth",
        "features": fields[2:], "sample_rate_hz": 1, "samples": len(rows), "segments": segment + 1,
    }, indent=2))
    print(f"Saved {len(rows)} pseudo-labelled temporal samples")
