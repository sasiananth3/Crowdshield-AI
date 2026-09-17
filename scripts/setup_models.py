"""Fetch official pretrained detectors. Custom density/LSTM weights are trained separately."""

from pathlib import Path
import argparse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", action="store_true")
    args = parser.parse_args()
    (ROOT / "models").mkdir(exist_ok=True)
    from ultralytics import YOLO

    YOLO(str(ROOT / "models/yolov8n.pt"))
    if args.pose:
        target = ROOT / "models/pose_landmarker_lite.task"
        if not target.exists():
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            part = target.with_suffix(".part")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                part.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if part.stat().st_size < 1_000_000:
                raise RuntimeError("Invalid MediaPipe model download")
            part.replace(target)
    print(
        "Pretrained model download complete. Custom density/LSTM checkpoints are included in the repository; see training/ to reproduce them."
    )
