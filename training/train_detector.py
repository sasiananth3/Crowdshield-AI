"""Fine-tune YOLOv8 on a CrowdHuman-style YOLO dataset.

Prepare an authorized CrowdHuman conversion under data/raw/CrowdHuman with a
standard YOLO data.yaml. The dataset must contain dense-scene person labels.
The trained checkpoint is saved as models/yolov8s-crowdhuman.pt and is selected
by validation metrics from Ultralytics.

Ultralytics expects standard YOLO train/val paths and class names in data.yaml.
See https://docs.ultralytics.com/datasets/detect/ for the supported format.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/CrowdHuman/data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(
            f"CrowdHuman YOLO data.yaml not found at {data}. Convert the licensed dataset first."
        )
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        lr0=0.001, lrf=0.01, mosaic=1.0, close_mosaic=10, patience=20,
        device=args.device, workers=args.workers, project=str(ROOT / "runs" / "detect"),
        name="crowdhuman_finetune", exist_ok=True, classes=[0], verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Ultralytics did not produce a best checkpoint: {best}")
    target = ROOT / "models" / "yolov8s-crowdhuman.pt"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(best.read_bytes())
    print(f"Saved fine-tuned detector to {target}")


if __name__ == "__main__":
    main()
