"""Standalone held-out ShanghaiTech Part B density evaluation.

This script evaluates a trained density checkpoint once on the official test
split and reports MAE/RMSE. It never changes weights and never writes a claimed
accuracy percentage.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from ml.density import DensityNet, predict_with_model

ROOT = Path(__file__).resolve().parents[1]


def locate_part_b(root):
    matches = [p for p in root.rglob("test_data") if "part_b" in str(p).lower()]
    if not matches:
        raise FileNotFoundError("ShanghaiTech Part B is not installed")
    return matches[0].parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/ShanghaiTech"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models/density.pt"))
    args = parser.parse_args()
    part = locate_part_b(Path(args.data))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DensityNet(); model.load_state_dict(state["state_dict"]); model.eval()
    predictions, truth = [], []
    for path in sorted((part / "test_data" / "images").glob("*.jpg")):
        points = loadmat(part / "test_data" / "ground_truth" / ("GT_" + path.stem + ".mat"))["image_info"][0,0][0,0][0]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        prediction = predict_with_model(model, bgr, tiled=True, tile_size=256)["count"]
        predictions.append(prediction); truth.append(len(points))
    error = np.asarray(predictions) - np.asarray(truth)
    report = {"dataset":"ShanghaiTech Part B", "test_images":len(truth),
              "mae":float(np.abs(error).mean()), "rmse":float(np.sqrt(np.mean(error**2))),
              "checkpoint":args.checkpoint, "protocol":"official test split; tiled 256px inference"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
