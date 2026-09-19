"""Evaluate a trained density checkpoint on ShanghaiTech Part B test images.

This command performs no optimization or checkpoint selection.  Keep the
official test split for final evaluation rather than repeatedly tuning to it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch

from ml.density import DensityNet, image_tensor
from training.train_density import discover, metrics


ROOT = Path(__file__).resolve().parents[1]


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Density checkpoint not found: {checkpoint}")

    torch.set_num_threads(args.threads)
    started = time.time()
    part = discover(Path(args.data).expanduser().resolve())
    test_directory = part / "test_data"
    image_paths = sorted((test_directory / "images").glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No test images found in {test_directory / 'images'}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise ValueError("Checkpoint must be a dictionary containing 'state_dict'")

    model = DensityNet()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    predictions = []
    truth = []
    with torch.inference_mode():
        for offset in range(0, len(image_paths), args.batch_size):
            paths = image_paths[offset : offset + args.batch_size]
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(image_tensor(np.array(image.convert("RGB"))))
                points = loadmat(
                    test_directory / "ground_truth" / ("GT_" + path.stem + ".mat")
                )["image_info"][0, 0][0, 0][0]
                truth.append(len(points))
            batch = torch.stack(tensors)
            predictions.extend(model(batch).sum((1, 2, 3)).cpu().tolist())

    errors = np.asarray(predictions) - np.asarray(truth)
    with checkpoint.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    evaluation = {
        "dataset": "ShanghaiTech Part B",
        "split": "official test_data",
        "test_images": len(image_paths),
        "held_out_test": metrics(truth, predictions),
        "mean_error": float(errors.mean()),
        "checkpoint": display_path(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_metadata": saved.get("metadata", {}),
        "evaluation_seconds": round(time.time() - started, 2),
        "note": "Evaluation only; no training or checkpoint selection was performed.",
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve training protocol/history when updating the project's combined
    # report, while making every evaluation field come from this fresh run.
    report = {}
    if output.is_file():
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            report = {}
    report.update(evaluation)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(evaluation, indent=2))
    return evaluation


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a density model on held-out ShanghaiTech Part B."
    )
    parser.add_argument("--data", default=str(ROOT / "data/raw/ShanghaiTech"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models/density.pt"))
    parser.add_argument(
        "--output", default=str(ROOT / "reports/density_evaluation.json")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
