"""Train on Part B train split; choose checkpoints on validation only.

Example: python -m training.train_density --epochs 35
"""

import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from ml.density import DensityNet, image_tensor

ROOT = Path(__file__).resolve().parents[1]


def discover(root):
    matches = [p for p in root.rglob("train_data") if "part_b" in str(p).lower()]
    if not matches:
        raise FileNotFoundError(
            "Download ShanghaiTech Part B with scripts/download_data.py first"
        )
    return matches[0].parent


def read_data(directory, limit=None):
    samples = []
    paths = sorted((directory / "images").glob("*.jpg"))
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        rgb = np.array(Image.open(path).convert("RGB"))
        points = loadmat(directory / "ground_truth" / ("GT_" + path.stem + ".mat"))[
            "image_info"
        ][0, 0][0, 0][0]
        density = np.zeros((16, 16), dtype=np.float32)
        h, w = rgb.shape[:2]
        for x, y in points:
            density[
                min(15, max(0, int(y * 16 / h))), min(15, max(0, int(x * 16 / w)))
            ] += 1
        density = gaussian_filter(density, 0.7, mode="reflect")
        if density.sum() > 0:
            density *= len(points) / density.sum()
        samples.append(
            (
                path.name,
                image_tensor(rgb),
                torch.from_numpy(density).unsqueeze(0),
                len(points),
            )
        )
    if not samples:
        raise ValueError("No labelled images found")
    return samples


def metrics(truth, prediction):
    error = np.asarray(prediction) - np.asarray(truth)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
    }


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    start = time.time()
    part = discover(Path(args.data))
    train = read_data(part / "train_data")
    order = np.random.default_rng(42).permutation(len(train))
    nval = max(1, len(train) // 5)
    val_ids, train_ids = order[:nval], order[nval:]
    net = DensityNet(pretrained=True)
    net.backbone.eval()
    for parameter in net.backbone.parameters():
        parameter.requires_grad_(False)

    def encode(samples):
        values = []
        with torch.inference_mode():
            for i in range(0, len(samples), 16):
                values.append(
                    net.backbone(torch.stack([s[1] for s in samples[i : i + 16]]))
                )
        return torch.cat(values).clone()

    features = encode(train)
    targets = torch.stack([s[2] for s in train])
    counts = targets.sum((1, 2, 3))
    optimizer = torch.optim.Adam(net.head.parameters(), lr=args.lr)
    best_loss = float("inf")
    best = copy.deepcopy(net.state_dict())
    history = []
    for epoch in range(args.epochs):
        net.head.train()
        ids = np.random.permutation(train_ids)
        for i in range(0, len(ids), 16):
            batch = ids[i : i + 16]
            output = net.head(features[batch])
            # Count and spatial objectives; both are on training annotations only.
            loss = nn.functional.mse_loss(
                output.sum((1, 2, 3)) / 100, counts[batch] / 100
            )
            loss = loss + 0.15 * nn.functional.mse_loss(output, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.head.eval()
        with torch.inference_mode():
            prediction = net.head(features[val_ids]).sum((1, 2, 3)).numpy()
        result = {"epoch": epoch + 1, **metrics(counts[val_ids].numpy(), prediction)}
        history.append(result)
        print(json.dumps(result), flush=True)
        if result["mae"] < best_loss:
            best_loss = result["mae"]
            best = copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    net.eval()
    # Only after selection: evaluate held-out official test set exactly once.
    test = read_data(part / "test_data")
    test_features = encode(test)
    with torch.inference_mode():
        predictions = net.head(test_features).sum((1, 2, 3)).numpy()
    true_counts = [s[3] for s in test]
    mean_count = float(counts[train_ids].mean())
    report = {
        "dataset": "ShanghaiTech Part B",
        "seed": 42,
        "train_images": len(train_ids),
        "validation_images": len(val_ids),
        "test_images": len(test),
        "epochs": args.epochs,
        "input_size": 256,
        "architecture": "ImageNet MobileNetV2 frozen features + trained dilated head",
        "training_seconds": round(time.time() - start, 2),
        "validation_best_mae": best_loss,
        "held_out_test": metrics(true_counts, predictions),
        "training_mean_count_baseline": metrics(true_counts, [mean_count] * len(test)),
        "baseline_note": "Constant-count sanity baseline, NOT a reproduction of either base paper",
        "limitations": [
            "Short prototype training run",
            "Square resize; no scale augmentation",
            "Not a risk-prediction benchmark",
            "No accuracy improvement over base papers established",
        ],
        "history": history,
        "splits": {
            "train": [train[int(i)][0] for i in train_ids],
            "validation": [train[int(i)][0] for i in val_ids],
            "test": [s[0] for s in test],
        },
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best,
            "metadata": {
                "dataset": report["dataset"],
                "epochs": args.epochs,
                "test_mae": report["held_out_test"]["mae"],
                "experimental": True,
            },
        },
        ROOT / "models/density.pt",
    )
    (ROOT / "reports/density_evaluation.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"history", "splits"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/ShanghaiTech"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
