"""Train the ShanghaiTech Part B density model with crop augmentation.

Phase-1 accuracy improvements:
- all official 400 Part-B training images participate in optimization;
- random 256x256 crops preserve local spatial detail;
- horizontal flip and color jitter improve appearance robustness;
- cosine LR decay is used for a 200-epoch default run;
- the official 316-image test split is never used for model selection.

Example: python -m training.train_density --epochs 200 --threads 4
"""

import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageEnhance
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from ml.density import DensityNet, image_tensor

ROOT = Path(__file__).resolve().parents[1]


def discover(root):
    matches = [p for p in root.rglob("train_data") if "part_b" in str(p).lower()]
    if not matches:
        raise FileNotFoundError("Download ShanghaiTech Part B with scripts/download_data.py first")
    return matches[0].parent


def read_data(directory):
    samples = []
    for path in sorted((directory / "images").glob("*.jpg")):
        rgb = np.array(Image.open(path).convert("RGB"))
        points = loadmat(directory / "ground_truth" / ("GT_" + path.stem + ".mat"))["image_info"][0, 0][0, 0][0]
        samples.append((path.name, rgb, np.asarray(points, dtype=np.float32)))
    if not samples:
        raise ValueError("No labelled images found")
    return samples


def crop_sample(rgb, points, rng, size=256, augment=True):
    h, w = rgb.shape[:2]
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        h, w = rgb.shape[:2]
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    crop = rgb[top:top + size, left:left + size]
    pts = points[(points[:, 0] >= left) & (points[:, 0] < left + size) &
                 (points[:, 1] >= top) & (points[:, 1] < top + size)].copy()
    if len(pts):
        pts[:, 0] -= left
        pts[:, 1] -= top
    if augment and rng.random() < 0.5:
        crop = crop[:, ::-1].copy()
        if len(pts):
            pts[:, 0] = size - 1 - pts[:, 0]
    image = Image.fromarray(crop)
    if augment:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.8, 1.2)))
    rgb = np.asarray(image, dtype=np.uint8)
    density = np.zeros((16, 16), dtype=np.float32)
    for x, y in pts:
        density[min(15, max(0, int(y * 16 / size))), min(15, max(0, int(x * 16 / size)))] += 1
    density = gaussian_filter(density, 0.7, mode="reflect")
    if density.sum() > 0:
        density *= len(pts) / density.sum()
    return image_tensor(rgb), torch.from_numpy(density).unsqueeze(0), len(pts)


def metrics(truth, prediction):
    error = np.asarray(prediction) - np.asarray(truth)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean()))}


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    start = time.time()
    part = discover(Path(args.data))
    train = read_data(part / "train_data")
    if len(train) != 400:
        raise ValueError(f"Expected the official 400-image Part B training split, found {len(train)}")

    net = DensityNet(pretrained=True)
    net.backbone.eval()
    for parameter in net.backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(net.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    best_loss = float("inf")
    best = copy.deepcopy(net.state_dict())
    history = []

    for epoch in range(args.epochs):
        net.head.train()
        order = rng.permutation(len(train))
        train_loss = []
        for start_id in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[start_id:start_id + args.batch_size]]
            encoded = [crop_sample(rgb, points, rng, args.crop_size, True) for _, rgb, points in batch]
            images = torch.stack([x[0] for x in encoded])
            targets = torch.stack([x[1] for x in encoded])
            counts = targets.sum((1, 2, 3))
            output = net(images)
            loss = nn.functional.mse_loss(output.sum((1, 2, 3)) / 100, counts / 100)
            loss = loss + 0.15 * nn.functional.mse_loss(output, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.head.parameters(), 5.0)
            optimizer.step()
            train_loss.append(float(loss.item()))
        scheduler.step()

        # Deterministic validation crop from each training image. This is a
        # development selection signal; the official 316-image test set stays untouched.
        net.head.eval()
        val_truth, val_pred = [], []
        with torch.inference_mode():
            for _, rgb, points in train:
                sample, _, count = crop_sample(rgb, points, np.random.default_rng(12345), args.crop_size, False)
                pred = net(sample.unsqueeze(0)).sum().item()
                val_truth.append(count)
                val_pred.append(pred)
        result = {"epoch": epoch + 1, "train_loss": float(np.mean(train_loss)),
                  "lr": float(scheduler.get_last_lr()[0]), **metrics(val_truth, val_pred)}
        history.append(result)
        print(json.dumps(result), flush=True)
        if result["mae"] < best_loss:
            best_loss = result["mae"]
            best = copy.deepcopy(net.state_dict())

    net.load_state_dict(best)
    net.eval()
    test = read_data(part / "test_data")
    predictions, true_counts = [], []
    with torch.inference_mode():
        for _, rgb, points in test:
            pred = net(image_tensor(rgb).unsqueeze(0)).sum().item()
            predictions.append(pred)
            true_counts.append(len(points))
    report = {
        "dataset": "ShanghaiTech Part B",
        "seed": 42,
        "train_images": len(train),
        "validation_images": len(train),
        "validation_protocol": "deterministic 256px crop from each training image; selection-only, not an independent image holdout",
        "test_images": len(test),
        "epochs": args.epochs,
        "input_size": args.crop_size,
        "augmentation": ["random_crop", "horizontal_flip", "brightness", "contrast", "saturation"],
        "scheduler": "CosineAnnealingLR",
        "architecture": "ImageNet MobileNetV2 frozen features + trained dilated head",
        "training_seconds": round(time.time() - start, 2),
        "validation_best_mae": best_loss,
        "held_out_test": metrics(true_counts, predictions),
        "limitations": [
            "Validation is crop-based on the official training images; the official test split remains untouched",
            "No claim of reproducing or beating a published paper without identical protocol and run",
            "Risk prediction is not evaluated by this counting benchmark",
        ],
        "history": history,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    torch.save({"state_dict": best, "metadata": {
        "dataset": report["dataset"], "epochs": args.epochs,
        "test_mae": report["held_out_test"]["mae"], "experimental": True,
        "inference_mode": "tiled_256", "crop_size": args.crop_size,
    }}, ROOT / "models/density.pt")
    (ROOT / "reports/density_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/ShanghaiTech"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=256)
    run(parser.parse_args())
