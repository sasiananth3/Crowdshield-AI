"""Fit an LSTM to chronological count windows, with purged time partitions.

Default input is explicitly detector-generated pseudo labels. Evaluation measures
count-sequence forecasting only; it cannot establish incident prediction accuracy.
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ml.forecast import make_lstm

ROOT = Path(__file__).resolve().parents[1]


def windows(rows, window, horizon):
    xs, ys = [], []
    for i in range(window, len(rows) - horizon + 1):
        part = rows[i - window : i + horizon]
        if len({r["sequence"] for r in part}) != 1:
            continue
        times = np.array([float(r["timestamp"]) for r in part])
        if not np.allclose(np.diff(times), 1, atol=0.05):
            continue
        xs.append([float(r["count"]) for r in rows[i - window : i]])
        ys.append(float(rows[i + horizon - 1]["count"]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    # Partition BEFORE windowing so no input or future target crosses a split.
    partitions = [
        rows[: int(n * 0.6)],
        rows[int(n * 0.6) : int(n * 0.8)],
        rows[int(n * 0.8) :],
    ]
    sets = [windows(part, args.window, args.horizon) for part in partitions]
    if any(len(y) < 5 for _, y in sets):
        raise ValueError(
            "Insufficient chronological sequences: require >=5 disjoint windows per split"
        )
    x, y = sets[0]
    mean = float(x.mean())
    std = max(1.0, float(x.std()))
    tensor = lambda values: torch.tensor((values - mean) / std, dtype=torch.float32)
    model = make_lstm()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    for _ in range(args.epochs):
        model.train()
        pred = model(tensor(x)[:, :, None])
        loss = torch.nn.functional.mse_loss(pred, tensor(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val = model(tensor(sets[1][0])[:, :, None]).numpy() * std + mean
        mae = float(np.abs(val - sets[1][1]).mean())
        if mae < best_mae:
            best_mae = mae
            best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    test_x, test_y = sets[2]
    with torch.inference_mode():
        prediction = np.maximum(
            0, model(tensor(test_x)[:, :, None]).numpy() * std + mean
        )
    metric = lambda p: {
        "mae": float(np.abs(p - test_y).mean()),
        "rmse": float(np.sqrt(np.mean((p - test_y) ** 2))),
    }
    meta = {
        "window": args.window,
        "horizon": args.horizon,
        "mean": mean,
        "std": std,
        "source": "YOLOv8n pseudo-counts from UMN demo; NOT independent crowd-risk ground truth",
        "sample_rate_hz": 1,
    }
    report = {
        "metadata": meta,
        "train_windows": len(sets[0][1]),
        "validation_windows": len(sets[1][1]),
        "test_windows": len(test_y),
        "validation_best_mae": best_mae,
        "lstm_test": metric(prediction),
        "persistence_baseline_test": metric(test_x[:, -1]),
        "split": "Chronological 60/20/20 before windowing; scene-cut windows excluded",
        "epochs": args.epochs,
        "limitations": [
            "One demonstration video, not external validation",
            "Predicted count targets are detector pseudo-labels",
            "No incident labels or warning-lead-time validation",
            "Performance may be worse than persistence",
        ],
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    torch.save({"state_dict": best, "metadata": meta}, ROOT / "models/forecast.pt")
    (ROOT / "reports/forecast_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data/processed/counts.csv"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    run(p.parse_args())
