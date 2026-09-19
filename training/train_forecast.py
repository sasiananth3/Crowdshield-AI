"""Train a multivariate count LSTM without cross-split temporal leakage.

The trainer accepts multiple independently recorded sequences in one CSV. It
partitions by sequence before windowing, requires 500+ training windows by
default, and reports persistence on the same test windows. Current repository
labels are detector pseudo-counts; independent manual counts are required for a
real forecasting accuracy claim.
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ml.forecast import make_lstm, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def windows(rows, window, horizon, features):
    xs, ys = [], []
    for i in range(window, len(rows) - horizon + 1):
        part = rows[i - window:i + horizon]
        if len({r["sequence"] for r in part}) != 1:
            continue
        times = np.asarray([float(r["timestamp"]) for r in part])
        if not np.allclose(np.diff(times), 1, atol=0.05):
            continue
        xs.append([[float(r[f]) for f in features] for r in rows[i-window:i]])
        ys.append(float(rows[i + horizon - 1]["count"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def partition_sequences(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    sequences = list(grouped.values())
    if len(sequences) < 3:
        raise ValueError("Need at least 3 independent temporal sequences for train/validation/test")
    sequences.sort(key=lambda s: len(s), reverse=True)
    train, val, test = [], [], []
    for index, sequence in enumerate(sequences):
        target = train if index % 5 < 3 else val if index % 5 == 3 else test
        target.extend(sequence)
    return train, val, test


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    features = [f for f in FEATURE_NAMES if f in rows[0]]
    if features != FEATURE_NAMES:
        raise ValueError(f"CSV must contain all multivariate features: {FEATURE_NAMES}")
    train_rows, val_rows, test_rows = partition_sequences(rows)
    sets = [windows(part, args.window, args.horizon, features) for part in (train_rows, val_rows, test_rows)]
    if len(sets[0][1]) < args.min_train_windows:
        raise ValueError(f"Need at least {args.min_train_windows} training windows; found {len(sets[0][1])}")
    if any(len(y) < 5 for _, y in sets[1:]):
        raise ValueError("Need >=5 validation/test windows")

    x, y = sets[0]
    feature_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    feature_std = np.maximum(x.reshape(-1, x.shape[-1]).std(axis=0), 1e-6)
    target_mean, target_std = float(feature_mean[0]), float(feature_std[0])
    model = make_lstm(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    best, best_mae = copy.deepcopy(model.state_dict()), float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(42 + epoch).permutation(len(y))
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            xb = torch.tensor((x[ids] - feature_mean) / feature_std, dtype=torch.float32)
            yb = torch.tensor((y[ids] - target_mean) / target_std, dtype=torch.float32)
            pred = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            vx = torch.tensor((sets[1][0] - feature_mean) / feature_std, dtype=torch.float32)
            val = model(vx).numpy() * target_std + target_mean
        mae = float(np.abs(val - sets[1][1]).mean())
        if mae < best_mae:
            best_mae, best = mae, copy.deepcopy(model.state_dict())

    model.load_state_dict(best); model.eval()
    test_x, test_y = sets[2]
    with torch.inference_mode():
        tx = torch.tensor((test_x - feature_mean) / feature_std, dtype=torch.float32)
        prediction = np.maximum(0, model(tx).numpy() * target_std + target_mean)
    metric = lambda p: {"mae": float(np.abs(p-test_y).mean()), "rmse": float(np.sqrt(np.mean((p-test_y)**2)))}
    meta = {"window": args.window, "horizon": args.horizon, "input_size": len(features),
            "feature_names": features, "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
            "source": "Multiple temporal sequences; repository default labels are YOLO-derived pseudo-counts", "sample_rate_hz": 1}
    report = {"metadata": meta, "train_windows": len(sets[0][1]), "validation_windows": len(sets[1][1]),
              "test_windows": len(test_y), "validation_best_mae": best_mae, "lstm_test": metric(prediction),
              "persistence_baseline_test": metric(test_x[:, -1, 0]),
              "split": "Sequence-level partition before windowing", "epochs": args.epochs,
              "limitations": ["Independent sequences are required for generalization", "Default labels are detector pseudo-counts", "No incident labels or warning-lead-time validation"]}
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
    torch.save({"state_dict": best, "metadata": meta}, ROOT / "models/forecast.pt")
    (ROOT / "reports/forecast_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data/processed/counts.csv"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-train-windows", type=int, default=500)
    run(p.parse_args())
