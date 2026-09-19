"""Fit an LSTM to chronological count windows, with purged time partitions.

Default input is explicitly detector-generated pseudo labels. Evaluation measures
count-sequence forecasting only; it cannot establish incident prediction accuracy.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from ml.forecast import make_lstm

ROOT = Path(__file__).resolve().parents[1]


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def windows(rows, window, horizon, step=1.0):
    xs, ys = [], []
    for i in range(window, len(rows) - horizon + 1):
        part = rows[i - window : i + horizon]
        if len({r["sequence"] for r in part}) != 1:
            continue
        times = np.array([float(r["timestamp"]) for r in part])
        if not np.allclose(np.diff(times), step, atol=max(0.05, step * 0.05)):
            continue
        xs.append([float(r["count"]) for r in rows[i - window : i]])
        ys.append(float(rows[i + horizon - 1]["count"]))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def read_rows(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Count CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "timestamp", "count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain sequence, timestamp, and count columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Count CSV contains no samples")
    previous = {}
    for line, row in enumerate(rows, start=2):
        try:
            timestamp = float(row["timestamp"])
            count = float(row["count"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric value on CSV line {line}") from error
        if not row["sequence"]:
            raise ValueError(f"Missing sequence on CSV line {line}")
        if not math.isfinite(timestamp) or not math.isfinite(count) or count < 0:
            raise ValueError(f"Invalid timestamp or count on CSV line {line}")
        if row["sequence"] in previous and timestamp <= previous[row["sequence"]]:
            raise ValueError(
                f"Timestamps must increase within sequence on CSV line {line}"
            )
        previous[row["sequence"]] = timestamp
    return path, rows


def run(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.window < 1 or args.horizon < 1:
        raise ValueError("--window and --horizon must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be greater than zero")
    csv_path, rows = read_rows(args.csv)
    source_metadata = {}
    metadata_path = csv_path.with_suffix(".metadata.json")
    if metadata_path.is_file():
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            source_metadata = {}
    sample_rate = float(source_metadata.get("sample_rate_hz", 1))
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Source sample_rate_hz must be greater than zero")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = len(rows)
    # Partition BEFORE windowing so no input or future target crosses a split.
    partitions = [
        rows[: int(n * 0.6)],
        rows[int(n * 0.6) : int(n * 0.8)],
        rows[int(n * 0.8) :],
    ]
    sets = [
        windows(part, args.window, args.horizon, step=1 / sample_rate)
        for part in partitions
    ]
    if any(len(y) < 5 for _, y in sets):
        raise ValueError(
            "Insufficient chronological sequences: require >=5 disjoint windows per split"
        )
    x, y = sets[0]
    mean = float(x.mean())
    std = max(1.0, float(x.std()))
    tensor = lambda values: torch.tensor((values - mean) / std, dtype=torch.float32)
    model = make_lstm()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        "source_csv": display_path(csv_path),
        "source_metadata": source_metadata,
        "label_warning": "Detector pseudo-counts are not independent crowd-risk ground truth",
        "sample_rate_hz": sample_rate,
        "seed": args.seed,
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
    model_output = Path(args.output).expanduser().resolve()
    report_output = Path(args.report).expanduser().resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    model_temporary = model_output.with_suffix(model_output.suffix + ".part")
    torch.save({"state_dict": best, "metadata": meta}, model_temporary)
    model_temporary.replace(model_output)
    report_temporary = report_output.with_suffix(report_output.suffix + ".part")
    report_temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_temporary.replace(report_output)
    print(json.dumps(report, indent=2))
    return report


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data/processed/counts.csv"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "models/forecast.pt"))
    p.add_argument("--report", default=str(ROOT / "reports/forecast_evaluation.json"))
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
