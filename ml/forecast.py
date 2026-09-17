"""Time-aware baseline and optional trained count LSTM. Never predicts stampedes."""

from collections import deque
from pathlib import Path
import numpy as np


class TrendForecast:
    name = "linear_trend_baseline"

    def __init__(self, max_history=60):
        self.history = deque(maxlen=max_history)

    def update(self, timestamp, count):
        if not self.history or timestamp > self.history[-1][0]:
            self.history.append((float(timestamp), float(count)))

    def predict(self, horizon=10):
        if len(self.history) < 5 or self.history[-1][0] - self.history[0][0] < 4:
            return {
                "method": self.name,
                "status": "warming_up",
                "horizon_seconds": horizon,
                "count": None,
            }
        values = np.array(self.history)
        t = values[:, 0] - values[-1, 0]
        slope = float(np.polyfit(t, values[:, 1], 1)[0])
        predicted = max(0.0, float(values[-1, 1] + slope * horizon))
        return {
            "method": self.name,
            "status": "experimental",
            "horizon_seconds": horizon,
            "count": round(predicted, 1),
            "slope_per_second": round(slope, 3),
        }


def make_lstm():
    import torch
    from torch import nn

    class CountLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1]).squeeze(-1)

    return CountLSTM()


class CountForecaster(TrendForecast):
    def __init__(self, checkpoint):
        super().__init__()
        self.model = None
        if Path(checkpoint).exists():
            import torch

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.config = saved["metadata"]
            self.model = make_lstm()
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()

    def predict(self, horizon=10):
        if self.model is None:
            return super().predict(horizon)
        import torch

        c = self.config
        if (
            len(self.history) < 2
            or self.history[-1][0] - self.history[0][0] < c["window"] - 1
        ):
            return {
                "method": "count_lstm",
                "status": "warming_up",
                "horizon_seconds": c["horizon"],
                "count": None,
            }
        values = np.array(self.history)
        times = np.arange(c["window"]) - (c["window"] - 1) + values[-1, 0]
        sequence = np.interp(times, values[:, 0], values[:, 1])
        normalized = (sequence - c["mean"]) / c["std"]
        with torch.inference_mode():
            pred = self.model(
                torch.tensor(normalized, dtype=torch.float32)[None, :, None]
            ).item()
        return {
            "method": "count_lstm",
            "status": "experimental",
            "horizon_seconds": c["horizon"],
            "count": round(max(0, pred * c["std"] + c["mean"]), 1),
            "training_source": c["source"],
        }


class PersistenceForecast(TrendForecast):
    def predict(self, horizon=3):
        return {
            "method": "persistence_baseline",
            "status": "comparison_baseline",
            "horizon_seconds": horizon,
            "count": round(self.history[-1][1], 1) if self.history else None,
        }


def create_forecaster(mode, checkpoint):
    if mode == "lstm":
        return CountForecaster(checkpoint)
    if mode == "linear":
        return TrendForecast()
    return PersistenceForecast()
