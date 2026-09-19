"""Time-aware count forecasting with persistence, linear and multivariate LSTM modes."""

from collections import deque
from pathlib import Path
import numpy as np

FEATURE_NAMES = ["count", "count_delta", "density_mean", "density_max_zone", "kinematic_score"]


class TrendForecast:
    name = "linear_trend_baseline"
    input_size = 1

    def __init__(self, max_history=60):
        self.history = deque(maxlen=max_history)

    def update(self, timestamp, count, features=None):
        if not self.history or timestamp > self.history[-1][0]:
            self.history.append((float(timestamp), float(count), list(features or [count])))

    def predict(self, horizon=10):
        if len(self.history) < 5 or self.history[-1][0] - self.history[0][0] < 4:
            return {"method": self.name, "status": "warming_up", "horizon_seconds": horizon, "count": None}
        values = np.asarray([[r[0], r[1]] for r in self.history])
        t = values[:, 0] - values[-1, 0]
        slope = float(np.polyfit(t, values[:, 1], 1)[0])
        return {"method": self.name, "status": "experimental", "horizon_seconds": horizon,
                "count": round(max(0.0, float(values[-1, 1] + slope * horizon)), 1),
                "slope_per_second": round(slope, 3)}


def make_lstm(input_size=5):
    import torch
    from torch import nn
    class CountLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1]).squeeze(-1)
    return CountLSTM()


class CountForecaster(TrendForecast):
    def __init__(self, checkpoint):
        super().__init__()
        self.model = None
        self.config = {}
        if Path(checkpoint).exists():
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.config = saved["metadata"]
            self.input_size = int(self.config.get("input_size", 1))
            self.model = make_lstm(self.input_size)
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()

    def update(self, timestamp, count, features=None):
        if self.input_size == 1:
            features = [float(count)]
        else:
            features = list(features or [])
            if len(features) != self.input_size:
                raise ValueError(f"Forecast expects {self.input_size} features, received {len(features)}")
        return super().update(timestamp, count, features)

    def predict(self, horizon=10):
        if self.model is None:
            return super().predict(horizon)
        import torch
        c = self.config
        window = int(c["window"])
        if len(self.history) < window or self.history[-1][0] - self.history[-window][0] < window - 1:
            return {"method": "count_lstm", "status": "warming_up", "horizon_seconds": c["horizon"], "count": None}
        times = np.asarray([r[0] for r in self.history], dtype=np.float32)
        values = np.asarray([r[2] for r in self.history], dtype=np.float32)
        target_times = np.arange(window, dtype=np.float32) - (window - 1) + times[-1]
        sequence = np.stack([np.interp(target_times, times, values[:, i]) for i in range(self.input_size)], axis=1)
        mean = np.asarray(c["feature_mean"], dtype=np.float32)
        std = np.maximum(np.asarray(c["feature_std"], dtype=np.float32), 1e-6)
        with torch.inference_mode():
            pred = self.model(torch.tensor((sequence - mean) / std, dtype=torch.float32)[None]).item()
        return {"method": "count_lstm", "status": "experimental", "horizon_seconds": c["horizon"],
                "count": round(max(0.0, pred * float(std[0]) + float(mean[0])), 1),
                "training_source": c["source"], "feature_names": c.get("feature_names", FEATURE_NAMES)}


class PersistenceForecast(TrendForecast):
    name = "persistence_baseline"
    def predict(self, horizon=3):
        return {"method": self.name, "status": "comparison_baseline", "horizon_seconds": horizon,
                "count": round(self.history[-1][1], 1) if self.history else None}


def create_forecaster(mode, checkpoint):
    if mode == "lstm":
        return CountForecaster(checkpoint)
    if mode == "linear":
        return TrendForecast()
    return PersistenceForecast()
