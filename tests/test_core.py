from pathlib import Path
import threading

import numpy as np

from backend.app.worker import Manager
from ml.forecast import TrendForecast
from ml.kinematics import MotionAnalyzer
from ml.risk import AlertDebouncer, assess


def test_forecast_uses_time_not_frame_index():
    forecast = TrendForecast()
    for t in [0, 2, 4, 6, 8]:
        forecast.update(t, 10 + t)
    assert forecast.predict(3)["count"] == 21


def test_no_risk_without_capacity():
    assert assess(20, None, {}, {})["tier"] == "Uncalibrated"


def test_risk_is_bounded_and_not_probability():
    risk = assess(
        100, 5, {"reversal_fraction": 1, "sudden_motion_fraction": 1}, {"count": 200}
    )
    assert 0 <= risk["score"] <= 100
    assert risk["is_probability"] is False


def test_alert_requires_persistence_and_allows_escalation():
    engine = AlertDebouncer()
    assert not engine.update("z", 0, {"tier": "High"})
    assert not engine.update("z", 2, {"tier": "High"})
    assert engine.update("z", 3, {"tier": "High"})
    assert not engine.update("z", 4, {"tier": "High"})
    assert not engine.update("z", 5, {"tier": "Critical"})
    assert engine.update("z", 8, {"tier": "Critical"})


def test_motion_has_unknown_warmup_and_finite_values():
    analyzer = MotionAnalyzer()
    assert analyzer.update([], 0)["mean_speed_normalized"] is None
    assert not analyzer.update([{"track_id": 1, "box": [0.1, 0.1, 0.2, 0.4]}], 1)[
        "motion_available"
    ]
    result = analyzer.update([{"track_id": 1, "box": [0.2, 0.1, 0.3, 0.4]}], 2)
    assert result["motion_available"]
    assert np.isclose(result["mean_speed_normalized"], 0.1)


def test_density_target_conserves_count():
    from training.train_density import read_data, discover
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data/raw/ShanghaiTech"
    if not root.exists():
        import pytest

        pytest.skip("Optional real dataset not downloaded")
    samples = read_data(discover(root) / "train_data", limit=5)
    for _, _, density, count in samples:
        assert abs(float(density.sum()) - count) < 0.001


def test_frame_publish_retries_transient_windows_lock(tmp_path, monkeypatch):
    manager = Manager.__new__(Manager)
    manager.runtime = tmp_path
    manager.lock = threading.RLock()
    folder = tmp_path / "jobs" / "job"
    folder.mkdir(parents=True)
    (folder / "frame.jpg").write_bytes(b"old")

    original_replace = Path.replace
    attempts = 0

    def transient_access_denied(path, target):
        nonlocal attempts
        if path.name == "frame-next.jpg" and attempts < 2:
            attempts += 1
            raise PermissionError(5, "Access is denied", str(target))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", transient_access_denied)
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    manager._publish_image(folder, "frame.jpg", image)

    assert attempts == 2
    assert manager.frame_bytes("job").startswith(b"\xff\xd8")
    assert not (folder / "frame-next.jpg").exists()
