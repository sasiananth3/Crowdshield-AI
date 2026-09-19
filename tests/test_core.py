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


def test_multi_person_motion_can_warn_without_capacity():
    risk = assess(
        6,
        None,
        {"tracked_people": 5, "mean_speed_normalized": 0.06},
        {},
    )
    assert risk["tier"] == "Moderate"
    assert risk["score"] is None
    assert risk["method"] == "crowd_dynamics_v1"


def test_single_fast_person_does_not_create_motion_warning():
    risk = assess(
        1,
        None,
        {"tracked_people": 1, "mean_speed_normalized": 0.2},
        {},
    )
    assert risk["tier"] == "Uncalibrated"


def test_multi_person_motion_sets_moderate_floor_with_capacity():
    risk = assess(
        3,
        100,
        {"tracked_people": 3, "mean_speed_normalized": 0.04},
        {},
    )
    assert risk["tier"] == "Moderate"
    assert risk["score"] == 35.0


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


def test_rapid_dispersal_uses_shorter_persistence():
    engine = AlertDebouncer()
    risk = {"tier": "Moderate", "alert_persistence_seconds": 1.0}
    assert not engine.update("z", 17.5, risk)
    assert not engine.update("z", 18.0, risk)
    assert engine.update("z", 18.5, risk)


def test_motion_window_preserves_dispersal_signal_when_tracks_drop():
    analyzer = MotionAnalyzer()

    def people(count, offset):
        return [
            {
                "track_id": index + 1,
                "box": [0.05 * index + offset, 0.1, 0.05 * index + 0.04 + offset, 0.4],
            }
            for index in range(count)
        ]

    analyzer.update(people(6, 0.0), 0.0)
    analyzer.update(people(6, 0.01), 1.0)
    analyzer.update(people(6, 0.08), 2.0)
    motion = analyzer.update(people(2, 0.09), 2.5)
    risk = assess(2, None, motion, {})

    assert motion["recent_peak_count"] == 6
    assert motion["count_drop_fraction"] >= 0.5
    assert motion["recent_group_peak_speed_normalized"] >= 0.04
    assert risk["tier"] == "Moderate"
    assert risk["alert_persistence_seconds"] == 1.0
    assert "dispersal" in risk["reasons"][0].lower()


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
