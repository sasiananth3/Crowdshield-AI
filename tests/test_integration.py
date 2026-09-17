"""Opt-in real-weight test: CROWDSHIELD_INTEGRATION=1 python -m pytest -q."""

import os
from pathlib import Path
import time
import cv2
import pytest
from fastapi.testclient import TestClient
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.getenv("CROWDSHIELD_INTEGRATION") != "1",
    reason="Opt-in real-model integration test",
)
def test_real_video_to_models_alerts_and_export(tmp_path):
    source = ROOT / "data/raw/umn-demo.avi"
    assert source.exists(), "Download UMN demo before running the integration test"
    clip = tmp_path / "clip.avi"
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for _ in range(450):
        ok, frame = cap.read()
        assert ok
        writer.write(frame)
    cap.release()
    writer.release()
    with TestClient(create_app(tmp_path / "runtime")) as client:
        with clip.open("rb") as video:
            uploaded = client.post(
                "/api/videos", files={"file": ("clip.avi", video, "video/x-msvideo")}
            )
        assert uploaded.status_code == 201
        job_id = uploaded.json()["id"]
        url = f"/api/jobs/{job_id}"
        started = client.post(
            url + "/start",
            json={
                "zones": [
                    {
                        "id": "main",
                        "name": "Test area",
                        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "capacity": 10,
                    }
                ],
                "forecast_mode": "lstm",
                "enable_pose": True,
            },
        )
        assert started.status_code == 200
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = client.get(url).json()
            if job["status"] not in {"processing", "stopping"}:
                break
            time.sleep(0.2)
        assert job["status"] == "completed", job.get("error")
        history = client.get(url + "/history").json()
        assert len(history) == 30
        assert all(s["density_estimate"] is not None for s in history)
        assert any(s["detected_people"] > 0 for s in history)
        assert history[-1]["zones"][0]["forecast"]["method"] == "count_lstm"
        assert history[-1]["zones"][0]["forecast"]["count"] is not None
        for endpoint in ["/frame", "/frame?heatmap=true", "/export"]:
            assert client.get(url + endpoint).status_code == 200
        alerts = client.get("/api/alerts").json()
        assert alerts
        assert (
            client.post(f"/api/alerts/{alerts[0]['id']}/acknowledge").status_code == 200
        )
        assert client.get("/api/alerts").json()[0]["acknowledged"]
