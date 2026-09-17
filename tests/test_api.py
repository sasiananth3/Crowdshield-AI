from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.store import Store


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "runtime")) as client:
        yield client


def video_bytes(tmp_path):
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(12):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    return path.read_bytes()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "local_research_prototype"


def test_invalid_uploads_are_rejected(client):
    assert (
        client.post("/api/videos", files={"file": ("file.txt", b"hello")}).status_code
        == 415
    )
    assert (
        client.post("/api/videos", files={"file": ("file.mp4", b"corrupt")}).status_code
        == 422
    )


def test_unknown_job(client):
    assert client.get("/api/jobs/not-real").status_code == 404
    assert client.get("/api/jobs/not-real/frame").status_code == 404


def test_upload_and_history(client, tmp_path):
    result = client.post(
        "/api/videos",
        files={"file": ("test.avi", video_bytes(tmp_path), "video/x-msvideo")},
    )
    assert result.status_code == 201
    job = result.json()
    assert job["status"] == "ready"
    assert "path" not in job
    assert (
        client.get(f"/api/jobs/{job['id']}/frame").headers["content-type"]
        == "image/jpeg"
    )
    assert client.get(f"/api/jobs/{job['id']}/history").json() == []
    assert len(client.get("/api/jobs").json()) == 1


def test_zone_validation(client, tmp_path):
    job = client.post(
        "/api/videos", files={"file": ("test.avi", video_bytes(tmp_path))}
    ).json()
    zone = {
        "id": "z",
        "name": "zone",
        "polygon": [[0, 0], [1, 0], [1, 1]],
        "capacity": 0,
    }
    assert (
        client.post(f"/api/jobs/{job['id']}/start", json={"zones": [zone]}).status_code
        == 422
    )
    zone["capacity"] = 1
    zone["polygon"] = [[2, 0], [1, 0], [1, 1]]
    assert (
        client.post(f"/api/jobs/{job['id']}/start", json={"zones": [zone]}).status_code
        == 422
    )


def test_alerts_persist_and_acknowledge(tmp_path):
    path = tmp_path / "test.sqlite3"
    store = Store(path)
    store.add_alert({"id": "a", "job_id": "j", "tier": "High"})
    assert store.acknowledge("a")
    store.close()
    second = Store(path)
    assert second.alerts()[0]["acknowledged"]
    second.close()


def test_websocket_missing_job(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/jobs/unknown"):
            pass


def test_external_origin_write_is_rejected(client):
    assert (
        client.post(
            "/api/demo", headers={"Origin": "https://untrusted.example"}
        ).status_code
        == 403
    )


def test_valid_job_websocket(client, tmp_path):
    job = client.post(
        "/api/videos", files={"file": ("test.avi", video_bytes(tmp_path))}
    ).json()
    with client.websocket_connect(f"/ws/jobs/{job['id']}") as websocket:
        message = websocket.receive_json()
        assert message["id"] == job["id"]
        assert message["status"] == "ready"
