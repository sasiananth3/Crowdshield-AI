"""Run on localhost only: python -m uvicorn backend.app.main:app --port 8000"""

import asyncio
from contextlib import asynccontextmanager
import csv
import io
import json
import os
from pathlib import Path
from typing import Literal
import uuid
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.store import Store
from backend.app.worker import Manager

ROOT = Path(__file__).resolve().parents[2]


class Zone(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=80)
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=16)
    capacity: int | None = Field(default=None, ge=1, le=100000)
    area_m2: float | None = Field(default=None, gt=0, le=1000000)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if any(not 0 <= v <= 1 for point in value for v in point):
            raise ValueError("Polygon coordinates must be normalized between 0 and 1")
        area = (
            abs(
                sum(
                    value[i][0] * value[(i + 1) % len(value)][1]
                    - value[(i + 1) % len(value)][0] * value[i][1]
                    for i in range(len(value))
                )
            )
            / 2
        )
        if area < 0.001:
            raise ValueError("Polygon area is too small")
        return value


class StartRequest(BaseModel):
    zones: list[Zone] = Field(min_length=1, max_length=8)
    enable_pose: bool = False
    sample_fps: float = Field(default=2, ge=0.5, le=5, allow_inf_nan=False)
    forecast_mode: Literal["persistence", "linear", "lstm"] = "persistence"
    verify_alerts: bool = False

    @field_validator("zones")
    @classmethod
    def unique_ids(cls, zones):
        if len({z.id for z in zones}) != len(zones):
            raise ValueError("Zone IDs must be unique")
        return zones


def create_app(runtime=None):
    runtime = Path(runtime or os.getenv("CROWDSHIELD_RUNTIME", str(ROOT / "runtime")))
    runtime.mkdir(parents=True, exist_ok=True)
    store = Store(runtime / "crowdshield.sqlite3")
    manager = Manager(ROOT, runtime, store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(manager.shutdown)
        store.close()

    app = FastAPI(title="CrowdShield AI", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.store = store
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_write_origin(request, call_next):
        origin = request.headers.get("origin")
        allowed = {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        }
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and origin not in allowed
        ):
            return Response("Cross-origin writes are not permitted", status_code=403)
        return await call_next(request)

    def require_job(job_id):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Analysis not found")
        return job

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "application": "CrowdShield AI",
            "mode": "local_research_prototype",
            "models": {
                "detector": (ROOT / "models/yolov8n.pt").exists(),
                "density": (ROOT / "models/density.pt").exists(),
                "forecast": (ROOT / "models/forecast.pt").exists(),
                "pose": (ROOT / "models/pose_landmarker_lite.task").exists(),
            },
            "alert_verifier": manager.alert_verifier.health(),
            "demo_available": (ROOT / "data/raw/umn-demo.avi").exists(),
            "disclaimer": "Experimental decision support. Not validated for public-safety use.",
        }

    @app.get("/api/evaluation")
    def evaluation():
        output = {}
        for name in ["density", "forecast"]:
            path = ROOT / f"reports/{name}_evaluation.json"
            output[name] = json.loads(path.read_text()) if path.exists() else None
        return output

    @app.post("/api/videos", status_code=201)
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise HTTPException(415, "Use MP4, AVI, MOV, MKV or WebM")
        directory = runtime / "uploads"
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + suffix)
        try:
            size = 0
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 250 * 1024 * 1024:
                        raise HTTPException(413, "Maximum upload size is 250 MB")
                    target.write(chunk)
            return await asyncio.to_thread(
                manager.create, path, Path(file.filename).name
            )
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.post("/api/demo", status_code=201)
    def demo():
        path = ROOT / "data/raw/umn-demo.avi"
        if not path.exists():
            raise HTTPException(
                409,
                "Download the sample: python scripts/download_data.py --only umn_demo",
            )
        return manager.create(path, "UMN academic demonstration")

    @app.get("/api/jobs")
    def jobs():
        with manager.lock:
            return [manager.public(j) for j in reversed(list(manager.jobs.values()))]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return require_job(job_id)

    @app.post("/api/jobs/{job_id}/start")
    def start(job_id: str, body: StartRequest):
        require_job(job_id)
        try:
            return manager.start(
                job_id,
                [z.model_dump() for z in body.zones],
                body.enable_pose,
                body.sample_fps,
                body.forecast_mode,
                body.verify_alerts,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: str):
        require_job(job_id)
        return manager.stop(job_id)

    @app.get("/api/jobs/{job_id}/history")
    def history(job_id: str):
        require_job(job_id)
        return store.history(job_id)

    @app.get("/api/jobs/{job_id}/frame")
    def frame(job_id: str, heatmap: bool = False):
        require_job(job_id)
        content = manager.frame_bytes(job_id, heatmap)
        if content is None:
            raise HTTPException(404, "Frame not available yet")
        return Response(
            content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/alerts")
    def alerts(job_id: str | None = None):
        return store.alerts(job_id)

    @app.post("/api/alerts/{alert_id}/acknowledge")
    def acknowledge(alert_id: str):
        if not store.acknowledge(alert_id):
            raise HTTPException(404, "Alert not found")
        return {"acknowledged": True}

    @app.get("/api/alert-verifications")
    def alert_verifications(job_id: str | None = None):
        return store.alert_verifications(job_id)

    @app.get("/api/jobs/{job_id}/export")
    def export(job_id: str):
        require_job(job_id)
        content = io.StringIO()
        writer = csv.writer(content)
        writer.writerow(
            [
                "timestamp_seconds",
                "detected_people",
                "density_estimate",
                "zone",
                "zone_count",
                "risk_score",
                "risk_tier",
                "forecast_count",
                "forecast_method",
            ]
        )
        for sample in store.history(job_id):
            for zone in sample["zones"]:
                # Mitigate spreadsheet formula injection in user-provided zone names.
                name = zone["name"]
                if name.startswith(("=", "+", "-", "@", "\t", "\r")):
                    name = "'" + name
                writer.writerow(
                    [
                        sample["timestamp"],
                        sample["detected_people"],
                        sample["density_estimate"],
                        name,
                        zone["count"],
                        zone["risk"]["score"],
                        zone["risk"]["tier"],
                        zone["forecast"]["count"],
                        zone["forecast"]["method"],
                    ]
                )
        return Response(
            content.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="crowdshield-{job_id}.csv"'
            },
        )

    @app.websocket("/ws/jobs/{job_id}")
    async def websocket(ws: WebSocket, job_id: str):
        origin = ws.headers.get("origin")
        if origin and origin not in {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        }:
            await ws.close(code=1008)
            return
        if manager.get(job_id) is None:
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                await ws.send_json(manager.get(job_id))
                await asyncio.sleep(0.75)
        except WebSocketDisconnect:
            pass

    dist = ROOT / "frontend/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:

        @app.get("/")
        def index():
            return {
                "message": "Build the dashboard with cd frontend && npm install && npm run build",
                "api_docs": "/docs",
            }

    return app


app = create_app()
