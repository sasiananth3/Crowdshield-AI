"""Bounded single-video worker. CPU inference stays off the HTTP event loop."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np
from ml.detection import PersonDetector
from ml.density import DensityEstimator
from ml.forecast import create_forecaster
from ml.kinematics import MotionAnalyzer
from ml.pose import PoseAnalyzer
from ml.risk import AlertDebouncer, assess


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def within(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0


class Manager:
    def __init__(self, root, runtime, store):
        self.root, self.runtime, self.store = Path(root), Path(runtime), store
        self.lock = threading.RLock()
        self.jobs = {job["id"]: job for job in store.jobs()}
        self.threads = {}
        self.stops = {}
        for job in self.jobs.values():
            if job["status"] in {"processing", "stopping"}:
                job["status"] = "interrupted"
                job["error"] = "Server restarted; create a new analysis to resume"
                self.store.save_job(job)

    def create(self, path, name):
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width, height = cap.get(3), cap.get(4)
            if (
                not cap.isOpened()
                or not np.isfinite(fps)
                or fps <= 0
                or not np.isfinite(frames)
                or frames <= 0
            ):
                raise ValueError("Unsupported, corrupt, or empty video")
            if frames / fps > 3600 or width * height > 3840 * 2160:
                raise ValueError("Use a video under 60 minutes and no larger than 4K")
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Cannot decode the first video frame")
        finally:
            cap.release()
        job_id = uuid.uuid4().hex
        folder = self.runtime / "jobs" / job_id
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "frame.jpg"), frame)
        job = {
            "id": job_id,
            "name": name,
            "path": str(path),
            "status": "ready",
            "created_at": utcnow(),
            "duration_seconds": round(frames / fps, 2),
            "source_fps": fps,
            "width": int(width),
            "height": int(height),
            "progress": 0,
            "processed_samples": 0,
            "latest": None,
            "error": None,
            "zones": [],
        }
        with self.lock:
            self.jobs[job_id] = job
        self.store.save_job(job)
        return self.public(job)

    def public(self, job):
        return {k: copy.deepcopy(v) for k, v in job.items() if k != "path"}

    def get(self, job_id):
        with self.lock:
            return self.public(self.jobs[job_id]) if job_id in self.jobs else None

    def frame_bytes(self, job_id, heatmap=False):
        """Read a published frame without racing the Windows file replacement."""
        name = "heatmap.jpg" if heatmap else "frame.jpg"
        path = self.runtime / "jobs" / job_id / name
        with self.lock:
            return path.read_bytes() if path.is_file() else None

    def _publish_image(self, folder, name, image):
        """Encode and atomically publish an image, retrying transient file locks."""
        target = folder / name
        temporary = folder / f"{target.stem}-next{target.suffix}"
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"Could not encode {name}")
        for attempt in range(7):
            try:
                # API frame reads use the same lock. The retry also covers short
                # locks held by antivirus scanners and filesystem indexers.
                with self.lock:
                    temporary.replace(target)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(min(0.01 * (2**attempt), 0.2))

    def start(
        self, job_id, zones, enable_pose, sample_fps, forecast_mode="persistence"
    ):
        with self.lock:
            if any(
                j["status"] in {"processing", "stopping"} for j in self.jobs.values()
            ):
                raise ValueError(
                    "One analysis can run at a time on this local prototype"
                )
            job = self.jobs[job_id]
            if job["status"] != "ready":
                raise ValueError(
                    "This analysis has already run; upload again to use different settings"
                )
            if not (self.root / "models/yolov8n.pt").exists():
                raise ValueError(
                    "YOLOv8 weights missing. Run python scripts/setup_models.py"
                )
            if (
                forecast_mode == "lstm"
                and not (self.root / "models/forecast.pt").exists()
            ):
                raise ValueError("LSTM checkpoint missing; train or install it first")
            job.update(
                status="processing",
                zones=zones,
                enable_pose=enable_pose,
                sample_fps=sample_fps,
                forecast_mode=forecast_mode,
            )
            self.stops[job_id] = threading.Event()
            self.store.save_job(job)
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self.threads[job_id] = thread
            thread.start()
        return self.get(job_id)

    def stop(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] == "processing":
                job["status"] = "stopping"
                self.stops[job_id].set()
                self.store.save_job(job)
        return self.get(job_id)

    def _run(self, job_id):
        import torch

        torch.set_num_threads(2)
        job = self.jobs[job_id]
        cap = None
        pose = None
        try:
            detector = PersonDetector(self.root / "models/yolov8n.pt")
            density = DensityEstimator(self.root / "models/density.pt")
            pose = (
                PoseAnalyzer(self.root / "models/pose_landmarker_lite.task")
                if job["enable_pose"]
                else None
            )
            motion = MotionAnalyzer()
            forecasters = {
                z["id"]: create_forecaster(
                    job["forecast_mode"], self.root / "models/forecast.pt"
                )
                for z in job["zones"]
            }
            debouncer = AlertDebouncer()
            cap = cv2.VideoCapture(job["path"])
            fps = job["source_fps"]
            stride = max(1, round(fps / job["sample_fps"]))
            frame_id = 0
            started = time.perf_counter()
            folder = self.runtime / "jobs" / job_id
            previous_hist = None
            while not self.stops[job_id].is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                current_id = frame_id
                frame_id += 1
                if current_id % stride:
                    continue
                timestamp = current_id / fps
                sample_started = time.perf_counter()
                # A cut invalidates image-space trajectories and temporal history.
                hist = cv2.calcHist([frame], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)
                cv2.normalize(hist, hist)
                scene_cut = (
                    previous_hist is not None
                    and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    > 0.55
                )
                if scene_cut:
                    detector = PersonDetector(self.root / "models/yolov8n.pt")
                    motion = MotionAnalyzer()
                    forecasters = {
                        z["id"]: create_forecaster(
                            job["forecast_mode"], self.root / "models/forecast.pt"
                        )
                        for z in job["zones"]
                    }
                    debouncer = AlertDebouncer()
                previous_hist = hist
                detections = detector.detect(frame)
                global_motion = motion.update(detections, timestamp)
                estimated = density.predict(frame)
                body = (
                    pose.analyze(frame, detections)
                    if pose
                    else {"status": "disabled", "poses": []}
                )
                zone_results = []
                for zone in job["zones"]:
                    selected = [
                        d
                        for d in detections
                        if within(
                            ((d["box"][0] + d["box"][2]) / 2, d["box"][3]),
                            zone["polygon"],
                        )
                    ]
                    count = len(selected)
                    # Per-zone trajectories are assessed independently.
                    zone_motion_engine = getattr(motion, "zone_engines", None)
                    if zone_motion_engine is None:
                        motion.zone_engines = {}
                    zone_motion = motion.zone_engines.setdefault(
                        zone["id"], MotionAnalyzer()
                    ).update(selected, timestamp)
                    forecasters[zone["id"]].update(timestamp, count)
                    forecast = forecasters[zone["id"]].predict()
                    risk = assess(count, zone["capacity"], zone_motion, forecast)
                    entry = {
                        "id": zone["id"],
                        "name": zone["name"],
                        "count": count,
                        "capacity": zone["capacity"],
                        "people_per_m2": round(count / zone["area_m2"], 3)
                        if zone.get("area_m2")
                        else None,
                        "motion": zone_motion,
                        "forecast": forecast,
                        "risk": risk,
                    }
                    zone_results.append(entry)
                    if debouncer.update(zone["id"], timestamp, risk):
                        self.store.add_alert(
                            {
                                "id": uuid.uuid4().hex,
                                "job_id": job_id,
                                "zone": zone["name"],
                                "created_at": utcnow(),
                                "video_timestamp": round(timestamp, 2),
                                **risk,
                            }
                        )
                # Do not sum detector and density estimates: they count the same people.
                sample = {
                    "timestamp": round(timestamp, 3),
                    "detected_people": len(detections),
                    "detections": detections,
                    "density_estimate": round(estimated["count"], 1)
                    if estimated
                    else None,
                    "density_status": estimated["status"]
                    if estimated
                    else "unavailable",
                    "motion": global_motion,
                    "pose": body,
                    "zones": zone_results,
                    "scene_cut": bool(scene_cut),
                    "inference_ms": round(
                        (time.perf_counter() - sample_started) * 1000, 1
                    ),
                }
                h, w = frame.shape[:2]
                for zone in job["zones"]:
                    points = (np.array(zone["polygon"]) * np.array([w, h])).astype(
                        np.int32
                    )
                    cv2.polylines(frame, [points], True, (255, 195, 70), 1)
                for item in detections:
                    x1, y1, x2, y2 = (
                        np.array(item["box"]) * np.array([w, h, w, h])
                    ).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (94, 210, 72), 1)
                    cv2.putText(
                        frame,
                        f"#{item['track_id'] or '-'}",
                        (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (94, 210, 72),
                        1,
                    )
                # Body-only pose overlay; no face landmarks or identity features.
                connections = [
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                    (11, 23),
                    (12, 24),
                    (23, 24),
                    (23, 25),
                    (25, 27),
                    (24, 26),
                    (26, 28),
                ]
                for skeleton in body.get("poses", []):
                    joints = {
                        p["id"]: (int(p["x"] * w), int(p["y"] * h))
                        for p in skeleton["joints"]
                    }
                    for a, b in connections:
                        if a in joints and b in joints:
                            cv2.line(frame, joints[a], joints[b], (255, 205, 40), 1)
                self._publish_image(folder, "frame.jpg", frame)
                if estimated:
                    density_map = estimated["map"]
                    normalized = (
                        density_map / max(1e-8, float(density_map.max())) * 255
                    ).astype(np.uint8)
                    heat = cv2.applyColorMap(
                        cv2.resize(normalized, (w, h)), cv2.COLORMAP_TURBO
                    )
                    self._publish_image(
                        folder,
                        "heatmap.jpg",
                        cv2.addWeighted(frame, 0.45, heat, 0.55, 0),
                    )
                self.store.sample(job_id, sample)
                with self.lock:
                    job.update(
                        latest=sample,
                        processed_samples=job["processed_samples"] + 1,
                        progress=min(99.9, 100 * timestamp / job["duration_seconds"]),
                        processing_fps=round(
                            job["processed_samples"] / (time.perf_counter() - started),
                            2,
                        ),
                    )
                    self.store.save_job(job)
            with self.lock:
                job["status"] = (
                    "stopped" if self.stops[job_id].is_set() else "completed"
                )
                if job["status"] == "completed":
                    job["progress"] = 100
        except Exception as exc:
            with self.lock:
                job.update(status="failed", error=str(exc))
        finally:
            if cap is not None:
                cap.release()
            if pose is not None:
                pose.close()
            self.store.save_job(job)

    def shutdown(self):
        for event in self.stops.values():
            event.set()
        for thread in self.threads.values():
            thread.join(timeout=30)
