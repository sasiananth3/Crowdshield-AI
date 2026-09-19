"""Optional MediaPipe body landmarks and track-aware kinematic scoring."""

from collections import deque
from pathlib import Path
import numpy as np


def compute_kinematic_score(history):
    """Compute a 0..1 heuristic from recent normalized hip-midpoint positions."""
    if len(history) < 10:
        return 0.0
    positions = np.asarray(history[-10:], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        return 0.0
    velocities = np.diff(positions, axis=0)
    displacement = float(np.linalg.norm(velocities, axis=1).sum())
    stalling = float(np.clip(1.0 - displacement / 0.05, 0.0, 1.0))
    reversal = 0
    previous = None
    for velocity in velocities:
        if np.linalg.norm(velocity) <= 1e-3:
            continue
        if previous is not None:
            denom = np.linalg.norm(previous) * np.linalg.norm(velocity)
            if denom > 1e-8 and float(np.dot(previous, velocity) / denom) < -0.5:
                reversal += 1
        previous = velocity
    return float(0.6 * stalling + 0.4 * min(1.0, reversal / 3.0))


class PoseAnalyzer:
    def __init__(self, checkpoint, history_size=10):
        self.engine = None
        self.error = None
        self.histories = {}
        self.history_size = history_size
        if not Path(checkpoint).exists():
            self.error = "Pose weights not installed"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self.mp = mp
            self.engine = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(checkpoint)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                )
            )
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            self.error = str(exc)

    def analyze(self, frame, detections, timestamp=0.0):
        if self.engine is None:
            return {"status": "unavailable", "reason": self.error, "poses": [], "kinematic_score": None}
        h, w = frame.shape[:2]
        poses, scores = [], []
        active = set()
        for item in sorted(detections, key=lambda d: d["confidence"], reverse=True)[:6]:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = item["box"]
            left, top = max(0, int(x1 * w)), max(0, int(y1 * h))
            right, bottom = min(w, int(x2 * w)), min(h, int(y2 * h))
            if right - left < 24 or bottom - top < 48:
                continue
            crop = np.ascontiguousarray(frame[top:bottom, left:right, ::-1])
            result = self.engine.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
            if not result.pose_landmarks:
                continue
            landmarks = result.pose_landmarks[0]
            joints = [
                {"id": i, "x": round((left + p.x * (right-left)) / w, 5),
                 "y": round((top + p.y * (bottom-top)) / h, 5), "visibility": round(p.visibility, 3)}
                for i, p in enumerate(landmarks) if i >= 11 and p.visibility > 0.5
            ]
            hips = [(j["x"], j["y"]) for j in joints if j["id"] in (23, 24)]
            score = 0.0
            if len(hips) == 2:
                midpoint = tuple(np.mean(np.asarray(hips), axis=0))
                history = self.histories.setdefault(track_id, deque(maxlen=self.history_size))
                if not history or timestamp > history[-1][0]:
                    history.append((float(timestamp), midpoint))
                score = compute_kinematic_score([point for _, point in history])
                scores.append(score)
            active.add(track_id)
            poses.append({"track_id": track_id, "joints": joints, "kinematic_score": round(score, 4)})
        self.histories = {k: v for k, v in self.histories.items() if k in active or (v and timestamp-v[-1][0] < 5)}
        return {
            "status": "experimental_landmarks_with_kinematics",
            "poses": poses,
            "kinematic_score": round(float(np.mean(scores)), 4) if scores else None,
            "note": "Body landmarks drive a heuristic kinematic component; this is not a trained incident classifier",
        }

    def close(self):
        if self.engine is not None:
            self.engine.close()
