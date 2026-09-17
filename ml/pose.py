"""Optional MediaPipe body landmarks. Pose is not evidence of panic or intent."""

from pathlib import Path
from functools import lru_cache
import numpy as np


class PoseKinematics:
    """Observable body movement rules, not a trained incident classifier."""

    def __init__(self):
        self.history = {}

    def update(self, poses, timestamp, aspect=1.0):
        for pose in poses:
            joints = {p["id"]: np.array([p["x"] * aspect, p["y"]]) for p in pose["joints"]}
            track_id = pose["track_id"]
            if not all(i in joints for i in [11, 12, 23, 24]):
                pose["features"] = {"available": False, "reason": "Torso landmarks occluded"}
                continue
            shoulder = (joints[11] + joints[12]) / 2
            hip = (joints[23] + joints[24]) / 2
            torso = shoulder - hip
            lean = float(np.degrees(np.arctan2(abs(torso[0]), max(1e-6, -torso[1]))))
            features = {"available": True, "temporal_available": False, "torso_lean_degrees": round(lean, 1), "leaning": lean > 45}
            previous = self.history.get(track_id) if track_id is not None else None
            velocity = None
            if previous is not None and 0 < timestamp - previous[0] <= 2:
                dt = timestamp - previous[0]
                velocity = (hip - previous[1]) / dt
                speed = float(np.linalg.norm(velocity))
                before = previous[2]
                reversal, acceleration = False, None
                if before is not None:
                    prior_speed = float(np.linalg.norm(before))
                    reversal = speed > 0.01 and prior_speed > 0.01 and float(np.dot(velocity, before) / (speed * prior_speed)) < -0.5
                    acceleration = float(np.linalg.norm(velocity - before) / dt)
                features.update(temporal_available=True, speed_normalized=round(speed, 4), stalled=speed < 0.005, reversal=bool(reversal), abrupt_motion=bool(acceleration is not None and acceleration > 0.08), acceleration_normalized=round(acceleration, 4) if acceleration is not None else None)
            if track_id is not None:
                self.history[track_id] = (timestamp, hip, velocity)
            pose["features"] = features
        self.history = {k: v for k, v in self.history.items() if timestamp - v[0] <= 2}
        return poses


def summarize_poses(poses, detected_count=0):
    valid = [p["features"] for p in poses if p.get("features", {}).get("available")]
    temporal = [p for p in valid if p.get("temporal_available")]
    def fraction(items, key):
        return round(float(np.mean([p[key] for p in items])), 3) if items else None
    return {"available": bool(valid), "temporal_available": bool(temporal), "observed_people": len(valid), "temporal_people": len(temporal), "coverage": round(len(valid) / detected_count, 3) if detected_count else 0, "stalled_fraction": fraction(temporal, "stalled"), "reversal_fraction": fraction(temporal, "reversal"), "abrupt_motion_fraction": fraction(temporal, "abrupt_motion"), "leaning_fraction": fraction(valid, "leaning")}


@lru_cache(maxsize=4)
def pose_capability(checkpoint):
    analyzer = PoseAnalyzer(checkpoint)
    result = {"available": analyzer.engine is not None, "reason": analyzer.error}
    analyzer.close()
    return result


class PoseAnalyzer:
    def __init__(self, checkpoint):
        self.engine = None
        self.error = None
        self.kinematics = PoseKinematics()
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
                    base_options=python.BaseOptions(model_asset_path=str(checkpoint), delegate=python.BaseOptions.Delegate.CPU),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                )
            )
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            self.error = str(exc)

    def analyze(self, frame, detections, timestamp=0):
        if self.engine is None:
            return {"status": "unavailable", "reason": self.error, "poses": [], "summary": summarize_poses([])}
        h, w = frame.shape[:2]
        poses = []
        for item in sorted(detections, key=lambda d: d["confidence"], reverse=True)[:6]:
            x1, y1, x2, y2 = item["box"]
            left, top = max(0, int(x1 * w)), max(0, int(y1 * h))
            right, bottom = min(w, int(x2 * w)), min(h, int(y2 * h))
            if right - left < 24 or bottom - top < 48:
                continue
            crop = np.ascontiguousarray(frame[top:bottom, left:right, ::-1])
            result = self.engine.detect(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop)
            )
            if result.pose_landmarks:
                # Exclude face landmarks; no embeddings or identity recognition.
                joints = [
                    {
                        "id": i,
                        "x": round((left + p.x * (right - left)) / w, 5),
                        "y": round((top + p.y * (bottom - top)) / h, 5),
                        "visibility": round(p.visibility, 3),
                    }
                    for i, p in enumerate(result.pose_landmarks[0])
                    if i >= 11 and p.visibility > 0.5
                ]
                if joints:
                    poses.append({"track_id": item["track_id"], "joints": joints})
        return {
            "status": "experimental_kinematics",
            "poses": self.kinematics.update(poses, timestamp, w / h),
            "summary": summarize_poses(poses, len(detections)),
            "note": "Up to six visible people. Rules measure motion and torso lean; no trained pushing/falling/panic classifier.",
        }

    def reset(self):
        self.kinematics = PoseKinematics()

    def close(self):
        if self.engine is not None:
            self.engine.close()
