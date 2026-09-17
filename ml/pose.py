"""Optional MediaPipe body landmarks. Pose is not evidence of panic or intent."""

from pathlib import Path
import numpy as np


class PoseAnalyzer:
    def __init__(self, checkpoint):
        self.engine = None
        self.error = None
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

    def analyze(self, frame, detections):
        if self.engine is None:
            return {"status": "unavailable", "reason": self.error, "poses": []}
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
            "status": "experimental_landmarks_only",
            "poses": poses,
            "note": "Up to six visible people; no trained fall/pushing/panic classifier",
        }

    def close(self):
        if self.engine is not None:
            self.engine.close()
