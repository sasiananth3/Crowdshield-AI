"""Track-local motion in normalized image coordinates/second, not physical speed."""

from collections import deque
import numpy as np


class MotionAnalyzer:
    def __init__(self):
        self.tracks = {}

    def update(self, detections, timestamp):
        active = []
        speeds, reversals = [], []
        for item in detections:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = item["box"]
            position = np.array([(x1 + x2) / 2, y2])
            history = self.tracks.setdefault(track_id, deque(maxlen=15))
            if history and timestamp > history[-1][0]:
                velocity = (position - history[-1][1]) / (timestamp - history[-1][0])
                speed = float(np.linalg.norm(velocity))
                active.append(track_id)
                speeds.append(speed)
                previous = history[-1][2]
                if np.linalg.norm(previous) > 0.01 and speed > 0.01:
                    reversals.append(
                        float(
                            np.dot(velocity, previous)
                            / (speed * np.linalg.norm(previous))
                            < -0.5
                        )
                    )
            else:
                velocity = np.zeros(2)
            history.append((timestamp, position, velocity))
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v[-1][0] < 5}
        return {
            "motion_available": len(speeds) > 0,
            "tracked_people": len(active),
            "mean_speed_normalized": round(float(np.mean(speeds)), 4)
            if speeds
            else None,
            "stalled_fraction": round(float(np.mean(np.array(speeds) < 0.005)), 3)
            if speeds
            else None,
            "reversal_fraction": round(float(np.mean(reversals)), 3)
            if reversals
            else 0.0,
            "sudden_motion_fraction": round(float(np.mean(np.array(speeds) > 0.15)), 3)
            if speeds
            else None,
        }
