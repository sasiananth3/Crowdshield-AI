"""Track-local motion in normalized image coordinates/second, not physical speed."""

from collections import deque
import numpy as np


class MotionAnalyzer:
    def __init__(self, dynamics_window_seconds=5.0):
        self.tracks = {}
        self.dynamics = deque()
        self.dynamics_window_seconds = dynamics_window_seconds

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
        mean_speed = float(np.mean(speeds)) if speeds else None
        self.dynamics.append(
            {
                "timestamp": timestamp,
                "count": len(detections),
                "tracked_people": len(active),
                "mean_speed_normalized": mean_speed,
            }
        )
        while (
            self.dynamics
            and timestamp - self.dynamics[0]["timestamp"]
            > self.dynamics_window_seconds
        ):
            self.dynamics.popleft()
        recent_peak_count = max(item["count"] for item in self.dynamics)
        count_drop_fraction = (
            (recent_peak_count - len(detections)) / recent_peak_count
            if recent_peak_count
            else 0.0
        )
        group_speeds = [
            item["mean_speed_normalized"]
            for item in self.dynamics
            if item["tracked_people"] >= 3
            and item["mean_speed_normalized"] is not None
        ]
        return {
            "motion_available": len(speeds) > 0,
            "tracked_people": len(active),
            "mean_speed_normalized": round(mean_speed, 4)
            if mean_speed is not None
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
            # Trackers can lose identities during rapid dispersal. Keep a short,
            # time-based count/movement window so that loss of track IDs does not
            # immediately erase the developing crowd-dynamics signal.
            "recent_peak_count": recent_peak_count,
            "count_drop_fraction": round(count_drop_fraction, 3),
            "recent_group_peak_speed_normalized": round(max(group_speeds), 4)
            if group_speeds
            else None,
        }
