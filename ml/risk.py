"""Experimental, configurable occupancy/motion score; NOT a probability of harm."""

import math

TIERS = [(80, "Critical"), (60, "High"), (35, "Moderate"), (0, "Safe")]


def assess(count, capacity, motion, forecast, density_count=None, pose=None):
    if not capacity or capacity <= 0:
        return {
            "score": None,
            "tier": "Uncalibrated",
            "reasons": [
                "Set a zone reference capacity to enable the experimental score"
            ],
            "action": "Review camera coverage and configure the zone.",
            "method": "late_fusion_v2",
        }
    occupancy_count = density_count if density_count is not None else count
    occupancy = max(0.0, occupancy_count / capacity)
    density_score = min(1.0, occupancy)
    reversal = motion.get("reversal_fraction") or 0.0
    sudden = motion.get("sudden_motion_fraction") or 0.0
    stalled = motion.get("stalled_fraction") or 0.0
    trajectory_score = min(
        1.0, 0.6 * reversal + 0.3 * sudden + 0.1 * stalled * min(occupancy, 1)
    )
    pose = pose or {}
    pose_score = None
    if pose.get("temporal_available"):
        pose_score = min(1.0, 0.45 * (pose.get("reversal_fraction") or 0) + 0.35 * (pose.get("abrupt_motion_fraction") or 0) + 0.1 * (pose.get("stalled_fraction") or 0) * min(occupancy, 1) + 0.1 * (pose.get("leaning_fraction") or 0))
    pose_weight = 0.5 * min(1.0, pose.get("coverage", 0)) if pose_score is not None else 0.0
    motion_score = (1 - pose_weight) * trajectory_score + pose_weight * (pose_score or 0)
    future = forecast.get("count")
    growth = max(0.0, (future - count) / capacity) if future is not None else 0.0
    score = round(
        min(100.0, 70 * density_score + 20 * motion_score + 10 * min(1.0, growth)), 1
    )
    tier = next(label for threshold, label in TIERS if score >= threshold)
    reasons = [
        f"{'CNN estimated' if density_count is not None else 'Detected'} count is {round(100 * occupancy)}% of the configured reference capacity"
    ]
    if reversal > 0.2:
        reasons.append("Direction reversals observed in tracked trajectories")
    if sudden > 0.2:
        reasons.append("Fast image-space movement observed")
    if stalled > 0.5 and occupancy > 0.6:
        reasons.append("Low movement with elevated occupancy")
    if growth > 0.1:
        reasons.append("Count forecast is rising")
    if pose_score is not None and pose_score > 0.2:
        reasons.append("Body-landmark motion rules contribute to the kinematic score")
    if density_count is None:
        reasons.append("Density CNN unavailable; using detector occupancy")
    action = "Continue observation; this status is not a safety guarantee."
    if tier in {"High", "Critical"}:
        action = "Ask the responsible operator to review this zone and follow the venue's approved crowd-management plan."
    elif tier == "Moderate":
        action = "Review the video and monitor whether occupancy or flow is changing."
    return {
        "score": score,
        "tier": tier,
        "reasons": reasons,
        "action": action,
        "method": "late_fusion_v2",
        "components": {"density": round(70 * density_score, 2), "kinematics": round(20 * motion_score, 2), "forecast": round(10 * min(1.0, growth), 2)},
        "weights": {"density": 0.7, "kinematics": 0.2, "forecast": 0.1},
        "density_source": "cnn" if density_count is not None else "detector_fallback",
        "pose_weight_within_kinematics": round(pose_weight, 3),
        "is_probability": False,
        "validated_for_safety": False,
    }


class AlertDebouncer:
    def __init__(self, persistence_seconds=3.0, cooldown_seconds=30.0):
        self.persistence = persistence_seconds
        self.cooldown = cooldown_seconds
        self.state = {}

    def update(self, zone_id, timestamp, risk):
        tier = risk["tier"]
        previous = self.state.get(zone_id)
        if previous is None or previous["tier"] != tier:
            previous = {
                "tier": tier,
                "since": timestamp,
                "last_alert": previous["last_alert"] if previous else -math.inf,
                "last_tier": previous["last_tier"] if previous else "Safe",
            }
            self.state[zone_id] = previous
        severe = tier in {"Moderate", "High", "Critical"}
        rank = {"Safe": 0, "Moderate": 1, "High": 2, "Critical": 3}
        escalation = rank.get(tier, 0) > rank.get(previous["last_tier"], 0)
        if (
            severe
            and timestamp - previous["since"] >= self.persistence
            and (timestamp - previous["last_alert"] >= self.cooldown or escalation)
        ):
            previous["last_alert"], previous["last_tier"] = timestamp, tier
            return True
        return False
