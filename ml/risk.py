"""Experimental late-fusion risk index. It is not a probability of harm."""

import math

TIERS = [(80, "Critical"), (60, "High"), (35, "Moderate"), (0, "Safe")]
WEIGHTS = {"density_score": 0.35, "count_score": 0.25, "kinematic_score": 0.25, "forecast_score": 0.15}


def _box_kinematic_score(motion, occupancy):
    reversal = motion.get("reversal_fraction") or 0.0
    sudden = motion.get("sudden_motion_fraction") or 0.0
    stalled = motion.get("stalled_fraction") or 0.0
    return min(1.0, 0.6 * reversal + 0.3 * sudden + 0.1 * stalled * min(occupancy, 1.0))


def assess(count, capacity, motion, forecast, density_count=None, pose_kinematic_score=None):
    if not capacity or capacity <= 0:
        return {"score": None, "tier": "Uncalibrated",
                "reasons": ["Set a zone reference capacity to enable the experimental score"],
                "action": "Review camera coverage and configure the zone.",
                "method": "heuristic_late_fusion_v2", "is_probability": False, "validated_for_safety": False}
    occupancy = max(0.0, count / capacity)
    count_score = min(1.0, occupancy)
    density_score = min(1.0, max(0.0, density_count) / capacity) if density_count is not None else None
    if pose_kinematic_score is not None:
        kinematic_score = max(0.0, min(1.0, float(pose_kinematic_score)))
        kinematic_source = "pose_landmarks"
    else:
        kinematic_score = _box_kinematic_score(motion, occupancy)
        kinematic_source = "bbox_trajectories"
    future = forecast.get("count") if forecast else None
    growth = max(0.0, (future - count) / capacity) if future is not None else 0.0
    components = {"density_score": density_score, "count_score": count_score,
                  "kinematic_score": kinematic_score, "forecast_score": min(1.0, growth)}
    available = sum(v for k, v in WEIGHTS.items() if components[k] is not None)
    weighted = sum(WEIGHTS[k] * components[k] for k in WEIGHTS if components[k] is not None)
    score = round(100.0 * weighted / max(available, 1e-9), 1)
    tier = next(label for threshold, label in TIERS if score >= threshold)
    reasons = [f"Detected count is {round(100 * occupancy)}% of the configured reference capacity"]
    if density_count is not None and abs(density_count - count) / max(1.0, capacity) > 0.15:
        reasons.append("Density and detection estimates differ materially")
    if kinematic_score > 0.5:
        reasons.append(f"Elevated kinematic signal from {kinematic_source.replace('_', ' ')}")
    if growth > 0.1:
        reasons.append("Count forecast is rising")
    action = "Continue observation; this status is not a safety guarantee."
    if tier in {"High", "Critical"}:
        action = "Ask the responsible operator to review this zone and follow the venue's approved crowd-management plan."
    elif tier == "Moderate":
        action = "Review the video and monitor whether occupancy or flow is changing."
    return {"score": score, "tier": tier, "reasons": reasons, "action": action,
            "method": "heuristic_late_fusion_v2", "is_probability": False,
            "validated_for_safety": False,
            "components": {k: (round(v, 4) if v is not None else None) for k, v in components.items()},
            "weights": WEIGHTS, "kinematic_source": kinematic_source}


class AlertDebouncer:
    def __init__(self, persistence_seconds=3.0, cooldown_seconds=30.0):
        self.persistence = persistence_seconds
        self.cooldown = cooldown_seconds
        self.state = {}

    def update(self, zone_id, timestamp, risk):
        tier = risk["tier"]
        previous = self.state.get(zone_id)
        if previous is None or previous["tier"] != tier:
            previous = {"tier": tier, "since": timestamp,
                        "last_alert": previous["last_alert"] if previous else -math.inf,
                        "last_tier": previous["last_tier"] if previous else "Safe"}
            self.state[zone_id] = previous
        severe = tier in {"Moderate", "High", "Critical"}
        rank = {"Safe": 0, "Moderate": 1, "High": 2, "Critical": 3}
        escalation = rank.get(tier, 0) > rank.get(previous["last_tier"], 0)
        if severe and timestamp - previous["since"] >= self.persistence and (timestamp - previous["last_alert"] >= self.cooldown or escalation):
            previous["last_alert"], previous["last_tier"] = timestamp, tier
            return True
        return False
