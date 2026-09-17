"""Experimental, configurable occupancy/motion score; NOT a probability of harm."""

import math

TIERS = [(80, "Critical"), (60, "High"), (35, "Moderate"), (0, "Safe")]


def assess(count, capacity, motion, forecast):
    if not capacity or capacity <= 0:
        return {
            "score": None,
            "tier": "Uncalibrated",
            "reasons": [
                "Set a zone reference capacity to enable the experimental score"
            ],
            "action": "Review camera coverage and configure the zone.",
            "method": "heuristic_v1",
        }
    occupancy = max(0.0, count / capacity)
    density_score = min(1.0, occupancy)
    reversal = motion.get("reversal_fraction") or 0.0
    sudden = motion.get("sudden_motion_fraction") or 0.0
    stalled = motion.get("stalled_fraction") or 0.0
    motion_score = min(
        1.0, 0.6 * reversal + 0.3 * sudden + 0.1 * stalled * min(occupancy, 1)
    )
    future = forecast.get("count")
    growth = max(0.0, (future - count) / capacity) if future is not None else 0.0
    score = round(
        min(100.0, 70 * density_score + 20 * motion_score + 10 * min(1.0, growth)), 1
    )
    tier = next(label for threshold, label in TIERS if score >= threshold)
    reasons = [
        f"Observed count is {round(100 * occupancy)}% of the configured reference capacity"
    ]
    if reversal > 0.2:
        reasons.append("Direction reversals observed in tracked trajectories")
    if sudden > 0.2:
        reasons.append("Fast image-space movement observed")
    if stalled > 0.5 and occupancy > 0.6:
        reasons.append("Low movement with elevated occupancy")
    if growth > 0.1:
        reasons.append("Count forecast is rising")
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
        "method": "heuristic_v1",
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
