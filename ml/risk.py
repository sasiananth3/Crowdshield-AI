"""Experimental, configurable occupancy/motion score; NOT a probability of harm."""

import math

TIERS = [(80, "Critical"), (60, "High"), (35, "Moderate"), (0, "Safe")]

# These image-space thresholds are exploratory and are intentionally kept here so
# that deployments can audit them. Persistence is applied separately by
# AlertDebouncer; a single frame that crosses a threshold does not create an alert.
MIN_MOVING_TRACKS = 3
FAST_CROWD_BODY_SPEED = 0.24
DISORDERED_CROWD_BODY_SPEED = 0.18
SPARSE_DISPERSAL_BODY_SPEED = 0.20
MIN_DISPERSAL_BASELINE = 5
RAPID_COUNT_DROP_FRACTION = 0.5
DISPERSAL_PERSISTENCE_SECONDS = 1.0
MOTION_PERSISTENCE_SECONDS = 1.5
ALERT_RECOVERY_SECONDS = 2.0


def _motion_warning(motion):
    """Return whether several tracked people show unusually strong movement."""
    tracked = motion.get("tracked_people") or 0
    body_speed = motion.get("mean_speed_body_lengths_per_second") or 0.0
    reversal = motion.get("reversal_fraction") or 0.0
    if tracked < MIN_MOVING_TRACKS:
        return False
    return (
        body_speed >= FAST_CROWD_BODY_SPEED
        or (body_speed >= DISORDERED_CROWD_BODY_SPEED and reversal >= 0.5)
    )


def _rapid_dispersal_warning(motion):
    """Detect a sharp count loss after recent multi-person movement.

    This complements track-local velocity: rapid movement commonly causes a
    tracker to lose identities just when the transition is most important.
    """
    recent_peak_count = motion.get("recent_peak_count") or 0
    drop = motion.get("count_drop_fraction") or 0.0
    recent_body_speed = (
        motion.get("recent_peak_mean_speed_body_lengths_per_second") or 0.0
    )
    return (
        recent_peak_count >= MIN_DISPERSAL_BASELINE
        and drop >= RAPID_COUNT_DROP_FRACTION
        and recent_body_speed >= SPARSE_DISPERSAL_BODY_SPEED
    )


def assess(count, capacity, motion, forecast):
    motion_warning = _motion_warning(motion)
    dispersal_warning = _rapid_dispersal_warning(motion)
    if not capacity or capacity <= 0:
        if motion_warning or dispersal_warning:
            reason = (
                "Possible rapid crowd dispersal observed after elevated movement"
                if dispersal_warning
                else "Elevated image-space movement observed across multiple tracked people"
            )
            return {
                "score": None,
                "tier": "Moderate",
                "reasons": [reason],
                "action": (
                    "Review the video and monitor whether crowd movement is continuing."
                ),
                "method": "crowd_dynamics_v1",
                "alert_persistence_seconds": (
                    DISPERSAL_PERSISTENCE_SECONDS
                    if dispersal_warning
                    else MOTION_PERSISTENCE_SECONDS
                ),
                "is_probability": False,
                "validated_for_safety": False,
            }
        return {
            "score": None,
            "tier": "Uncalibrated",
            "reasons": [
                "Set a zone reference capacity to enable the experimental score"
            ],
            "action": "Review camera coverage and configure the zone.",
            "method": "heuristic_v2",
            "is_probability": False,
            "validated_for_safety": False,
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
    # A persistent multi-person movement warning must remain alertable even when a
    # generously configured capacity keeps the occupancy contribution low.
    if motion_warning or dispersal_warning:
        score = max(35.0, score)
    tier = next(label for threshold, label in TIERS if score >= threshold)
    reasons = [
        f"Observed count is {round(100 * occupancy)}% of the configured reference capacity"
    ]
    if reversal > 0.2:
        reasons.append("Direction reversals observed in tracked trajectories")
    if sudden > 0.2:
        reasons.append("Fast image-space movement observed")
    if motion_warning and sudden <= 0.2:
        reasons.append(
            "Elevated image-space movement observed across multiple tracked people"
        )
    if dispersal_warning:
        reasons.append(
            "Possible rapid crowd dispersal observed after elevated movement"
        )
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
        "method": "heuristic_v2",
        "alert_persistence_seconds": (
            DISPERSAL_PERSISTENCE_SECONDS
            if dispersal_warning
            else MOTION_PERSISTENCE_SECONDS
        ),
        "is_probability": False,
        "validated_for_safety": False,
    }


class AlertDebouncer:
    """Turn noisy per-sample risk into incident-level alerts.

    A short non-alertable gap is treated as detector/tracker noise rather than the
    end of an incident. After a sustained clear interval, the zone is re-armed so
    a genuinely new incident can alert even if it begins inside the repeat
    cooldown for the previous incident.
    """

    def __init__(
        self,
        persistence_seconds=MOTION_PERSISTENCE_SECONDS,
        cooldown_seconds=30.0,
        recovery_seconds=ALERT_RECOVERY_SECONDS,
    ):
        self.persistence = persistence_seconds
        self.cooldown = cooldown_seconds
        self.recovery = recovery_seconds
        self.state = {}

    def update(self, zone_id, timestamp, risk):
        tier = risk["tier"]
        previous = self.state.get(zone_id)
        if previous is None:
            previous = {
                "tier": tier,
                "active_since": None,
                "tier_since": timestamp,
                "clear_since": timestamp,
                "required_persistence": self.persistence,
                "last_alert": -math.inf,
                "last_tier": "Safe",
            }
            self.state[zone_id] = previous
        severe = tier in {"Moderate", "High", "Critical"}
        rank = {"Safe": 0, "Moderate": 1, "High": 2, "Critical": 3}

        if not severe:
            if previous["clear_since"] is None:
                previous["clear_since"] = timestamp
            if timestamp - previous["clear_since"] >= self.recovery:
                previous.update(
                    active_since=None,
                    required_persistence=self.persistence,
                    last_tier="Safe",
                )
            previous["tier"] = tier
            return False

        required = risk.get("alert_persistence_seconds", self.persistence)
        if previous["active_since"] is None:
            previous["active_since"] = timestamp
            previous["required_persistence"] = required
        else:
            # A rapid-dispersal signal is intentionally allowed to shorten the
            # response time of an already developing motion incident.
            previous["required_persistence"] = min(
                previous["required_persistence"], required
            )
        if previous["tier"] != tier:
            previous["tier_since"] = timestamp
        previous.update(tier=tier, clear_since=None)

        active_ready = (
            timestamp - previous["active_since"]
            >= previous["required_persistence"]
        )
        cooldown_elapsed = timestamp - previous["last_alert"] >= self.cooldown
        rearmed = previous["last_tier"] == "Safe"
        escalation = rank.get(tier, 0) > rank.get(previous["last_tier"], 0)
        escalation_ready = (
            escalation and timestamp - previous["tier_since"] >= required
        )
        if (
            active_ready
            and (rearmed or cooldown_elapsed or escalation_ready)
        ):
            previous["last_alert"], previous["last_tier"] = timestamp, tier
            return True
        return False
