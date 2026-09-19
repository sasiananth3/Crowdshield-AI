import json
from types import SimpleNamespace

from ml.alert_verifier import OpenClawAlertVerifier, infer_alert_type


def evidence(**overrides):
    value = {
        "count": 9,
        "capacity": 10,
        "forecast": {"count": 11, "horizon_seconds": 3},
        "motion": {"reversal_fraction": 0.0, "sudden_motion_fraction": 0.0},
        "risk": {
            "tier": "High",
            "reasons": ["Observed count is 90% of the configured reference capacity"],
        },
        "frame_timestamps": [1.0, 1.5, 2.0],
    }
    value.update(overrides)
    return value


def completed_with(decision, model="openrouter/free"):
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "ok": True,
                "provider": "openrouter",
                "model": model,
                "outputs": [{"text": json.dumps(decision)}],
            }
        ),
        stderr="",
    )


def test_rule_based_type_is_available_without_agent():
    assert infer_alert_type(evidence()) == "Crowding / capacity"
    assert (
        infer_alert_type(
            evidence(
                capacity=None,
                motion={"count_drop_fraction": 0.7},
                risk={"tier": "Moderate", "reasons": ["Possible dispersal"]},
            )
        )
        == "Rapid crowd dispersal"
    )


def test_openclaw_confirmation_is_normalized(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return completed_with(
            {
                "decision": "confirm",
                "alert_type": "Crowding / capacity",
                "confidence": 0.92,
                "summary": "Visible crowding supports the capacity warning.",
            }
        )

    verifier = OpenClawAlertVerifier(command="python", runner=runner)
    result = verifier.verify(evidence(), [frame])

    assert result["status"] == "confirmed"
    assert result["display_alert"] is True
    assert result["provider"] == "openrouter"
    assert "openrouter/openrouter/free" in calls[0][0]
    assert calls[0][1]["timeout"] == 45


def test_only_high_confidence_rejection_suppresses_alert(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")

    def response(confidence):
        return completed_with(
            {
                "decision": "reject",
                "alert_type": "Crowding / capacity",
                "confidence": confidence,
                "summary": "The marked zone appears substantially less occupied.",
            }
        )

    high = OpenClawAlertVerifier(
        command="python", runner=lambda *args, **kwargs: response(0.91)
    ).verify(evidence(), [frame])
    low = OpenClawAlertVerifier(
        command="python", runner=lambda *args, **kwargs: response(0.7)
    ).verify(evidence(), [frame])

    assert high["status"] == "rejected"
    assert high["display_alert"] is False
    assert low["status"] == "inconclusive"
    assert low["display_alert"] is True


def test_bad_openclaw_output_fails_open(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    verifier = OpenClawAlertVerifier(
        command="python",
        runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "outputs": [{"text": "not json"}]}),
            stderr="",
        ),
    )

    result = verifier.verify(evidence(), [frame])

    assert result["status"] == "unavailable"
    assert result["display_alert"] is True
    assert result["alert_type"] == "Crowding / capacity"
