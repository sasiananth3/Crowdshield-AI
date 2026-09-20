import json
import os
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


def test_project_local_smoke_test_marker_controls_configured_state(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    verifier = OpenClawAlertVerifier(
        command="python",
        runtime_root=tmp_path,
        project_root=tmp_path,
    )
    assert verifier.available is True
    assert verifier.configured is False

    verifier.verification_marker.write_text("{}", encoding="utf-8")

    assert verifier.configured is True


def test_project_local_node_is_added_to_openclaw_path(tmp_path):
    node_dir = tmp_path / "tools" / "node-v26.8.2-win-x64"
    node_dir.mkdir(parents=True)
    (node_dir / "node.exe").write_bytes(b"node")
    verifier = OpenClawAlertVerifier(
        command="python",
        runtime_root=tmp_path / "runtime",
        project_root=tmp_path,
        auth_verified=True,
    )

    environment = verifier._scoped_environment()
    path_value = next(value for key, value in environment.items() if key.upper() == "PATH")

    assert path_value.split(os.pathsep)[0] == str(node_dir.resolve())


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

    verifier = OpenClawAlertVerifier(
        command="python",
        runner=runner,
        runtime_root=tmp_path,
        project_root=tmp_path,
        auth_verified=True,
    )
    result = verifier.verify(evidence(), [frame])

    assert result["status"] == "confirmed"
    assert result["display_alert"] is True
    assert result["provider"] == "openrouter"
    assert "openrouter/free" in calls[0][0]
    assert calls[0][1]["timeout"] == 45
    assert calls[0][1]["cwd"] == tmp_path / "openclaw" / "workspace"
    assert calls[0][1]["env"]["OPENCLAW_STATE_DIR"] == str(
        tmp_path / "openclaw" / "state"
    )
    assert calls[0][1]["env"]["TEMP"] == str(tmp_path / "openclaw" / "temp")


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
        command="python",
        runner=lambda *args, **kwargs: response(0.91),
        runtime_root=tmp_path,
        project_root=tmp_path,
        auth_verified=True,
    ).verify(evidence(), [frame])
    low = OpenClawAlertVerifier(
        command="python",
        runner=lambda *args, **kwargs: response(0.7),
        runtime_root=tmp_path,
        project_root=tmp_path,
        auth_verified=True,
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
        runtime_root=tmp_path,
        project_root=tmp_path,
        auth_verified=True,
    )

    result = verifier.verify(evidence(), [frame])

    assert result["status"] == "unavailable"
    assert result["display_alert"] is True
    assert result["alert_type"] == "Crowding / capacity"


def test_openclaw_cannot_receive_frames_outside_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"jpeg")
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True

    verifier = OpenClawAlertVerifier(
        command="python",
        runner=runner,
        runtime_root=runtime,
        project_root=tmp_path,
        auth_verified=True,
    )
    result = verifier.verify(evidence(), [outside])

    assert result["status"] == "unavailable"
    assert result["display_alert"] is True
    assert called is False
