"""Configure OpenClaw/OpenRouter with state kept inside CrowdShield's runtime.

The API key is entered directly into OpenClaw's interactive login. This script
does not read, print, or store the key itself.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
STATE = RUNTIME / "openclaw" / "state"
WORKSPACE = RUNTIME / "openclaw" / "workspace"
VERIFIED = RUNTIME / "openclaw" / "verified.json"


def find_openclaw():
    configured = os.getenv("CROWDSHIELD_OPENCLAW_COMMAND")
    candidates = [
        Path(configured) if configured else None,
        ROOT / "node_modules" / ".bin" / "openclaw.cmd",
        ROOT / "node_modules" / ".bin" / "openclaw",
        ROOT / "tools" / "openclaw" / "node_modules" / ".bin" / "openclaw.cmd",
        ROOT / "tools" / "openclaw" / "node_modules" / ".bin" / "openclaw",
        ROOT / "tools" / "openclaw" / "openclaw.exe",
        ROOT / "tools" / "openclaw" / "openclaw.cmd",
    ]
    discovered = shutil.which(configured or "openclaw")
    if discovered:
        return discovered
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate.resolve())
    return None


def main():
    command = find_openclaw()
    if command is None:
        raise SystemExit(
            "OpenClaw was not found. Install a compatible OpenClaw release, then "
            "rerun: python scripts/configure_openclaw.py"
        )

    STATE.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    node_directories = sorted(
        (
            path
            for path in (ROOT / "tools").glob("node-v26*-win-x64")
            if (path / "node.exe").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if node_directories:
        environment["PATH"] = os.pathsep.join(
            [str(node_directories[0]), environment.get("PATH", "")]
        )
    environment.update(
        OPENCLAW_STATE_DIR=str(STATE),
        OPENCLAW_WORKSPACE_DIR=str(WORKSPACE),
    )
    # Never leave a previous success marker behind while replacing credentials.
    VERIFIED.unlink(missing_ok=True)

    print(f"OpenClaw command: {command}")
    print(f"Private state: {STATE}")
    print(f"Inference workspace: {WORKSPACE}")
    print("Enter the OpenRouter API key only when OpenClaw prompts for it.")
    model = os.getenv("CROWDSHIELD_OPENCLAW_MODEL", "openrouter/free")
    subprocess.run(
        [
            command,
            "models",
            "auth",
            "login",
            "--provider",
            "openrouter",
            "--method",
            "api-key",
        ],
        cwd=WORKSPACE,
        env=environment,
        check=True,
    )
    subprocess.run(
        [command, "models", "set", model],
        cwd=WORKSPACE,
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            command,
            "infer",
            "model",
            "run",
            "--local",
            "--model",
            model,
            "--prompt",
            "Reply with exactly OK",
            "--json",
        ],
        cwd=WORKSPACE,
        env=environment,
        check=True,
    )
    VERIFIED.write_text(
        json.dumps(
            {
                "provider": "openrouter",
                "model": model,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "OpenClaw authentication succeeded. Restart CrowdShield before enabling "
        "verification."
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"OpenClaw configuration failed (exit {exc.returncode}). "
            "No API key was printed by CrowdShield."
        )
    except KeyboardInterrupt:
        raise SystemExit("OpenClaw configuration cancelled.")
