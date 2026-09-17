"""One-time local setup. Run with Python 3.12; requires Node.js 22+ and internet."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=ROOT):
    print("\n> " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", action="store_true", help="Install optional MediaPipe body landmarks"
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Do not download the university sample video",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Use Python 3.12 for the tested dependency set. Windows: py -3.12 scripts/bootstrap.py"
        )
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not npm:
        raise SystemExit(
            "Install Node.js 22+ (including npm), reopen the terminal, and run setup again."
        )
    node = shutil.which("node")
    major = int(
        subprocess.check_output(
            [node, "-p", "process.versions.node.split('.')[0]"], text=True
        ).strip()
    )
    if major < 22:
        raise SystemExit("Use Node.js 22 or newer for this dashboard.")
    environment = ROOT / ".venv"
    python = environment / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    # CPU wheels avoid installing an unnecessary CUDA runtime on Windows/Linux.
    if sys.platform in {"win32", "linux"}:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "torch==2.8.0",
                "torchvision==0.23.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
    run([python, "-m", "pip", "install", "-r", "requirements.txt"])
    if args.pose:
        run([python, "-m", "pip", "install", "-r", "requirements-pose.txt"])
    run([python, "scripts/setup_models.py", *(["--pose"] if args.pose else [])])
    if not args.skip_demo:
        run([python, "scripts/download_data.py", "--only", "umn_demo"])
    run([npm, "ci"], ROOT / "frontend")
    run([npm, "run", "build"], ROOT / "frontend")
    run([python, "-m", "pytest", "-q"])
    print(
        "\nSetup complete. Run start.bat on Windows, or .venv/bin/python scripts/run.py on Linux/macOS."
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Setup stopped at a failed command (exit {exc.returncode}). Resolve the error above and rerun setup."
        )
