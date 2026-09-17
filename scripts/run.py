"""Start the local dashboard and API using the active Python environment."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    if not (ROOT / "frontend/dist/index.html").exists():
        raise SystemExit("Dashboard is not built. Run scripts/bootstrap.py first.")
    if not (ROOT / "models/yolov8n.pt").exists():
        raise SystemExit(
            "Detector weights missing. Run python scripts/setup_models.py first."
        )
    import uvicorn

    print(
        "\nOpen http://127.0.0.1:8000 in your browser. Keep this terminal open. Ctrl+C stops the server.\n",
        flush=True,
    )
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000)
