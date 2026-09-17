# Development handoff — 16 September 2026

## Delivered in this archive

Source code for the model adapters, training, FastAPI backend, SQLite persistence,
React dashboard and setup scripts; the trained density and LSTM checkpoints;
dataset download tools and provenance; measured evaluation reports; unit/API and
opt-in real-model integration tests; and methodology/run documentation.

## Verified

- 16 unit, API and real-model integration tests passed on Linux/Python 3.12.
- The React dashboard production build succeeded.
- A 15-second real UMN video clip produced 30 inference samples, real detections,
  density estimates, LSTM forecasts, an alert, image/heatmap responses, and CSV.
- Optional pose gracefully reported its missing Linux graphics dependency without
  interrupting the remaining analysis.

## Not yet verified or completed

- Browser visual/end-to-end verification was attempted but did not complete:
  the standard browser download failed and the fallback browser crashed during
  font rendering. `frontend/scripts/verify.mjs` is included for a normal local
  Playwright installation; do not describe browser verification as passed.
- Windows/macOS setup has not been executed on those operating systems.
- Pose inference did not run successfully in this development environment.
- No improvement over the original reference papers is established. The LSTM's
  held-out MAE was worse than persistence. See the actual evaluation reports.
- No webcam/RTSP support, trained incident/action classifier or safety validation.

## GitHub publication

The first publication attempt was rejected by GitHub with HTTP 403. After the
owner updated access, a real file creation succeeded and publication resumed.
This document records the initial prototype's validation status; later commits
and evaluation reports record subsequent implementation work.

You can run this archive without GitHub access. Extract it, install Python 3.12
and Node.js 22+, and follow the setup commands in README.md. Raw datasets, videos,
third-party downloaded detector/pose weights, dependency installations, runtime
records and build caches are deliberately not included. Setup downloads what is
needed for the sample demonstration.
