# Model artifacts

Included custom checkpoints:

| File | Training | SHA-256 |
|---|---|---|
| density.pt | Frozen ImageNet MobileNetV2 features + density head, ShanghaiTech Part B | fc52a3b55e0450ccaf8e0f20231173e4501b9449c889571b9dd7b6fcf83659ed |
| forecast.pt | Count LSTM on UMN demonstration YOLO pseudo-labels | 1e0d987e6226b7afe285807089b8a298eca42888b69781ff26dc616a46614fd7 |

Downloaded separately with `python scripts/setup_models.py`:

- `yolov8n.pt`: official pretrained detector; SHA-256 observed in this run
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- `pose_landmarker_lite.task`: optional Google task asset (`--pose`); SHA-256
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.

Custom checkpoints contain state dictionaries and metadata, loaded on CPU with
PyTorch `weights_only=True`. Do not replace any model with an untrusted checkpoint.
See `reports/` for measured results and `docs/THIRD_PARTY.md` for source attribution.
