# Model artifacts

Included custom checkpoints:

| File | Training | SHA-256 |
|---|---|---|
| density.pt | Frozen ImageNet MobileNetV2 features + density head, ShanghaiTech Part B | 7f387b2e2d0da0c7811b5c261bd97c94d86e6e615852da610f348d4267844df5 |
| forecast.pt | Count LSTM on UMN demonstration YOLO pseudo-labels | 0caa19f9e2899b767fd9a3cfdafb934d9e5f45d3377f8631bc20b747d920936b |

Downloaded separately with `python scripts/setup_models.py`:

- `yolov8n.pt`: official pretrained detector; SHA-256 observed in this run
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- `pose_landmarker_lite.task`: optional Google task asset (`--pose`); SHA-256
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.

Optional `python -m training.train_detector` output:

- `detector.pt`: best checkpoint from an explicitly supplied Ultralytics-format
  person dataset. `detector.metadata.json` records its inputs and checksum.

Custom checkpoints contain state dictionaries and metadata, loaded on CPU with
PyTorch `weights_only=True`. Do not replace any model with an untrusted checkpoint.
See `reports/` for measured results and `docs/THIRD_PARTY.md` for source attribution.
