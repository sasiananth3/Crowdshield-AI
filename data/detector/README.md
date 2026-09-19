# Detector dataset staging

Place person-detection images under `images/` and matching YOLO bounding-box
labels under `labels/`. Nested paths must be mirrored. For example:

```text
images/camera-a/frame001.jpg
labels/camera-a/frame001.txt
```

Every non-empty label line must contain:

```text
0 x_center y_center width height
```

Coordinates and sizes are normalized to `[0, 1]`; width and height must be
positive. Class `0` means person. Empty label files are accepted as negative
examples, but every image must have a matching label file.

Prepare and validate an 80/20 deterministic split:

```bash
python -m training.train_detector --prepare-only
```

This creates `train.txt`, `val.txt`, and `../detector.yaml`. The command does
not copy or move source images. Run `python -m training.train_detector` to train.
