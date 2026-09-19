from argparse import Namespace
import csv
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from training.train_detector import prepare_detector_dataset
from training.train_detector import run as train_detector
from training.train_forecast import read_rows, windows


def test_windows_respect_sequence_and_sample_rate():
    rows = [
        {"sequence": "a", "timestamp": str(i / 2), "count": str(i)}
        for i in range(8)
    ]
    x, y = windows(rows, window=3, horizon=2, step=0.5)
    assert x.shape == (4, 3)
    assert y.tolist() == [4, 5, 6, 7]
    assert np.array_equal(x[0], [0, 1, 2])


def test_windows_do_not_cross_sequence_boundaries():
    rows = [
        {"sequence": sequence, "timestamp": str(i), "count": str(i)}
        for sequence in ["a", "b"]
        for i in range(5)
    ]
    x, _ = windows(rows, window=2, horizon=1)
    assert len(x) == 6


def test_read_rows_validates_schema_and_order(tmp_path):
    path = tmp_path / "counts.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sequence", "timestamp", "count"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sequence": "a", "timestamp": 1, "count": 2},
                {"sequence": "a", "timestamp": 0, "count": 3},
            ]
        )
    with pytest.raises(ValueError, match="Timestamps must increase"):
        read_rows(path)


def test_detector_training_requires_a_real_dataset(tmp_path):
    args = Namespace(
        data=tmp_path / "missing.yaml",
        source=tmp_path / "detector",
        weights=tmp_path / "missing.pt",
        epochs=1,
        imgsz=640,
        seed=42,
    )
    with pytest.raises(FileNotFoundError, match="staging folders"):
        train_detector(args)


def test_prepare_detector_dataset_validates_and_splits_pairs(tmp_path):
    source = tmp_path / "detector"
    images = source / "images"
    labels = source / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(10):
        Image.new("RGB", (32, 32), color=(index, 0, 0)).save(
            images / f"frame-{index}.jpg"
        )
        (labels / f"frame-{index}.txt").write_text(
            "0 0.5 0.5 0.25 0.5\n", encoding="utf-8"
        )

    yaml_path = tmp_path / "detector.yaml"
    result = prepare_detector_dataset(source, yaml_path, 0.2, seed=7)
    train = set((source / "train.txt").read_text(encoding="utf-8").splitlines())
    validation = set(
        (source / "val.txt").read_text(encoding="utf-8").splitlines()
    )
    assert result["train_images"] == 8
    assert result["validation_images"] == 2
    assert len(train | validation) == 10
    assert not train & validation
    assert "0: person" in yaml_path.read_text(encoding="utf-8")


def test_prepare_detector_dataset_rejects_invalid_boxes(tmp_path):
    source = tmp_path / "detector"
    images = source / "images"
    labels = source / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ["one", "two"]:
        Image.new("RGB", (16, 16)).save(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_text(
            "0 0.9 0.5 0.4 0.4\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="beyond image width"):
        prepare_detector_dataset(source, tmp_path / "detector.yaml")


def test_detector_training_publishes_trainer_best(tmp_path, monkeypatch):
    data = tmp_path / "data.yaml"
    weights = tmp_path / "start.pt"
    best = tmp_path / "run" / "weights" / "best.pt"
    data.write_text("names: {0: person}", encoding="utf-8")
    weights.write_bytes(b"start")
    best.parent.mkdir(parents=True)
    best.write_bytes(b"trained")

    class FakeYOLO:
        def __init__(self, _weights):
            self.trainer = None

        def train(self, **_kwargs):
            self.trainer = SimpleNamespace(save_dir=best.parents[1], best=best)
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.75})

    module = ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    output = tmp_path / "published.pt"
    args = Namespace(
        data=data,
        weights=weights,
        output=output,
        epochs=1,
        imgsz=640,
        batch=2,
        workers=0,
        device="cpu",
        project=tmp_path / "runs",
        name="test",
        exist_ok=False,
        seed=42,
        person_class=0,
        patience=1,
    )
    metadata = train_detector(args)
    assert output.read_bytes() == b"trained"
    assert metadata["validation_metrics"]["metrics/mAP50(B)"] == 0.75
    assert Path(output.with_suffix(".metadata.json")).is_file()
