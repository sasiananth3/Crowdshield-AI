"""Fine-tune the YOLO person detector on an Ultralytics-format dataset.

ShanghaiTech contains point annotations, not detection boxes, so it is not a
valid input for this command.  Supply a dataset YAML whose labels use class 0
for people (the convention used by the runtime adapter).
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _existing_file(value, description):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def validate_yolo_label(path):
    """Validate one bounding-box label file and return its object count."""
    objects = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_number}: expected 'class x_center y_center width height'"
            )
        try:
            class_id, x_center, y_center, width, height = map(float, parts)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: values must be numeric") from error
        if class_id != 0:
            raise ValueError(
                f"{path}:{line_number}: class must be 0 (person), got {parts[0]}"
            )
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            raise ValueError(f"{path}:{line_number}: box centre must be within [0, 1]")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError(f"{path}:{line_number}: box size must be within (0, 1]")
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            raise ValueError(f"{path}:{line_number}: box extends beyond image width")
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            raise ValueError(f"{path}:{line_number}: box extends beyond image height")
        objects += 1
    return objects


def prepare_detector_dataset(source, data_yaml, validation_ratio=0.2, seed=42):
    """Validate staged YOLO pairs and write deterministic split manifests/YAML."""
    source = Path(source).expanduser().resolve()
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    images_directory = source / "images"
    labels_directory = source / "labels"
    if not images_directory.is_dir() or not labels_directory.is_dir():
        raise FileNotFoundError(
            "Detector staging folders are missing. Put images in "
            f"{images_directory} and matching YOLO labels in {labels_directory}"
        )

    images = sorted(
        path
        for path in images_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < 2:
        raise ValueError("At least two labelled detector images are required")

    object_count = 0
    missing_labels = []
    for image_path in images:
        relative = image_path.relative_to(images_directory)
        label_path = (labels_directory / relative).with_suffix(".txt")
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Unreadable detector image: {image_path}") from error
        object_count += validate_yolo_label(label_path)
    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:3])
        extra = f" (+{len(missing_labels) - 3} more)" if len(missing_labels) > 3 else ""
        raise FileNotFoundError(f"Missing labels: {preview}{extra}")
    if object_count == 0:
        raise ValueError("The detector dataset contains no person bounding boxes")

    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1, max(1, round(len(shuffled) * validation_ratio))
    )
    validation = sorted(shuffled[:validation_count])
    train = sorted(shuffled[validation_count:])

    source.mkdir(parents=True, exist_ok=True)
    manifests = {"train": train, "val": validation}
    for split, paths in manifests.items():
        manifest = source / f"{split}.txt"
        temporary = manifest.with_suffix(manifest.suffix + ".part")
        lines = ["./" + path.relative_to(source).as_posix() for path in paths]
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(manifest)

    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"path: {json.dumps(source.as_posix())}\n"
        "train: train.txt\n"
        "val: val.txt\n\n"
        "names:\n"
        "  0: person\n"
    )
    yaml_temporary = data_yaml.with_suffix(data_yaml.suffix + ".part")
    yaml_temporary.write_text(yaml_text, encoding="utf-8")
    yaml_temporary.replace(data_yaml)
    return {
        "dataset_yaml": display_path(data_yaml),
        "source": display_path(source),
        "images": len(images),
        "person_boxes": object_count,
        "train_images": len(train),
        "validation_images": len(validation),
        "validation_ratio": validation_ratio,
        "seed": seed,
    }


def run(args):
    """Train a detector, publish its best checkpoint, and return its metadata."""
    data_path = Path(args.data).expanduser().resolve()
    preparation = None
    if getattr(args, "prepare_only", False) or not data_path.is_file():
        preparation = prepare_detector_dataset(
            getattr(args, "source", ROOT / "data/detector"),
            data_path,
            getattr(args, "validation_ratio", 0.2),
            getattr(args, "seed", 42),
        )
        print(json.dumps({"dataset_preparation": preparation}, indent=2))
        if getattr(args, "prepare_only", False):
            return {"dataset_preparation": preparation}
    data = _existing_file(data_path, "Detector dataset YAML")
    weights = _existing_file(args.weights, "Starting detector checkpoint")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch must be -1 (automatic) or greater than zero")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.person_class != 0:
        raise ValueError(
            "--person-class must be 0 because the CrowdShield runtime selects class 0"
        )

    # Keep this import inside run() so --help and unit tests do not initialize
    # Ultralytics or require its optional runtime dependencies.
    from ultralytics import YOLO

    started = time.time()
    model = YOLO(str(weights))
    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(Path(args.project).expanduser().resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        deterministic=True,
        classes=[args.person_class],
        patience=args.patience,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics training completed without trainer state")
    save_dir = Path(trainer.save_dir)
    best = Path(trainer.best)
    if not best.is_file():
        raise RuntimeError(f"Training completed without a best checkpoint at {best}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Copy via a sibling temporary file so an interrupted copy does not corrupt
    # an existing runtime checkpoint.
    temporary = output.with_suffix(output.suffix + ".part")
    shutil.copy2(best, temporary)
    temporary.replace(output)

    with output.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    results_dict = getattr(result, "results_dict", {}) if result is not None else {}
    validation_metrics = {
        str(key): float(value)
        for key, value in results_dict.items()
        if isinstance(value, (int, float)) or hasattr(value, "item")
    }
    metadata = {
        "task": "person_detection",
        "dataset_yaml": display_path(data),
        "starting_weights": display_path(weights),
        "output": display_path(output),
        "sha256": digest,
        "epochs": args.epochs,
        "image_size": args.imgsz,
        "person_class": args.person_class,
        "seed": args.seed,
        "training_seconds": round(time.time() - started, 2),
        "ultralytics_run": display_path(save_dir.resolve()),
        "validation_metrics": validation_metrics,
    }
    if preparation is not None:
        metadata["dataset_preparation"] = preparation
    metadata_path = output.with_suffix(".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".part")
    metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_temporary.replace(metadata_path)
    print(json.dumps(metadata, indent=2))
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO for person detection on labelled bounding boxes."
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data/detector.yaml"),
        help="Ultralytics dataset YAML (default: data/detector.yaml)",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "data/detector"),
        help="staging root containing images/ and labels/",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="validation fraction used when preparing staged data",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate staged data and build manifests/YAML without training",
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "models/yolov8n.pt"),
        help="starting YOLO checkpoint",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "models/detector.pt"),
        help="where to publish the best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--person-class", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default="crowdshield-person")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
