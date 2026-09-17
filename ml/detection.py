from pathlib import Path


class PersonDetector:
    def __init__(self, weights, confidence=0.25):
        if not Path(weights).exists():
            raise FileNotFoundError(
                "YOLOv8 weights missing. Run: python scripts/setup_models.py"
            )
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.confidence = confidence

    def detect(self, frame, tracking=True):
        options = dict(
            classes=[0], conf=self.confidence, imgsz=640, verbose=False, device="cpu"
        )
        if tracking:
            result = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml", **options
            )[0]
        else:
            result = self.model.predict(frame, **options)[0]
        items = []
        if result.boxes is not None:
            boxes = result.boxes.xyxyn.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            ids = (
                result.boxes.id.cpu().tolist()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )
            for box, confidence, track_id in zip(boxes, confidences, ids):
                items.append(
                    {
                        "box": [round(float(v), 5) for v in box],
                        "confidence": round(float(confidence), 3),
                        "track_id": int(track_id) if track_id is not None else None,
                    }
                )
        return items
