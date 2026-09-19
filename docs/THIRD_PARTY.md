# Sources, attribution and third-party terms

The project does not relicense third-party datasets, photographs, footage, model
weights, fonts or dependencies. Review original terms before reuse/distribution.

- YOLOv8 / Ultralytics: [official repository](https://github.com/ultralytics/ultralytics),
  [licensing information](https://www.ultralytics.com/license). Official detector
  weights are downloaded by setup, not included here; no detector training claimed.
- ByteTrack: [original implementation](https://github.com/ifzhang/ByteTrack),
  used through the Ultralytics tracker integration.
- MobileNetV2 / torchvision: [official model documentation](https://docs.pytorch.org/vision/stable/models/mobilenetv2.html).
  The supplied density checkpoint incorporates pretrained torchvision features.
- MediaPipe: [official repository](https://github.com/google-ai-edge/mediapipe) and
  [Pose Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python).
  Optional pretrained task assets download from Google's model storage.
- ShanghaiTech: [authors' dataset repository](https://github.com/desenzhou/ShanghaiTechDataset).
  Cite Zhang et al., *Single-Image Crowd Counting via Multi-Column Convolutional
  Neural Network*, CVPR 2016, when reporting this dataset's results.
- UMN demonstration: [University of Minnesota project page](https://mha.cs.umn.edu/proj_events.shtml).
  The downloaded public AVI is used for academic demonstration and pseudo-count
  experiments; public download does not imply unrestricted footage redistribution.
- React, Vite, FastAPI, PyTorch, OpenCV and lucide-react: retain their upstream
  notices and observe their licenses. No external font/image service is required
  by the dashboard during normal use.
- OpenClaw and OpenRouter are optional external services for candidate-alert
  verification. They are not bundled by this repository. Review their current
  software, model-provider, privacy, retention and rate-limit terms before sending
  footage. The OpenRouter free router may select different upstream models.

Model and dataset checksums document the exact development artifacts; they do
not certify that a dataset is appropriate or that a model is safe for deployment.
