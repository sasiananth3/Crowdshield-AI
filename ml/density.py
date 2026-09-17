"""MobileNetV2 features plus a trainable dilated density head.

Maps integrate to estimated people, NOT people/m². No checkpoint -> unavailable.
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def integrate_zone(density_map, polygon, subpixels=8):
    """Integrate image-space count mass, with fractional cell coverage."""
    h, w = density_map.shape
    yy, xx = np.mgrid[:h * subpixels, :w * subpixels]
    x, y = (xx + 0.5) / (w * subpixels), (yy + 0.5) / (h * subpixels)
    inside = np.zeros(x.shape, dtype=bool)
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if a[1] != b[1]:
            crossing = (a[1] > y) != (b[1] > y)
            boundary = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            inside ^= crossing & (x < boundary)
    coverage = inside.reshape(h, subpixels, w, subpixels).mean(axis=(1, 3))
    return float((density_map * coverage).sum())


class DensityNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights).features[:14]
        self.head = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def image_tensor(rgb, size=256):
    from PIL import Image

    array = np.array(Image.fromarray(rgb).resize((size, size)), dtype=np.float32) / 255
    array = (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.tensor(array.transpose(2, 0, 1), dtype=torch.float32)


class DensityEstimator:
    def __init__(self, checkpoint):
        self.model = None
        self.metadata = {}
        if Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = DensityNet()
            self.model.load_state_dict(state["state_dict"])
            self.model.eval()
            self.metadata = state.get("metadata", {})

    def predict(self, bgr):
        if self.model is None:
            return None
        with torch.inference_mode():
            result = self.model(image_tensor(bgr[:, :, ::-1]).unsqueeze(0))[
                0, 0
            ].numpy()
        return {
            "count": float(result.sum()),
            "map": result,
            "status": "experimental_trained_density",
            "metadata": self.metadata,
        }
