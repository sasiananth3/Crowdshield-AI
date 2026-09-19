"""MobileNetV2 density estimation with crop-trained/tiled inference support."""

from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class DensityNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights).features[:14]
        self.head = nn.Sequential(nn.Conv2d(96, 64, 3, padding=2, dilation=2), nn.ReLU(),
                                  nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.ReLU(),
                                  nn.Conv2d(32, 1, 1), nn.Softplus())

    def forward(self, x):
        return self.head(self.backbone(x))


def image_tensor(rgb, size=256):
    from PIL import Image
    array = np.asarray(Image.fromarray(rgb).resize((size, size)), dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.tensor(array.transpose(2, 0, 1), dtype=torch.float32)


def _pad_tile(rgb, tile_size):
    h, w = rgb.shape[:2]
    if h == tile_size and w == tile_size: return rgb, h, w
    return np.pad(rgb, ((0, tile_size-h), (0, tile_size-w), (0,0)), mode="edge"), h, w


def predict_with_model(model, bgr, tiled=True, tile_size=256):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1]); h, w = rgb.shape[:2]; model.eval()
    with torch.inference_mode():
        if not tiled:
            result = model(image_tensor(rgb).unsqueeze(0))[0,0].cpu().numpy()
            return {"count": float(result.sum()), "map": result}
        scale = 16
        stitched = np.zeros((max(1, math.ceil(h/scale)), max(1, math.ceil(w/scale))), dtype=np.float32)
        for top in range(0,h,tile_size):
            for left in range(0,w,tile_size):
                crop=rgb[top:min(top+tile_size,h), left:min(left+tile_size,w)]
                padded,crop_h,crop_w=_pad_tile(crop,tile_size)
                pred=model(image_tensor(padded).unsqueeze(0))[0,0].cpu().numpy()
                rows=max(1,min(scale,math.ceil(crop_h*scale/tile_size))); cols=max(1,min(scale,math.ceil(crop_w*scale/tile_size)))
                stitched[top//scale:top//scale+rows,left//scale:left//scale+cols]+=pred[:rows,:cols]
        return {"count": float(stitched.sum()), "map": stitched}


def density_count_in_polygon(density_map, polygon):
    import cv2
    h,w=density_map.shape[:2]
    points=(np.asarray(polygon,dtype=np.float32)*np.array([w,h])).astype(np.int32)
    mask=np.zeros((h,w),dtype=np.uint8); cv2.fillPoly(mask,[points],1)
    return float(density_map[mask.astype(bool)].sum())


class DensityEstimator:
    def __init__(self, checkpoint):
        self.model=None; self.metadata={}
        if Path(checkpoint).exists():
            state=torch.load(checkpoint,map_location="cpu",weights_only=True)
            self.model=DensityNet(); self.model.load_state_dict(state["state_dict"]); self.model.eval()
            self.metadata=state.get("metadata",{})

    def predict(self,bgr):
        if self.model is None: return None
        result=predict_with_model(self.model,bgr,tiled=self.metadata.get("inference_mode")=="tiled_256",tile_size=256)
        result.update({"status":"experimental_trained_density","metadata":self.metadata})
        return result
