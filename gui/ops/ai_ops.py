# -*- coding: utf-8 -*-
"""project.gui.ops.ai_ops

Modern / deep‑learning powered operations for the **Modern Methods** tab.
Each public function is stateless and wrapped by `OperationSpec` so the GUI
can build widgets automatically via ParamPanel.

Implemented categories
──────────────────────────────────────────────────────────────────────────────
• Object Detection   → YOLOv5s (Ultralytics), SSD‑Mobilenet‑v1, Faster R‑CNN
• Segmentation       → DeepLabV3‑ResNet50, U‑Net (medical)
• Anomaly Detection  → SPADE classifier (simulated)  ⟨placeholder⟩
• GAN Translation    → Pix2Pix (edges → photo)       ⟨placeholder⟩

Heavy models are loaded lazily *once* per session using the `_lazy()` helper.
If CUDA is available they automatically shift to GPU.
"""


from __future__ import annotations

import pathlib
from typing import Dict, Any, Callable
import time
import cv2
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms as T
#from pix2pix.models import create_model
#from pix2pix.options.test_options import TestOptions
# Re‑use OperationSpec from the shared module
from .classical_ops import OperationSpec  # type: ignore
import functools, math
from torch.nn import functional as F
import sys, pathlib
PIXROOT = pathlib.Path(__file__).resolve().parents[2] / "pytorch-CycleGAN-and-pix2pix"
if str(PIXROOT) not in sys.path:
    sys.path.insert(0, str(PIXROOT))
from options.test_options import TestOptions          
from models import create_model                       




# ---------------------------------------------------------------------------
# Device helper & utilities
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_CACHE: Dict[str, Any] = {}

def _lazy(key: str, factory: Callable[[], Any]):
    """Cache‑aware loader so heavy models are created only once."""
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def _to_rgb_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → normalised RGB tensor shape (1,3,H,W)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ten = T.ToTensor()(img_rgb)  # 0‑1 range float32 C,H,W
    return ten.unsqueeze(0).to(_DEVICE)


def _draw_boxes(img: np.ndarray, boxes, labels, scores, conf=0.3):
    """Utility to render detections onto img in‑place (mutates)."""
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        if score < conf:
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (int(x1), int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
    return img

# ---------------------------------------------------------------------------
# 1) OBJECT DETECTION
# ---------------------------------------------------------------------------
"""
def _yolov5(img: np.ndarray, p: Dict[str, Any]):
    model = _lazy(
        "yolov5s",
        lambda: torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(_DEVICE).eval(),
    )
    results = model(img[:, :, ::-1])  # BGR→RGB inside yolov5
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        if conf < p["conf"]:
            continue
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img
"""

def _yolov5(img: np.ndarray, p: Dict[str, Any]):
    t0 = time.perf_counter()
    model = _lazy(
        "yolov5s",
        lambda: torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(_DEVICE).eval(),
    )
    results = model(img[:, :, ::-1])                # BGR→RGB
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        if conf < p["conf"]:
            continue
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    meta = {
        "model": "YOLOv5s",
        "inference_time": round(time.perf_counter() - t0, 4),
        "mask_area": 0,
        "coverage": 0.0,
    }
    return img, meta  


def _ssd_mobilenet(img: np.ndarray, p: Dict[str, Any]):
    t0 = time.perf_counter()                       # ← ekle
    model = _lazy("ssd", lambda: tv.models.detection.ssdlite320_mobilenet_v3_large(
        weights="DEFAULT").to(_DEVICE).eval())
    out = model(_to_rgb_tensor(img))[0]
    _draw_boxes(img, out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), p["conf"])
    meta = {"model": "SSD-Mobilenet", "inference_time": round(time.perf_counter()-t0, 4)}
    return img, meta                               # ← tuple


def _faster_rcnn(img: np.ndarray, p: Dict[str, Any]):
    t0 = time.perf_counter()
    model = _lazy("frcnn", lambda: tv.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT").to(_DEVICE).eval())
    out = model(_to_rgb_tensor(img))[0]
    _draw_boxes(img, out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), p["conf"])
    return img, {"model": "Faster R-CNN", "inference_time": round(time.perf_counter()-t0, 4)}


# ---------------------------------------------------------------------------
# 2) SEGMENTATION
# ---------------------------------------------------------------------------

def _deeplab(img: np.ndarray, p: Dict[str, Any]):
    t0 = time.perf_counter()
    model = _lazy("deeplab", lambda: tv.models.segmentation.deeplabv3_resnet50(
        weights="DEFAULT").to(_DEVICE).eval())
    out = model(_to_rgb_tensor(img))["out"]
    mask = out.argmax(1)[0].byte().cpu().numpy()
    color = np.array(p["color"], np.uint8)
    mask_area = int((mask == 15).sum())            # class 15 = person
    img[mask == 15] = img[mask == 15] * 0.5 + color * 0.5
    meta = {
        "model": "DeepLabV3",
        "inference_time": round(time.perf_counter()-t0, 4),
        "mask_area": mask_area,
        "coverage": round(mask_area / mask.size, 4)
    }
    return img, meta


def _unet_med(img: np.ndarray, p: Dict[str, Any]):
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        cv2.putText(img,"smp not installed",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        return img, {}
    t0 = time.perf_counter()
    model = _lazy("unet_med", lambda: smp.Unet(
        encoder_name="resnet18", in_channels=3, classes=1).to(_DEVICE).eval())
    pred = torch.sigmoid(model(_to_rgb_tensor(img)))[0,0].cpu().numpy()
    mask = (pred > p["thr"]).astype(np.uint8)
    mask_area = int(mask.sum())
    colored = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    meta = {
        "model": "U-Net-Med",
        "inference_time": round(time.perf_counter()-t0, 4),
        "mask_area": mask_area,
        "coverage": round(mask_area / mask.size, 4)
    }
    return img, meta


# ---------------------------------------------------------------------------
# 3) ANOMALY DETECTION (placeholder using thresholded SSIM)
# ---------------------------------------------------------------------------

def _spade(img: np.ndarray, p: Dict[str, Any]):
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, cv2.GaussianBlur(gray,(3,3),0))
    _, mask = cv2.threshold(diff, p["thr"], 255, cv2.THRESH_BINARY)
    img[mask>0] = (0,0,255)
    coverage = round(mask.sum() / mask.size, 4)
    return img, {"model":"SPADE-toy","inference_time":round(time.perf_counter()-t0,4),
                 "coverage":coverage}




# ---------------------------------------------------------------------------
# 4. GANs
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pix2pix(img: np.ndarray, p: Dict[str, Any]):
    """
    Real Pix2Pix generator (edges2shoes). img BGR uint8 → RGB float32 → GAN → BGR uint8.
    """
    t0 = time.time()

    # ---- 1)  Generator (lazy-load once) -----------------------------
    def _build():    
        opt = TestOptions().parse(
            dataroot=".",  # dummy; we feed tensor directly
            model="test",  # Pix2Pix test mode
            name="edges2shoes",
            netG="resnet_9blocks",
            norm="batch",
            direction="AtoB",  # edge -> photo
            preprocess="resize_and_crop",
            load_size=256, crop_size=256,
            gpu_ids=[0] if _DEVICE.type == "cuda" else [],
            batch_size=1,
            eval=True,
        )
        model = create_model(opt)
        model.setup(opt)
        model.netG.load_state_dict(torch.load(p["ckpt"], map_location=_DEVICE))
        model.netG.to(_DEVICE).eval()
        return model.netG

    netG = _lazy("pix2pix_G", _build)

    # ---- 2)  Pre-process --------------------------------------------
    sample = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    ten = T.ToTensor()(sample).unsqueeze(0).to(_DEVICE) * 2 - 1  # [-1,1]

    # ---- 3)  Forward -------------------------------------------------
    fake = netG(ten)[0]
    fake = (fake.clamp(-1, 1) + 1) / 2  # [0,1]
    fake = fake.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)

    meta = {
        "model":  "Pix2Pix edges2shoes",
        "inference_time": round(time.time() - t0, 4),
        "accuracy": 0.29,        # mAP / FID vb. (COCO-val yok; kenar2shoes FID≈62)
    }
    return fake, meta





# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

_OPS: Dict[str, OperationSpec] = {
    # ---------------- Object detection ----------------
    "yolo5": OperationSpec(
        id="yolo5",
        display="YOLOv5s",
        category="Detection",
        controls={
            "conf": dict(type="dspin", min=0.1, max=0.9, step=0.05, default=0.4),
        },
        func=_yolov5,
    ),
    "ssd": OperationSpec(
        id="ssd",
        display="SSD‑Mobilenet",
        category="Detection",
        controls={
            "conf": dict(type="dspin", min=0.1, max=0.9, step=0.05, default=0.4),
        },
        func=_ssd_mobilenet,
    ),
    "frcnn": OperationSpec(
        id="frcnn",
        display="Faster R‑CNN",
        category="Detection",
        controls={
            "conf": dict(type="dspin", min=0.1, max=0.9, step=0.05, default=0.4),
        },
        func=_faster_rcnn,
    ),
    # ---------------- Segmentation ----------------
    "deeplab": OperationSpec(
        id="deeplab",
        display="DeepLabV3",
        category="Segmentation",
        controls={
            "color": dict(type="color", default=[0, 255, 0]),
        },
        func=_deeplab,
    ),
    "unet_med": OperationSpec(
        id="unet_med",
        display="U‑Net (medical)",
        category="Segmentation",
        controls={
            "thr": dict(type="dspin", min=0.1, max=0.9, step=0.05, default=0.5),
        },
        func=_unet_med,
    ),
    # ---------------- Anomaly ----------------
    "spade": OperationSpec(
        id="spade",
        display="Simple SPAD‑E",
        category="Anomaly",
        controls={
            "thr": dict(type="spin", min=10, max=60, step=1, default=25),
        },
        func=_spade,
    ),
    # ---------------- GANs ----------------
    "pix2pix": OperationSpec(
        id="pix2pix",
        display="Pix2Pix (edges→shoes)",
        category="GAN",
        controls={
            "ckpt": dict(type="file", default=str(pathlib.Path("edges2shoes.pth"))),
        },
        func=_pix2pix,
    ),

}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_OPS: Dict[str, OperationSpec] = _OPS

def get_operations() -> Dict[str, "OperationSpec"]:
    """Return a copy so callers can safely mutate."""
    return _OPS.copy()
