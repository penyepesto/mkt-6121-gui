# -*- coding: utf-8 -*-
"""project.gui.ops.geometric_ops

Geometric‑level operations exposed in the **Geometric Methods** tab.
Each public function is *pure* (stateless) and registered via
`OperationSpec` so the ParamPanel can auto‑create Qt widgets.

SUB‑TABS & CATEGORIES
────────────────────────────────────────────────────────────────────────────
1. Feature Detection   → Harris/Shi‑Tomasi corner, Hough Line, Hough Circle
2. Basic Transforms    → Resize, Rotate, Flip
3. Advanced            → Shear, Scale, Reflection, Interpolation demo,
                          Stereo disparity (BM)

The module only depends on NumPy/OpenCV and the shared `OperationSpec` class
imported from *classical_ops* to avoid a separate spec.py.
"""
from __future__ import annotations

from typing import Dict, Any, Callable, List
import math

import cv2
import numpy as np

# Re‑use the same OperationSpec class defined in classical_ops to avoid cycles
from .classical_ops import OperationSpec

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ Helper utils ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #

def _as_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image is 8‑bit single/multi‑channel for Qt display."""
    if img.dtype == np.uint8:
        return img
    img2 = np.clip(img, 0, 255).astype(np.uint8)
    return img2

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑ 1. FEATURE DETECTION ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #

def _corner_detect(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = p["method"]
    bs = int(p["block"])
    k = p["k"]

    if method == "Harris":
        dst = cv2.cornerHarris(gray, bs, 3, k)
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
    else:  # Shi‑Tomasi
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=bs)
        if corners is not None:
            for c in corners.astype(int):
                cv2.circle(img, tuple(c[0]), 3, (0, 255, 0), 1)
    return img


def _hough_lines(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=p["thresh"], minLineLength=p["min_len"], maxLineGap=p["gap"])
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img


def _hough_circles(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, p["min_dist"], param1=100, param2=p["param2"], minRadius=p["min_r"], maxRadius=p["max_r"])
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    return img

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑ 2. BASIC TRANSFORMS ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #

def _resize(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    fx = p["scale"]
    interp = {
        "Nearest": cv2.INTER_NEAREST,
        "Bilinear": cv2.INTER_LINEAR,
        "Bicubic": cv2.INTER_CUBIC,
    }[p["interp"]]
    out = cv2.resize(img, None, fx=fx, fy=fx, interpolation=interp)
    return out


def _rotate(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    angle = p["angle"]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


def _flip(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[p["mode"]]
    return cv2.flip(img, code)

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑ 3. ADVANCED TRANSFORMS / FEATURES ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #

def _shear(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    shx = p["shx"]
    shy = p["shy"]
    h, w = img.shape[:2]
    M = np.float32([[1, shx, 0], [shy, 1, 0]])
    nW = int(w + abs(shy) * h)
    nH = int(h + abs(shx) * w)
    return cv2.warpAffine(img, M, (nW, nH))


def _scale(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    sx, sy = p["sx"], p["sy"]
    h, w = img.shape[:2]
    M = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(img, M, (int(w * sx), int(h * sy)))


def _reflect(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    axis = p["axis"]
    if axis == "x":
        M = np.float32([[1, 0, 0], [0, -1, img.shape[0]]])
    elif axis == "y":
        M = np.float32([[-1, 0, img.shape[1]], [0, 1, 0]])
    else:  # xy
        M = np.float32([[-1, 0, img.shape[1]], [0, -1, img.shape[0]]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def _interp_demo(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    method = p["method"]
    interp_map = {"Nearest": cv2.INTER_NEAREST, "Bilinear": cv2.INTER_LINEAR, "Bicubic": cv2.INTER_CUBIC}
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=interp_map[method])
    big = cv2.resize(small, img.shape[1::-1], interpolation=interp_map[method])
    return big


def _disparity(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    # naive demo: assume the left/right halves of the input make a stereo pair
    h, w = img.shape[:2]
    left = cv2.cvtColor(img[:, : w // 2], cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(img[:, w // 2 :], cv2.COLOR_BGR2GRAY)
    num = int(p["num"])
    block = int(p["block"])
    stereo = cv2.StereoBM_create(numDisparities=num, blockSize=block)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp = cv2.applyColorMap(_as_uint8(disp), cv2.COLORMAP_JET)
    return disp

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑ Operation registry ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ #

_OPS: Dict[str, OperationSpec] = {
    # 1. Feature Detection
    "corner": OperationSpec(
        id="corner",
        display="Corner Detect",
        category="Feature",
        controls={
            "method": dict(type="combo", items=["Harris", "Shi‑Tomasi"], default="Harris"),
            "block": dict(type="spin", min=2, max=10, step=1, default=2),
            "k": dict(type="dspin", min=0.01, max=0.2, step=0.01, default=0.04),
        },
        func=_corner_detect,
    ),
    "hline": OperationSpec(
        id="hline",
        display="Hough Line",
        category="Feature",
        controls={
            "thresh": dict(type="spin", min=10, max=200, step=5, default=100),
            "min_len": dict(type="spin", min=10, max=200, step=5, default=30),
            "gap": dict(type="spin", min=1, max=50, step=1, default=5),
        },
        func=_hough_lines,
    ),
    "hcircle": OperationSpec(
        id="hcircle",
        display="Hough Circle",
        category="Feature",
        controls={
            "min_dist": dict(type="spin", min=10, max=200, step=5, default=90),
            "param2": dict(type="spin", min=10, max=200, step=5, default=90),
            "min_r": dict(type="spin", min=0, max=100, step=1, default=0),
            "max_r": dict(type="spin", min=0, max=200, step=1, default=0),
        },
        func=_hough_circles,
    ),
    # 2. Basic Transforms
    "resize": OperationSpec(
        id="resize",
        display="Resize",
        category="Transform",
        controls={
            "scale": dict(type="dspin", min=0.1, max=4.0, step=0.1, default=1.0),
            "interp": dict(type="combo", items=["Nearest", "Bilinear", "Bicubic"], default="Bilinear"),
        },
        func=_resize,
    ),
    "rotate": OperationSpec(
        id="rotate",
        display="Rotate",
        category="Transform",
        controls={
            "angle": dict(type="dspin", min=-180.0, max=180.0, step=1.0, default=0.0),
        },
        func=_rotate,
    ),
    "flip": OperationSpec(
        id="flip",
        display="Flip",
        category="Transform",
        controls={
            "mode": dict(type="combo", items=["Horizontal", "Vertical", "Both"], default="Horizontal"),
        },
        func=_flip,
    ),
    # 3. Advanced
    "shear": OperationSpec(
        id="shear",
        display="Shear",
        category="Advanced",
        controls={
            "shx": dict(type="dspin", min=-1.0, max=1.0, step=0.05, default=0.2),
            "shy": dict(type="dspin", min=-1.0, max=1.0, step=0.05, default=0.0),
        },
        func=_shear,
    ),
    "scale": OperationSpec(
        id="scale",
        display="Scale (Affine)",
        category="Advanced",
        controls={
            "sx": dict(type="dspin", min=0.1, max=3.0, step=0.1, default=1.2),
            "sy": dict(type="dspin", min=0.1, max=3.0, step=0.1, default=1.2),
        },
        func=_scale,
    ),
    "reflect": OperationSpec(
        id="reflect",
        display="Reflect",
        category="Advanced",
        controls={
            "axis": dict(type="combo", items=["x", "y", "xy"], default="x"),
        },
        func=_reflect,
    ),
    "interp_demo": OperationSpec(
        id="interp_demo",
        display="Interpolation Demo",
        category="Advanced",
        controls={
            "method": dict(type="combo", items=["Nearest", "Bilinear", "Bicubic"], default="Bilinear"),
        },
        func=_interp_demo,
    ),
    "disparity": OperationSpec(
        id="disparity",
        display="Stereo Disparity",
        category="Advanced",
        controls={
            "num": dict(type="spin", min=16, max=256, step=16, default=96),
            "block": dict(type="spin", min=5, max=51, step=2, default=15),
        },
        func=_disparity,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


ALL_OPS: Dict[str, OperationSpec] = _OPS

def get_operations() -> Dict[str, "OperationSpec"]:
    """Return a *copy* of the operation registry so callers can safely mutate."""
    return ALL_OPS.copy()
