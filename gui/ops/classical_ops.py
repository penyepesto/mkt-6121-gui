# -*- coding: utf-8 -*-
"""project.gui.ops.classical_ops

Classical (non‑deep‑learning) image‑processing operations that populate the
**Classical Methods** tab of the GUI.  Every operation is exposed through an
`OperationSpec` instance so that ParamPanel can auto‑generate the correct
widgets and MainWindow can build its processing chain dynamically.

This module is **self‑contained**; it has **no side‑effects** on import and
contains only pure functions.  Heavy objects (none here) would be instantiated
lazily inside their respective wrappers to avoid GUI freezes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Optional import of OperationSpec.  In the reference template it lives in
# project/gui/ops/spec.py, but we fall back to a NamedTuple when running the
# file stand‑alone (e.g. during unit tests).
# -----------------------------------------------------------------------------
try:
    from .spec import OperationSpec  # type: ignore
except (ImportError, ModuleNotFoundError):
    from collections import namedtuple

    OperationSpec = namedtuple(  # type: ignore
        "OperationSpec", ["id", "display", "category", "controls", "func"]
    )

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _odd(k: int) -> int:
    """Ensure *k* is odd & ≥ 1, converting even numbers to the next odd."""
    k = max(int(k), 1)
    return k if k % 2 else k + 1


def _cv2_kernel(size: int) -> np.ndarray:
    """Create a square 8‑bit structuring element of given *size*."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(size), _odd(size)))

# -----------------------------------------------------------------------------
# 1. Colour‑space conversion
# -----------------------------------------------------------------------------

_COLOR_MAP: Dict[str, int] = {
    "BGR→GRAY": cv2.COLOR_BGR2GRAY,
    "BGR→HSV": cv2.COLOR_BGR2HSV,
    "BGR→LAB": cv2.COLOR_BGR2LAB,
    "BGR→YCrCb": cv2.COLOR_BGR2YCrCb,
    "BGR→XYZ": cv2.COLOR_BGR2XYZ,
}


def _color_convert(img: np.ndarray, p: Dict) -> np.ndarray:
    code = _COLOR_MAP[p["mode"]]
    out = cv2.cvtColor(img, code)
    return out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

# -----------------------------------------------------------------------------
# 2. Filtering operations
# -----------------------------------------------------------------------------


def _gaussian_blur(img: np.ndarray, p: Dict) -> np.ndarray:
    k = _odd(p["kernel"])
    return cv2.GaussianBlur(img, (k, k), 0)


def _median_blur(img: np.ndarray, p: Dict) -> np.ndarray:
    k = _odd(p["kernel"])
    return cv2.medianBlur(img, k)


def _bilateral(img: np.ndarray, p: Dict) -> np.ndarray:
    d = p["diameter"]
    sigma = p["sigma"]
    return cv2.bilateralFilter(img, d, sigmaColor=sigma, sigmaSpace=sigma)


def _box_blur(img: np.ndarray, p: Dict) -> np.ndarray:
    k = _odd(p["kernel"])
    return cv2.blur(img, (k, k))


def _unsharp_mask(img: np.ndarray, p: Dict) -> np.ndarray:
    k = _odd(p["kernel"])
    amt = float(p["amount"])
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    return cv2.addWeighted(img, 1 + amt, blurred, -amt, 0)


def _emboss(img: np.ndarray, p: Dict) -> np.ndarray:
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    embossed = cv2.filter2D(gray, cv2.CV_16S, kernel)
    embossed = cv2.convertScaleAbs(embossed)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)


def _sharpen(img: np.ndarray, p: Dict) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

# -----------------------------------------------------------------------------
# 3. Edge‑detection
# -----------------------------------------------------------------------------


def _sobel(img: np.ndarray, p: Dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32)))
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)


def _laplacian(img: np.ndarray, p: Dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def _canny(img: np.ndarray, p: Dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, p["low"], p["high"])
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# -----------------------------------------------------------------------------
# 4. Morphological operations
# -----------------------------------------------------------------------------

_MORPH_MAP: Dict[str, int] = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def _morphology(img: np.ndarray, p: Dict) -> np.ndarray:
    op_code = _MORPH_MAP[p["op"]]
    k = _cv2_kernel(p["kernel"])
    iters = max(1, int(p["iters"]))
    out = cv2.morphologyEx(img, op_code, k, iterations=iters)
    return out

# -----------------------------------------------------------------------------
# Operation specification table – order controls GUI listing
# -----------------------------------------------------------------------------

ALL_OPS: Dict[str, OperationSpec] = {
    # Colour‑space ------------------------------------------------------------
    "cvt_color": OperationSpec(
        id="cvt_color",
        display="Color‑space Conversion",
        category="Color",
        controls={
            "mode": dict(type="combo", items=list(_COLOR_MAP.keys()), default="BGR→GRAY"),
        },
        func=_color_convert,
    ),
    # Filtering ---------------------------------------------------------------
    "gaussian": OperationSpec(
        id="gaussian",
        display="Gaussian Blur",
        category="Filtering",
        controls={
            "kernel": dict(type="spin", min=1, max=31, step=2, default=5),
        },
        func=_gaussian_blur,
    ),
    "median": OperationSpec(
        id="median",
        display="Median Blur",
        category="Filtering",
        controls={
            "kernel": dict(type="spin", min=1, max=31, step=2, default=5),
        },
        func=_median_blur,
    ),
    "bilateral": OperationSpec(
        id="bilateral",
        display="Bilateral Filter",
        category="Filtering",
        controls={
            "diameter": dict(type="spin", min=1, max=15, step=1, default=7),
            "sigma": dict(type="spin", min=1, max=150, step=5, default=50),
        },
        func=_bilateral,
    ),
    "box": OperationSpec(
        id="box",
        display="Box Filter",
        category="Filtering",
        controls={
            "kernel": dict(type="spin", min=1, max=31, step=2, default=3),
        },
        func=_box_blur,
    ),
    "unsharp": OperationSpec(
        id="unsharp",
        display="Unsharp Mask",
        category="Filtering",
        controls={
            "kernel": dict(type="spin", min=1, max=31, step=2, default=5),
            "amount": dict(type="dspin", min=0.1, max=2.0, step=0.1, default=0.7),
        },
        func=_unsharp_mask,
    ),
    "emboss": OperationSpec(
        id="emboss",
        display="Emboss",
        category="Filtering",
        controls={},
        func=_emboss,
    ),
    "sharpen": OperationSpec(
        id="sharpen",
        display="Sharpen",
        category="Filtering",
        controls={},
        func=_sharpen,
    ),
    # Edge detection ----------------------------------------------------------
    "sobel": OperationSpec(
        id="sobel",
        display="Sobel Edge",
        category="Edge",
        controls={},
        func=_sobel,
    ),
    "laplacian": OperationSpec(
        id="laplacian",
        display="Laplacian Edge",
        category="Edge",
        controls={},
        func=_laplacian,
    ),
    "canny": OperationSpec(
        id="canny",
        display="Canny Edge",
        category="Edge",
        controls={
            "low": dict(type="spin", min=0, max=255, step=1, default=100),
            "high": dict(type="spin", min=0, max=255, step=1, default=200),
        },
        func=_canny,
    ),
    # Morphology --------------------------------------------------------------
    "morph": OperationSpec(
        id="morph",
        display="Morphology",
        category="Morphology",
        controls={
            "op": dict(type="combo", items=list(_MORPH_MAP.keys()), default="erode"),
            "kernel": dict(type="spin", min=1, max=31, step=2, default=3),
            "iters": dict(type="spin", min=1, max=10, step=1, default=1),
        },
        func=_morphology,
    ),
}

# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------

def get_operations() -> Dict[str, OperationSpec]:
    """Return a *copy* of the operation table so callers cannot mutate globals."""
    return ALL_OPS.copy()
