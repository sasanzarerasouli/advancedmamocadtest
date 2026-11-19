"""Stage 1 - preprocessing.  All functions are pure (no global state)."""
from __future__ import annotations
import cv2
import numpy as np


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """BGR/RGB/uint8/uint16 -> float32 grey in [0, 255].  16-bit input is scaled by its own max."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 255:
        img = img / img.max() * 255.0
    return img


def normalize_in_mask(gray: np.ndarray, mask: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8) -> np.ndarray:
    """N = clip((I - P_lo) / (P_hi - P_lo), 0, 1) * 255 inside `mask`, 0 outside.

    Percentiles are computed from breast pixels only, so the black background and the
    burnt-in laterality label cannot compress the dynamic range.
    """
    vals = gray[mask > 0]
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    n = np.clip((gray - lo) / max(hi - lo, 1.0), 0, 1) * 255.0
    return (n * (mask > 0)).astype(np.float32)


def clahe(n: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE for *display / mass-margin viewing only*.  Do NOT feed it to the calcification
    detector: CLAHE amplifies noise non-linearly and breaks the Z-score calibration."""
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return c.apply(np.clip(n, 0, 255).astype(np.uint8)).astype(np.float32)


def gaussian(n: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(n, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def resize_limit(img: np.ndarray, max_side: int = 3000) -> tuple[np.ndarray, float]:
    """Down-size only if larger than max_side (INTER_AREA).  Returns (image, scale).
    A 0.07 mm pixel with a 0.2 mm calcification is ~3 px; halving the image leaves 1.5 px
    and makes the tophat SNR collapse, so keep native resolution for calcification detection."""
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s
