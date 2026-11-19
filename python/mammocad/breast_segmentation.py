"""Stage 2a - breast region + orientation."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from .image_preprocessing import gaussian


def segment_breast(gray: np.ndarray, k_std: float = 5.0, min_offset: float = 6.0, blur: float = 1.5):
    """Background-statistics threshold (robust to dark fatty breasts where Otsu fails).

        bg    = mode of histogram in [0, 60]
        sigma = std of pixels <= bg + 10
        T     = bg + max(min_offset, k_std * sigma)
        mask  = largest connected component of (G_blur > T), opened r=2, holes filled, closed r=3
    """
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256)
    bg = int(np.argmax(hist[:61]))
    sig = float(gray[gray <= bg + 10].std())
    thr = bg + max(min_offset, k_std * sig)
    m = (gaussian(gray, blur) > thr).astype(np.uint8)
    se = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se(2))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(m), dict(bg=bg, sigma=sig, thr=thr)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)  # drops laterality label, markers, text
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se(3))
    return ndi.binary_fill_holes(m).astype(np.uint8), dict(bg=bg, sigma=sig, thr=thr)


def chest_wall_is_right(mask: np.ndarray) -> bool:
    """Chest wall = image side where the breast mask touches the border over more rows."""
    return int(mask[:, -1].sum()) >= int(mask[:, 0].sum())


def row_leftmost(mask: np.ndarray) -> np.ndarray:
    any_ = mask.any(axis=1)
    xb = np.where(any_, mask.argmax(axis=1), -1)
    return xb
