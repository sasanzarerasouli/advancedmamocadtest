"""Stage 3a - mass candidate localisation (scale-space DoG blobs, anatomy-aware)."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from .image_preprocessing import gaussian

SIGMA_FRACS = (0.009, 0.013, 0.019, 0.027, 0.039, 0.056, 0.08, 0.115)


def _norm_conv(img, mask, s):
    """Normalised convolution: G*(I.M) / G*M so excluded pixels do not leak into blob responses."""
    a, b = gaussian(img * mask, s), gaussian(mask, s)
    return np.where(b > 0.05, a / np.maximum(b, 1e-6), np.median(img[mask > 0]))


def mass_candidates(N, eligible, z_thr: float = 4.0, max_cand: int = 6, sigma_fracs=SIGMA_FRACS):
    """D_s = G_s - G_1.6s (a scale-normalised LoG approximation, blob radius r = sqrt2*s).
    Z_s = (D_s - median) / (1.4826 MAD) computed over eligible pixels only.  Keep 3x3x3 local maxima
    with Z >= z_thr whose disc is >= 70 % eligible tissue, NMS at 0.7*max(r1, r2).
    Run on an image down-sampled to ~200 px for speed (blob finding does not need native resolution)."""
    H, W = N.shape
    f = max(1, round(max(H, W) / 200))
    sm = cv2.resize(N, (W // f, H // f), interpolation=cv2.INTER_AREA)
    el = (cv2.resize(eligible.astype(np.float32), (W // f, H // f), interpolation=cv2.INTER_AREA) > 0.6).astype(np.float32)
    sig = [max(1.2, fr * max(H, W) / f) for fr in sigma_fracs]
    stack = []
    for s in sig:
        d = _norm_conv(sm, el, s) - _norm_conv(sm, el, 1.6 * s)
        v = d[el > 0]
        med = np.median(v)
        mad = max(np.median(np.abs(v - med)) * 1.4826, 0.25)
        stack.append((d - med) / mad)
    Z = np.stack(stack)
    peaks = (Z == ndi.maximum_filter(Z, size=(3, 3, 3))) & (Z >= z_thr) & (el[None] > 0)
    out = []
    for si, y, x in zip(*np.nonzero(peaks)):
        r = sig[si] * np.sqrt(2)
        yy, xx = np.ogrid[:el.shape[0], :el.shape[1]]
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        if el[disc].mean() < 0.7:
            continue
        out.append(dict(x=x * f + f / 2, y=y * f + f / 2, r=r * f, z=float(Z[si, y, x])))
    out.sort(key=lambda c: -c["z"])
    kept = []
    for c in out:
        if all(np.hypot(k["x"] - c["x"], k["y"] - c["y"]) >= 0.7 * max(k["r"], c["r"]) for k in kept):
            kept.append(c)
        if len(kept) >= max_cand:
            break
    return kept
