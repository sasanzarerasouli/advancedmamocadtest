"""Stage 3b - lesion boundary: local 2-class Fuzzy C-Means + morphology + curvature smoothing."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from .image_preprocessing import gaussian


def fcm_1d(v: np.ndarray, m: float = 2.0, iters: int = 40, eps: float = 1e-3):
    """Fuzzy C-Means, c=2, on intensities.   J = sum_i sum_k u_ik^m (v_i - c_k)^2
        u_ik = 1 / sum_j (|v_i-c_k| / |v_i-c_j|)^(2/(m-1)),   c_k = sum u_ik^m v_i / sum u_ik^m
    Initial centres: P20 / P85 of the window."""
    c = np.array([np.percentile(v, 20), np.percentile(v, 85)], dtype=float)
    for _ in range(iters):
        d = np.abs(v[:, None] - c[None]) + 1e-6
        u = 1.0 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)
        nc = (u ** m * v[:, None]).sum(0) / (u ** m).sum(0)
        if np.abs(nc - c).max() < eps:
            c = nc
            break
        c = nc
    return c


def segment_mass(N, eligible, breast, cand, fcm_sigma: float = 1.5):
    """Returns (mask uint8, dict(centers, excluded_frac, lucent_frac)) or None.

    1. window = 2.2 r around the blob; FCM on eligible window pixels (<=20k samples);
    2. bright = membership > 0.5 (== v > mean of the two fuzzy centres), restricted to breast & disc(1.7 r);
    3. open (r = max(2, .06 r)), close (r = max(2, .1 r)), fill holes; keep component under the blob centre;
    4. mean-curvature-flow surrogate: Gaussian(sigma = max(1.5, .04 r)) of the indicator, threshold .5;
    5. remove excluded anatomy; keep largest component."""
    H, W = N.shape
    Ns = gaussian(N, fcm_sigma)
    r = cand["r"]
    half = int(np.ceil(2.2 * r))
    cx, cy = int(round(cand["x"])), int(round(cand["y"]))
    x0, y0, x1, y1 = max(0, cx - half), max(0, cy - half), min(W, cx + half + 1), min(H, cy + half + 1)
    L, E, B = Ns[y0:y1, x0:x1], eligible[y0:y1, x0:x1], breast[y0:y1, x0:x1]
    v = L[E > 0]
    if v.size < 200:
        return None
    v = v[:: max(1, v.size // 20000)]
    c = fcm_1d(v.astype(np.float64))
    if c[1] - c[0] < 8:
        return None
    lcx, lcy = cx - x0, cy - y0
    yy, xx = np.ogrid[:L.shape[0], :L.shape[1]]
    disc = (yy - lcy) ** 2 + (xx - lcx) ** 2 <= (1.7 * r) ** 2
    m0 = ((L > c.mean()) & (B > 0) & disc).astype(np.uint8)
    se = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
    m1 = cv2.morphologyEx(cv2.morphologyEx(m0, cv2.MORPH_OPEN, se(max(2, round(.06 * r)))), cv2.MORPH_CLOSE, se(max(2, round(.1 * r))))
    pre = m1.copy()
    m1 = ndi.binary_fill_holes(m1).astype(np.uint8)
    lab = label(m1, connectivity=2)
    k = lab[min(lcy, lab.shape[0] - 1), min(lcx, lab.shape[1] - 1)]
    if k == 0:
        return None
    m2 = (lab == k).astype(np.float32)
    area0 = m2.sum()
    m2 = (cv2.GaussianBlur(m2, (0, 0), max(1.5, .04 * r)) > .5).astype(np.uint8)
    fin = (m2 & (E > 0)).astype(np.uint8)
    lab = label(fin, connectivity=2)
    if lab.max() == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    fin = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    if fin.sum() < 20:
        return None
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = fin
    return full, dict(centers=c.tolist(), excluded_frac=float(max(0, 1 - fin.sum() / area0)),
                      lucent_frac=float(((m2 > 0) & (pre == 0)).sum() / max(1, fin.sum())))


def mass_features(mask, Ns, eligible):
    """Shape/margin/contrast descriptors used by classify.py (see Methodology for definitions)."""
    rp = regionprops(mask.astype(np.uint8))[0]
    ring = cv2.dilate(mask, np.ones((0 + 2 * max(4, round(.12 * rp.equivalent_diameter_area / 2)) + 5,) * 2, np.uint8)) & ~cv2.dilate(mask, np.ones((5, 5), np.uint8)) & eligible
    inside, around = Ns[mask > 0], Ns[ring > 0]
    d_n = float(inside.mean() - around.mean()) if around.size else 0.0
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = cv2.arcLength(max(cnt, key=cv2.contourArea), True)
    gx, gy = cv2.Sobel(Ns, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(Ns, cv2.CV_32F, 0, 1, ksize=3) / 8
    edge = (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0
    sharp = float(np.hypot(gx, gy)[edge].mean() / max(d_n / (1.5 * np.sqrt(2 * np.pi)), 1e-6))
    cc = max(cnt, key=cv2.contourArea)[:, 0, :]
    ang = np.arctan2(cc[:, 1] - rp.centroid[0], cc[:, 0] - rp.centroid[1])
    rad = np.hypot(cc[:, 1] - rp.centroid[0], cc[:, 0] - rp.centroid[1])
    bins = np.digitize(ang, np.linspace(-np.pi, np.pi, 37))
    rm = np.array([np.median(rad[bins == b]) for b in np.unique(bins)])
    sm = np.convolve(np.r_[rm[-2:], rm, rm[:2]], np.ones(5) / 5, mode="valid")
    return dict(area=int(rp.area), equiv_diam=float(rp.equivalent_diameter_area), axis_ratio=float(rp.axis_minor_length / rp.axis_major_length),
                circularity=float(min(1, 4 * np.pi * rp.area / per ** 2)), solidity=float(rp.solidity), contrast=d_n,
                heterogeneity=float(inside.std() / max(inside.mean(), 1)), margin_sharpness=sharp,
                radial_roughness=float(np.sqrt(np.mean((rm - sm) ** 2)) / rm.mean()))
