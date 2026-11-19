"""Stage 2b - pectoral muscle, skin band, nipple, markers, exclusion map.
Convention: images are mirrored so the chest wall is on the RIGHT before calling these."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from .image_preprocessing import gaussian


def detect_pectoral(N, breast, xb, min_contrast: float = 5.0, seed: int = 12345):
    """Edge-point extraction + RANSAC line + contrast validation.

    For every 2nd row y < 0.75 H, scanning x, the boundary candidate is
        x*(y) = argmax_x  mean(G[y, x+1 .. x+k]) - mean(G[y, x-k .. x-1]),   G = Gaussian(N, 3), k = 2% W
    (strongest dark->bright step towards the chest wall).  Points with response >= max(4, 0.4*P90)
    are fitted with  x = a*y + b  by RANSAC (300 iters, tolerance 8 px, a in [0.1, 2]).
    Validation: mean(N) inside (6-20 px) minus outside (6-20 px) of the line >= min_contrast and the line
    must meet the chest wall at y_end in [0.12H, ...].  Returns (mask, info).  CC views fail the
    validation (no top-border contact / no oblique line) and return an empty mask."""
    H, W = N.shape
    empty = np.zeros_like(breast)
    if not breast[0].any():
        return empty, dict(found=False, note="breast does not reach top border")
    G = gaussian(N, 3.0)
    k = max(6, round(0.02 * W))
    ker = np.r_[-np.ones(k), 0, np.ones(k)] / k
    resp = cv2.filter2D(G, -1, ker.reshape(1, -1).astype(np.float32), borderType=cv2.BORDER_REPLICATE)
    pts = []
    for y in range(0, int(0.75 * H), 2):
        if xb[y] < 0:
            continue
        lo, hi = max(k + 1, xb[y] + int(0.06 * W)), W - 1 - k
        if hi <= lo:
            continue
        x = lo + int(np.argmax(resp[y, lo:hi + 1]))
        if resp[y, x] > 0:
            pts.append((x, y, resp[y, x]))
    if len(pts) < 12:
        return empty, dict(found=False, note="too few edge points")
    pts = np.array(pts)
    pts = pts[pts[:, 2] >= max(4.0, 0.4 * np.percentile(pts[:, 2], 90))]
    if len(pts) < 12:
        return empty, dict(found=False, note="weak edge response")
    rng = np.random.default_rng(seed)
    best, best_s = None, 0.0
    for _ in range(300):
        i, j = rng.integers(0, len(pts), 2)
        if abs(pts[i, 1] - pts[j, 1]) < 0.1 * H:
            continue
        a = (pts[j, 0] - pts[i, 0]) / (pts[j, 1] - pts[i, 1])
        b = pts[i, 0] - a * pts[i, 1]
        if not 0.1 <= a <= 2:
            continue
        inl = np.abs(pts[:, 0] - (a * pts[:, 1] + b)) <= 8
        s = pts[inl, 2].sum()
        if s > best_s:
            best, best_s = inl, s
    if best is None or best.sum() < 12:
        return empty, dict(found=False, note="no RANSAC consensus")
    a, b = np.polyfit(pts[best, 1], pts[best, 0], 1)
    y_end = (W - 1 - b) / a
    if not (0.1 <= a <= 2 and b >= 0.2 * W and y_end >= 0.12 * H):
        return empty, dict(found=False, note="implausible line")
    ins, outs = [], []
    for y in range(0, min(H - 1, int(0.9 * y_end)), 4):
        xe = int(round(a * y + b))
        if xe - 20 < 0 or xe + 20 >= W:
            continue
        ins.append(N[y, xe + 6:xe + 20].mean())
        outs.append(N[y, xe - 20:xe - 6].mean())
    contrast = float(np.mean(ins) - np.mean(outs)) if ins else 0.0
    if contrast < min_contrast:
        return empty, dict(found=False, note=f"no brightness step ({contrast:.1f})")
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx >= a * yy + b) & (yy <= y_end) & (breast > 0)).astype(np.uint8)
    return mask, dict(found=True, a=float(a), b=float(b), y_end=float(y_end), contrast=contrast, inliers=int(best.sum()))


def skin_band(breast, frac: float = 0.012):
    """Pixels whose distance to the BACKGROUND (not to the image border) is <= max(3, frac*W)."""
    d = cv2.distanceTransform(breast.astype(np.uint8), cv2.DIST_L2, 3)
    return ((d <= max(3, round(frac * breast.shape[1]))) & (breast > 0)).astype(np.uint8), d


def detect_nipple(breast, xb, radius_frac: float = 0.05):
    """Apex = row where the smoothed left boundary is furthest from the chest wall.
    In-profile test compares narrow curvature (+-4% H) with wide curvature (+-12% H):
        bump = mean(xb[y*-d], xb[y*+d]) - xb[y*]  >  max(0.008 W, 0.3 * wide)
    A circular breast gives bump ~ wide/9; a true nipple gives much more.  A precautionary
    disc (r = 5% of breast depth, x1.3 if in profile) is excluded either way."""
    H, W = breast.shape
    ys = np.where(xb >= 0)[0]
    lo, hi = ys[0] + int(0.12 * (ys[-1] - ys[0])), ys[-1] - int(0.1 * (ys[-1] - ys[0]))
    sm = np.convolve(np.where(xb >= 0, xb, W).astype(float), np.ones(7) / 7, mode="same")
    ay = int(np.median(np.where(sm[lo:hi + 1] <= sm[lo:hi + 1].min() + 2)[0]) + lo)
    ax = int(xb[ay])
    v = lambda y: xb[min(max(y, 0), H - 1)] if xb[min(max(y, 0), H - 1)] >= 0 else ax
    d, D = round(0.04 * H), round(0.12 * H)
    bump = (v(ay - d) + v(ay + d)) / 2 - ax
    wide = (v(ay - D) + v(ay + D)) / 2 - ax
    in_prof = bump > max(0.008 * W, 0.3 * wide)
    r = max(8, radius_frac * (W - 1 - ax)) * (1.3 if in_prof else 1.0)
    m = np.zeros_like(breast)
    cv2.circle(m, (ax, ay), int(round(r)), 1, -1)
    return (m & breast).astype(np.uint8), dict(apex=(ax, ay), in_profile=bool(in_prof), radius=float(r))


def detect_markers(N, breast, dist, skin_px: int, max_area: int = 700):
    """Saturated (>= 250 after normalisation, i.e. top 0.2 % of breast) compact blobs within 4 skin-widths
    of the skin line.  Metal markers/BBs are near-saturated, compact (fill>=0.5, elongation<=3) and
    sit on the skin; calcifications are 10-100x smaller and dimmer; pectoral/mass are far larger."""
    cand = ((N >= 250) & (breast > 0)).astype(np.uint8)
    lab = label(cand, connectivity=2)
    out = np.zeros_like(breast)
    items = []
    for r in regionprops(lab):
        h0, w0, h1, w1 = r.bbox
        fill = r.area / ((h1 - h0) * (w1 - w0))
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-6)
        cy, cx = map(int, r.centroid)
        if 3 <= r.area <= max_area and fill >= 0.5 and elong <= 3 and dist[cy, cx] <= 4 * skin_px:
            cv2.circle(out, (cx, cy), int(max(h1 - h0, w1 - w0) / 2 + 3), 1, -1)
            items.append(dict(centroid=(cx, cy), area=int(r.area)))
    return out, items


def exclusion_map(breast, pectoral, skin, nipple, marker, pect_margin: int = 3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pect_margin + 1,) * 2)
    pect = cv2.dilate(pectoral, k)
    excl = (pect | skin | nipple | marker).astype(np.uint8)
    return excl, ((breast > 0) & (excl == 0)).astype(np.uint8)
