"""Stage 4 - microcalcification candidates, shape gating, DBSCAN clustering."""
from __future__ import annotations
import cv2
import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
from .image_preprocessing import gaussian


def tophat_response(N, sigma: float = 0.7, radii=(2, 3, 5)):
    """T(x) = max_r [ S - open_r(S) ],  S = Gaussian(N, sigma);  disk structuring elements of radius r.
    White top-hat keeps bright structures smaller than the SE (calcifications 2-10 px), removes
    anything larger (glandular tissue, masses, pectoral)."""
    S = gaussian(N, sigma)
    T = np.zeros_like(S)
    for r in radii:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        T = np.maximum(T, S - cv2.morphologyEx(S, cv2.MORPH_OPEN, se))
    return T


def local_z(T, radius: int = 24):
    """Z = (T - mu_loc) / (sigma_loc + 1) with box statistics (box = 2*radius+1) -> adapts to dense/noisy regions."""
    k = (2 * radius + 1, 2 * radius + 1)
    mu = cv2.boxFilter(T, -1, k, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(T * T, -1, k, borderType=cv2.BORDER_REFLECT)
    return (T - mu) / (np.sqrt(np.maximum(mu2 - mu * mu, 0)) + 1.0)


def detect_calcifications(N, eligible, mass_union=None, z_thr: float = 4.5, t_abs: float = 6.0,
                          min_area: int = 2, max_area: int = 80, max_elong: float = 3.5,
                          max_candidates: int = 300, eps_px: float | None = None, pixel_mm: float = 0.0,
                          min_pts: int = 5):
    H, W = N.shape
    T = tophat_response(N)
    Z = local_z(T)
    elig = eligible.copy()
    elig[:3] = elig[-3:] = 0
    elig[:, :3] = elig[:, -3:] = 0
    kz = z_thr
    for _ in range(8):  # adaptive noise guard
        low = ((Z >= .5 * kz) & (T >= .5 * t_abs) & (elig > 0)).astype(np.uint8)  # hysteresis low mask
        lab = label(low, connectivity=2)
        props = regionprops(lab, intensity_image=Z)
        ok = [p for p in props if p.intensity_max >= kz and min_area <= p.area <= max_area]
        if len(ok) <= max_candidates:
            break
        kz += .5
    cands, rej = [], dict(small=0, large=0, elongated=0, low_peak=0)
    for p in props:
        peakT = float(T[lab == p.label].max())
        if p.area < min_area: rej["small"] += 1; continue
        if p.area > max_area: rej["large"] += 1; continue
        elong = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        if p.area >= 4 and elong > max_elong: rej["elongated"] += 1; continue
        if p.intensity_max < kz or peakT < t_abs: rej["low_peak"] += 1; continue
        cy, cx = p.centroid
        cands.append(dict(x=float(cx), y=float(cy), area=int(p.area), peak_tophat=peakT, peak_z=float(p.intensity_max),
                          elongation=float(elong), inside_mass=bool(mass_union is not None and mass_union[int(cy), int(cx)])))
    eps = eps_px if eps_px else (5.0 / pixel_mm if pixel_mm > 0 else 0.04 * max(H, W))
    clusters = []
    if len(cands) >= min_pts:
        xy = np.array([[c["x"], c["y"]] for c in cands])
        lbl = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(xy)
        for k in sorted(set(lbl) - {-1}):
            idx = np.where(lbl == k)[0]
            hull = cv2.convexHull(xy[idx].astype(np.float32))[:, 0, :]
            clusters.append(dict(members=idx.tolist(), hull=hull.tolist(), count=int(idx.size), hull_area=float(cv2.contourArea(hull))))
            for i in idx:
                cands[i]["cluster"] = k
    return cands, clusters, dict(z_used=kz, rejected=rej, eps=eps)
