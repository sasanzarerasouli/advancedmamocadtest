"""Orchestrator + CLI:  python -m mammocad.pipeline image.png --out outdir"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
import cv2
import numpy as np
from . import image_preprocessing as pre, breast_segmentation as bs, anatomical_segmentation as an
from . import mass_detection as md, mass_segmentation as ms, calcification_detection as cd
from . import candidate_classification as cc, visualization as vz


@dataclass
class PipelineConfig:
    calc_z: float = 4.5
    mass_z: float = 4.0
    pixel_mm: float = 0.0
    skin_frac: float = 0.012


def run_pipeline(img: np.ndarray, cfg: PipelineConfig = PipelineConfig()) -> dict:
    gray = pre.to_gray_float(img)
    H, W = gray.shape
    breast, binfo = bs.segment_breast(gray)
    flipped = not bs.chest_wall_is_right(breast)
    if flipped:
        gray, breast = gray[:, ::-1].copy(), breast[:, ::-1].copy()
    N = pre.normalize_in_mask(gray, breast)
    xb = bs.row_leftmost(breast)
    pect, pinfo = an.detect_pectoral(N, breast, xb)
    skin, dist = an.skin_band(breast, cfg.skin_frac)
    nip, ninfo = an.detect_nipple(breast, xb)
    mark, minfo = an.detect_markers(N, breast, dist, max(3, round(cfg.skin_frac * W)))
    excl, elig = an.exclusion_map(breast, pect, skin, nip, mark)
    Ns = pre.gaussian(N, 1.5)
    masses, rejected = [], []
    area_b = int(breast.sum())
    for c in md.mass_candidates(N, elig, cfg.mass_z):
        if any(m["mask"][int(c["y"]), int(c["x"])] for m in masses):
            continue
        seg = ms.segment_mass(N, elig, breast, c)
        if seg is None:
            rejected.append(dict(c, reason="no separable bright class")); continue
        mask, sinfo = seg
        f = ms.mass_features(mask, Ns, elig)
        f.update(blob_z=c["z"])
        ok, why = cc.rule_gate_mass(f, area_b)
        if ok and sinfo["excluded_frac"] > .4:
            ok, why = False, "overlaps excluded anatomy"
        if not ok:
            rejected.append(dict(c, reason=why)); continue
        conf = float(np.cbrt(min(1, c["z"] / 8) * min(1, f["contrast"] / 40) * min(1, f["axis_ratio"] / .6)))
        masses.append(dict(id=f"M{len(masses) + 1}", mask=mask, confidence=conf, features=f, **sinfo))
    mu = np.zeros_like(breast)
    for m in masses:
        mu |= m["mask"]
    calcs, clusters, cinfo = cd.detect_calcifications(N, elig, mu, z_thr=cfg.calc_z, pixel_mm=cfg.pixel_mm)
    res = dict(flipped=flipped, breast_info=binfo, pectoral=pinfo, nipple=ninfo, markers=minfo, masses=masses,
               rejected_masses=rejected, calcs=calcs, clusters=clusters, calc_info=cinfo,
               masks=dict(breast=breast, pectoral=pect, skin=skin, nipple=nip, marker=mark, excluded=excl))
    if flipped:  # return to the original orientation
        for k, v in res["masks"].items():
            res["masks"][k] = v[:, ::-1].copy()
        for m in res["masses"]:
            m["mask"] = m["mask"][:, ::-1].copy()
        for c in res["calcs"]:
            c["x"] = W - 1 - c["x"]
        for cl in res["clusters"]:
            cl["hull"] = [[W - 1 - x, y] for x, y in cl["hull"]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--out", default="out"); ap.add_argument("--pixel-mm", type=float, default=0.0)
    a = ap.parse_args()
    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    res = run_pipeline(img, PipelineConfig(pixel_mm=a.pixel_mm))
    g8 = np.clip(pre.to_gray_float(img), 0, 255).astype(np.uint8)
    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(f"{a.out}/1_original.png", g8)
    cv2.imwrite(f"{a.out}/2_anatomy.png", vz.render(g8, res, ("anatomy", "labels")))
    cv2.imwrite(f"{a.out}/3_masses.png", vz.render(g8, res, ("masses", "labels")))
    cv2.imwrite(f"{a.out}/4_calcs.png", vz.render(g8, res, ("calcs", "clusters")))
    cv2.imwrite(f"{a.out}/5_combined.png", vz.render(g8, res))
    ser = {k: v for k, v in res.items() if k not in ("masks",)}
    ser["masses"] = [{k: v for k, v in m.items() if k != "mask"} for m in res["masses"]]
    json.dump(ser, open(f"{a.out}/findings.json", "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
