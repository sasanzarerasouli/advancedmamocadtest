"""Layered overlays: original is never modified; every layer is alpha-blended on a copy."""
from __future__ import annotations
import cv2
import numpy as np

BGR = dict(breast=(120, 220, 60), pectoral=(255, 120, 60), skin=(40, 210, 250), nipple=(220, 70, 240),
           marker=(20, 140, 255), mass=(60, 50, 255), calc=(240, 230, 40), cluster=(60, 240, 255))


def blend(img, mask, color, alpha=.3):
    out = img.copy()
    sel = mask > 0
    out[sel] = (out[sel] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def contour(img, mask, color, th=1):
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, th)
    return img


def render(gray8, res: dict, layers=("anatomy", "masses", "calcs", "clusters", "labels")):
    img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    s = max(1.0, max(img.shape[:2]) / 700)
    if "anatomy" in layers:
        for k in ("skin", "pectoral", "nipple", "marker"):
            img = blend(img, res["masks"][k], BGR[k], .3)
        contour(img, res["masks"]["breast"], BGR["breast"])
    if "masses" in layers:
        for m in res["masses"]:
            img = blend(img, m["mask"], BGR["mass"], .28)
            contour(img, m["mask"], BGR["mass"], 2)
            if "labels" in layers:
                x, y, w, h = cv2.boundingRect(m["mask"])
                cv2.putText(img, f'{m["id"]} {m["confidence"]:.0%}', (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5 * s, BGR["mass"], 1, cv2.LINE_AA)
    if "calcs" in layers:
        for c in res["calcs"]:
            cv2.circle(img, (int(c["x"]), int(c["y"])), int(5 * s), BGR["calc"], 1, cv2.LINE_AA)
    if "clusters" in layers:
        for i, cl in enumerate(res["clusters"], 1):
            cv2.polylines(img, [np.array(cl["hull"], np.int32)], True, BGR["cluster"], 2, cv2.LINE_AA)
    return img
