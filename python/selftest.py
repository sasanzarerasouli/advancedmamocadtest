"""Synthetic-phantom smoke test (NOT a clinical validation).  python python/selftest.py"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mammocad import run_pipeline

def phantom(w=480, h=680, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, a, b = w - 10, .56 * h, .82 * w, .43 * h
    inb = ((((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1) | ((xx >= .68 * w) & (yy <= cy))
    tex = lambda s: cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), s)
    img = np.where(inb, 48 + tex(5) * 55 + tex(14) * 110, 0).astype(np.float32)
    pect = (xx >= .45 * yy + .62 * w) & inb
    img[pect] = 150 + tex(5)[pect] * 12
    m = (((xx - .45 * w) / 46) ** 2 + ((yy - .62 * h) / 38) ** 2 <= 1).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    img = img * (1 - .4 * m) + 120 * m
    pts = []
    for _ in range(14):
        ang, r = rng.uniform(0, 2 * np.pi), 28 * np.sqrt(rng.uniform())
        px, py = int(.4 * w + np.cos(ang) * r), int(.3 * h + np.sin(ang) * r); pts.append((px, py))
        img[py - 3:py + 4, px - 3:px + 4] += 75 * np.exp(-((np.arange(-3, 4)[None] ** 2 + np.arange(-3, 4)[:, None] ** 2) / 1.6))
    img = cv2.GaussianBlur(img, (0, 0), .8) + rng.normal(0, 1.6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), pts

if __name__ == "__main__":
    img, truth = phantom()
    res = run_pipeline(img)
    print("pectoral:", res["pectoral"]); print("nipple:", res["nipple"])
    print("masses:", [(m["id"], round(m["confidence"], 2), m["features"]["area"]) for m in res["masses"]])
    print("calcs:", len(res["calcs"]), "clusters:", [c["count"] for c in res["clusters"]], "z_used", res["calc_info"]["z_used"])
    hit = sum(any(np.hypot(c["x"] - x, c["y"] - y) < 3 for c in res["calcs"]) for x, y in truth)
    print(f"phantom calcification recall {hit}/{len(truth)}")
    assert res["masses"] and res["pectoral"]["found"], "self-test failed"
