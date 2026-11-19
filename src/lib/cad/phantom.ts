import { gaussian, rng, type Img } from "./imgops";

/**
 * Synthetic MLO-like phantom used ONLY as a software self-test (known ground truth).
 * It is not a clinical image and results on it say nothing about diagnostic accuracy.
 */
export interface PhantomTruth {
  mass: { cx: number; cy: number; rx: number; ry: number };
  calcs: Array<[number, number]>;
  marker: [number, number];
}

export function makePhantom(w = 480, h = 680, seed = 7): { gray: Img; truth: PhantomTruth } {
  const rand = rng(seed);
  const gauss = () => {
    const u = Math.max(1e-9, rand()),
      v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const cx = w - 10,
    cy = Math.round(h * 0.56),
    a = Math.round(w * 0.82),
    b = Math.round(h * 0.43);
  const inBreast = (x: number, y: number) => ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1 || (x >= w * 0.68 && y <= cy);
  const tex = new Float32Array(w * h);
  for (let i = 0; i < tex.length; i++) tex[i] = gauss();
  const texS = gaussian(tex, w, h, 5);
  const texL = gaussian(tex, w, h, 14);
  const img = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (inBreast(x, y)) img[y * w + x] = 48 + texS[y * w + x] * 55 + texL[y * w + x] * 110;

  // fibroglandular blobs
  for (let k = 0; k < 7; k++) {
    const bx = w * (0.3 + 0.6 * rand()),
      by = h * (0.45 + 0.45 * rand()),
      s = 25 + 25 * rand();
    for (let y = Math.max(0, by - 3 * s); y < Math.min(h, by + 3 * s); y++)
      for (let x = Math.max(0, bx - 3 * s); x < Math.min(w, bx + 3 * s); x++)
        if (inBreast(x | 0, y | 0)) img[(y | 0) * w + (x | 0)] += 28 * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s));
  }
  // vessels (curvy bright lines)
  for (let v = 0; v < 6; v++) {
    let x = w * (0.4 + 0.5 * rand()),
      y = h * (0.35 + 0.1 * rand());
    let ang = Math.PI * (0.55 + 0.4 * rand());
    for (let s = 0; s < 330; s++) {
      ang += (rand() - 0.5) * 0.12;
      x += Math.cos(ang) * 1.5;
      y += Math.sin(ang) * 1.5;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xi = Math.round(x) + dx,
            yi = Math.round(y) + dy;
          if (xi >= 0 && yi >= 0 && xi < w && yi < h && inBreast(xi, yi)) img[yi * w + xi] += 26;
        }
    }
  }
  // pectoral muscle: x >= 0.45*y + 0.62*w
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const xe = 0.45 * y + 0.62 * w;
      if (x >= xe && inBreast(x, y)) img[y * w + x] = Math.max(img[y * w + x], 0) * 0.35 + 150 + texS[y * w + x] * 12;
    }
  // skin line
  const bnd = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      if (!inBreast(x, y)) continue;
      if (!inBreast(x - 1, y) || !inBreast(x, y - 1) || !inBreast(x, y + 1)) bnd[y * w + x] = 1;
    }
  const bsm = gaussian(Float32Array.from(bnd), w, h, 2);
  for (let i = 0; i < img.length; i++) img[i] += bsm[i] * 160;

  // mass
  const mass = { cx: Math.round(w * 0.45), cy: Math.round(h * 0.62), rx: 46, ry: 38 };
  const mAdd = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const d = ((x - mass.cx) / mass.rx) ** 2 + ((y - mass.cy) / mass.ry) ** 2;
      if (d <= 1) mAdd[y * w + x] = 1;
    }
  const mSm = gaussian(mAdd, w, h, 1.5);
  for (let i = 0; i < img.length; i++) img[i] = img[i] * (1 - mSm[i] * 0.4) + mSm[i] * 120;

  // calcification cluster
  const calcs: Array<[number, number]> = [];
  const ccx = w * 0.4,
    ccy = h * 0.3;
  for (let k = 0; k < 14; k++) {
    const ang = rand() * Math.PI * 2,
      rr = 28 * Math.sqrt(rand());
    const px = Math.round(ccx + Math.cos(ang) * rr),
      py = Math.round(ccy + Math.sin(ang) * rr);
    calcs.push([px, py]);
    for (let dy = -3; dy <= 3; dy++)
      for (let dx = -3; dx <= 3; dx++) img[(py + dy) * w + px + dx] += 75 * Math.exp(-(dx * dx + dy * dy) / (2 * 0.9 * 0.9));
  }
  // two bright round dots inside pectoral (must be excluded)
  for (const [px, py] of [
    [Math.round(w * 0.92), 70],
    [Math.round(w * 0.9), 110],
  ])
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) img[(py + dy) * w + px + dx] += 70 * Math.exp(-(dx * dx + dy * dy) / 3);

  // skin marker (saturated disc just inside the skin line, lower breast)
  const my = Math.round(h * 0.88);
  let mxx = 0;
  while (!inBreast(mxx, my)) mxx++;
  const marker: [number, number] = [mxx + 6, my];
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) if (dx * dx + dy * dy <= 9) img[(my + dy) * w + marker[0] + dx] = 255;

  // acquisition blur + noise, background stays ~0
  const out = gaussian(img, w, h, 0.8);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(255, out[i] + gauss() * 1.6 + (out[i] > 1 ? 0 : 0)));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (!inBreast(x, y) && out[y * w + x] < 3) out[y * w + x] = Math.abs(gauss()) * 0.6;
  return { gray: out, truth: { mass, calcs, marker } };
}
