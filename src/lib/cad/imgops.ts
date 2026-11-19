/**
 * Minimal, dependency-free image-processing primitives (Float32 grey images, Uint8 binary masks).
 * Everything here is deterministic and executes in the browser or in Node.
 */

export type Img = Float32Array;
export type Mask = Uint8Array;

export function gaussianKernel(sigma: number): Float32Array {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    s += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return k;
}

export function gaussian(src: Img, w: number, h: number, sigma: number): Img {
  if (sigma <= 0) return src.slice();
  const k = gaussianKernel(sigma);
  const r = (k.length - 1) / 2;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let xx = x + i;
        if (xx < 0) xx = -xx;
        else if (xx >= w) xx = 2 * w - xx - 2;
        if (xx < 0) xx = 0;
        acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let yy = y + i;
        if (yy < 0) yy = -yy;
        else if (yy >= h) yy = 2 * h - yy - 2;
        if (yy < 0) yy = 0;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/** Box-average downsample by integer factor. */
export function downsample(src: Img, w: number, h: number, f: number): { img: Img; w: number; h: number } {
  if (f <= 1) return { img: src, w, h };
  const nw = Math.floor(w / f);
  const nh = Math.floor(h / f);
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++)
    for (let x = 0; x < nw; x++) {
      let s = 0;
      for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) s += src[(y * f + j) * w + x * f + i];
      out[y * nw + x] = s / (f * f);
    }
  return { img: out, w: nw, h: nh };
}

export function diskOffsets(r: number): Array<[number, number]> {
  const o: Array<[number, number]> = [];
  const rr = r + 0.5;
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= rr * rr) o.push([dx, dy]);
  return o;
}

/** Grey-scale erosion / dilation with a disk (separable by rows of the disk for speed). */
function diskRowSpans(r: number): Array<[number, number]> {
  // for each dy in [-r..r] the half-width of the disk
  const spans: Array<[number, number]> = [];
  const rr = r + 0.5;
  for (let dy = -r; dy <= r; dy++) spans.push([dy, Math.floor(Math.sqrt(rr * rr - dy * dy))]);
  return spans;
}

function rowExtreme(src: Img, w: number, h: number, hw: number, isMin: boolean): Img {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const a = Math.max(0, x - hw);
      const b = Math.min(w - 1, x + hw);
      let v = src[row + a];
      if (isMin) for (let i = a + 1; i <= b; i++) v = Math.min(v, src[row + i]);
      else for (let i = a + 1; i <= b; i++) v = Math.max(v, src[row + i]);
      out[row + x] = v;
    }
  }
  return out;
}

const rowCache = new Map<string, Img>();
void rowCache;

export function greyMorph(src: Img, w: number, h: number, r: number, isMin: boolean): Img {
  const spans = diskRowSpans(r);
  const hwSet = new Map<number, Img>();
  for (const [, hw] of spans) if (!hwSet.has(hw)) hwSet.set(hw, rowExtreme(src, w, h, hw, isMin));
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = isMin ? Infinity : -Infinity;
      for (const [dy, hw] of spans) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        const t = hwSet.get(hw)![yy * w + x];
        if (isMin) {
          if (t < v) v = t;
        } else if (t > v) v = t;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

export const greyErode = (s: Img, w: number, h: number, r: number) => greyMorph(s, w, h, r, true);
export const greyDilate = (s: Img, w: number, h: number, r: number) => greyMorph(s, w, h, r, false);
export const greyOpen = (s: Img, w: number, h: number, r: number) => greyDilate(greyErode(s, w, h, r), w, h, r);

/** White top-hat  WTH_r(I) = I - open_r(I). */
export function whiteTophat(src: Img, w: number, h: number, r: number): Img {
  const o = greyOpen(src, w, h, r);
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = src[i] - o[i];
  return out;
}

export function binMorph(m: Mask, w: number, h: number, r: number, erode: boolean): Mask {
  if (r <= 0) return m.slice();
  const f = new Float32Array(w * h);
  for (let i = 0; i < f.length; i++) f[i] = m[i];
  const g = greyMorph(f, w, h, r, erode);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = g[i] > 0.5 ? 1 : 0;
  return out;
}
export const binErode = (m: Mask, w: number, h: number, r: number) => binMorph(m, w, h, r, true);
export const binDilate = (m: Mask, w: number, h: number, r: number) => binMorph(m, w, h, r, false);
export const binOpen = (m: Mask, w: number, h: number, r: number) =>
  binDilate(binErode(m, w, h, r), w, h, r);
export const binClose = (m: Mask, w: number, h: number, r: number) =>
  binErode(binDilate(m, w, h, r), w, h, r);

export function percentile(values: ArrayLike<number>, p: number): number {
  const a = Float32Array.from(values as ArrayLike<number>);
  a.sort();
  if (a.length === 0) return 0;
  const idx = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[idx];
}

export function median(values: ArrayLike<number>): number {
  return percentile(values, 50);
}

export function maskedValues(img: Img, m: Mask): Float32Array {
  let n = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) n++;
  const out = new Float32Array(n);
  let j = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) out[j++] = img[i];
  return out;
}

export function meanStd(values: ArrayLike<number>): { mean: number; std: number } {
  const n = values.length;
  if (!n) return { mean: 0, std: 0 };
  let s = 0;
  for (let i = 0; i < n; i++) s += values[i];
  const mean = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (values[i] - mean) ** 2;
  return { mean, std: Math.sqrt(v / n) };
}

/** Otsu threshold on 256-bin histogram of values in [0,255]. */
export function otsu(values: ArrayLike<number>): number {
  const hist = new Float64Array(256);
  for (let i = 0; i < values.length; i++) hist[Math.max(0, Math.min(255, Math.round(values[i])))]++;
  const total = values.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let wB = 0,
    sumB = 0,
    best = 0,
    thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) {
      best = between;
      thr = t;
    }
  }
  return thr;
}

export interface Component {
  label: number;
  area: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  cx: number;
  cy: number;
  /** second central moments */
  mu20: number;
  mu02: number;
  mu11: number;
}

export function labelComponents(
  m: Mask,
  w: number,
  h: number,
  conn: 4 | 8 = 8,
): { labels: Int32Array; comps: Component[] } {
  const labels = new Int32Array(w * h);
  const comps: Component[] = [];
  const stack = new Int32Array(w * h);
  let next = 0;
  for (let i = 0; i < w * h; i++) {
    if (!m[i] || labels[i]) continue;
    next++;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = next;
    let area = 0,
      sx = 0,
      sy = 0,
      sxx = 0,
      syy = 0,
      sxy = 0;
    let minX = w,
      minY = h,
      maxX = 0,
      maxY = 0;
    while (sp) {
      const p = stack[--sp];
      const x = p % w;
      const y = (p / w) | 0;
      area++;
      sx += x;
      sy += y;
      sxx += x * x;
      syy += y * y;
      sxy += x * y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          if (conn === 4 && dx && dy) continue;
          const xx = x + dx,
            yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (m[q] && !labels[q]) {
            labels[q] = next;
            stack[sp++] = q;
          }
        }
    }
    const cx = sx / area,
      cy = sy / area;
    comps.push({
      label: next,
      area,
      minX,
      minY,
      maxX,
      maxY,
      cx,
      cy,
      mu20: sxx / area - cx * cx + 1 / 12,
      mu02: syy / area - cy * cy + 1 / 12,
      mu11: sxy / area - cx * cy,
    });
  }
  return { labels, comps };
}

export function largestComponent(m: Mask, w: number, h: number): Mask {
  const { labels, comps } = labelComponents(m, w, h, 8);
  if (!comps.length) return new Uint8Array(w * h);
  let best = comps[0];
  for (const c of comps) if (c.area > best.area) best = c;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) if (labels[i] === best.label) out[i] = 1;
  return out;
}

export function fillHoles(m: Mask, w: number, h: number): Mask {
  const outside = new Uint8Array(w * h);
  const stack: number[] = [];
  const push = (x: number, y: number) => {
    const i = y * w + x;
    if (!m[i] && !outside[i]) {
      outside[i] = 1;
      stack.push(i);
    }
  };
  for (let x = 0; x < w; x++) {
    push(x, 0);
    push(x, h - 1);
  }
  for (let y = 0; y < h; y++) {
    push(0, y);
    push(w - 1, y);
  }
  while (stack.length) {
    const p = stack.pop()!;
    const x = p % w,
      y = (p / w) | 0;
    if (x > 0) push(x - 1, y);
    if (x < w - 1) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y < h - 1) push(x, y + 1);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = outside[i] ? 0 : 1;
  return out;
}

/** Chamfer (3-4) distance to the nearest zero pixel of `m`; image borders are NOT treated as zeros. */
export function distanceToBackground(m: Mask, w: number, h: number): Float32Array {
  const INF = 1e9;
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = m[i] ? INF : 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + 3);
      if (y > 0) {
        v = Math.min(v, d[i - w] + 3);
        if (x > 0) v = Math.min(v, d[i - w - 1] + 4);
        if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4);
      }
      d[i] = v;
    }
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = d[i];
      if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
      if (y < h - 1) {
        v = Math.min(v, d[i + w] + 3);
        if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4);
        if (x > 0) v = Math.min(v, d[i + w - 1] + 4);
      }
      d[i] = v;
    }
  for (let i = 0; i < d.length; i++) d[i] = d[i] >= INF ? 1e6 : d[i] / 3;
  return d;
}

export function boundaryPixels(m: Mask, w: number, h: number): Mask {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!m[i]) continue;
      if (
        x === 0 ||
        y === 0 ||
        x === w - 1 ||
        y === h - 1 ||
        !m[i - 1] ||
        !m[i + 1] ||
        !m[i - w] ||
        !m[i + w]
      )
        out[i] = 1;
    }
  return out;
}

export function convexHull(points: Array<[number, number]>): Array<[number, number]> {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

export function polygonArea(poly: Array<[number, number]>): number {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

/** Eigen-axes of a component from central moments. */
export function ellipseAxes(c: Pick<Component, "mu20" | "mu02" | "mu11">): {
  major: number;
  minor: number;
  angle: number;
} {
  const common = Math.sqrt(((c.mu20 - c.mu02) / 2) ** 2 + c.mu11 ** 2);
  const l1 = (c.mu20 + c.mu02) / 2 + common;
  const l2 = (c.mu20 + c.mu02) / 2 - common;
  return {
    major: 4 * Math.sqrt(Math.max(l1, 1e-9)),
    minor: 4 * Math.sqrt(Math.max(l2, 1e-9)),
    angle: 0.5 * Math.atan2(2 * c.mu11, c.mu20 - c.mu02),
  };
}

/** Integral-image box mean (clamped at borders). */
export function boxMean(src: Img, w: number, h: number, r: number): Img {
  const W = w + 1;
  const ii = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += src[y * w + x];
      ii[(y + 1) * W + x + 1] = ii[y * W + x + 1] + row;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r),
      y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r),
        x1 = Math.min(w, x + r + 1);
      const s = ii[y1 * W + x1] - ii[y0 * W + x1] - ii[y1 * W + x0] + ii[y0 * W + x0];
      out[y * w + x] = s / ((x1 - x0) * (y1 - y0));
    }
  }
  return out;
}

export function flipMaskX(m: Mask, w: number, h: number): Mask {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = m[y * w + (w - 1 - x)];
  return out;
}
export function flipImgX(m: Img, w: number, h: number): Img {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = m[y * w + (w - 1 - x)];
  return out;
}

/** Deterministic PRNG (mulberry32) so RANSAC is reproducible. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Two-class fuzzy c-means on 1-D intensities. Returns the centres [dark, bright] and bright-membership for every sample. */
export function fcm2(
  values: Float32Array,
  m = 2,
  iters = 40,
  eps = 1e-3,
): { centers: [number, number]; bright: Float32Array } {
  let c0 = percentile(values, 20);
  let c1 = percentile(values, 85);
  const n = values.length;
  const u1 = new Float32Array(n);
  const e = 2 / (m - 1);
  for (let it = 0; it < iters; it++) {
    let n0 = 0,
      d0 = 0,
      n1 = 0,
      d1 = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(values[i] - c0) + 1e-6;
      const b = Math.abs(values[i] - c1) + 1e-6;
      const ub = 1 / (1 + (b / a) ** e); // membership of bright cluster
      u1[i] = ub;
      const ua = 1 - ub;
      const wa = ua ** m,
        wb = ub ** m;
      n0 += wa * values[i];
      d0 += wa;
      n1 += wb * values[i];
      d1 += wb;
    }
    const nc0 = d0 ? n0 / d0 : c0;
    const nc1 = d1 ? n1 / d1 : c1;
    const delta = Math.max(Math.abs(nc0 - c0), Math.abs(nc1 - c1));
    c0 = nc0;
    c1 = nc1;
    if (delta < eps) break;
  }
  return { centers: [c0, c1], bright: u1 };
}
