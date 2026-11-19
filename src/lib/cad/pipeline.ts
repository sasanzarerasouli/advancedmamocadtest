import {
  type Img,
  type Mask,
  binClose,
  binDilate,
  binErode,
  binOpen,
  boundaryPixels,
  boxMean,
  convexHull,
  distanceToBackground,
  ellipseAxes,
  fcm2,
  fillHoles,
  flipImgX,
  flipMaskX,
  gaussian,
  greyDilate,
  greyErode,
  labelComponents,
  largestComponent,
  maskedValues,
  meanStd,
  median,
  otsu,
  percentile,
  polygonArea,
  rng,
  downsample,
  whiteTophat,
} from "./imgops";

/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */

export interface CadParams {
  bgStdMultiplier: number;
  bgMinOffset: number;
  breastBlurSigma: number;
  skinFrac: number;
  nippleRadiusFrac: number;
  pectoralMinContrast: number;
  massSigmaFracs: number[];
  massZ: number;
  massFcmSigma: number;
  massMinContrast: number;
  massMinAreaFrac: number;
  massMaxAreaFrac: number;
  massMinAxisRatio: number;
  massMinSolidity: number;
  calcSigma: number;
  calcRadii: number[];
  calcLocalRadius: number;
  calcZ: number;
  calcTAbs: number;
  calcMinArea: number;
  calcMaxArea: number;
  calcMaxElong: number;
  calcMaxCandidates: number;
  clusterMinPts: number;
  clusterEpsFrac: number;
  /** 0 = unknown. If given, cluster eps = 5 mm / spacing. */
  pixelSpacingMm: number;
}

export const DEFAULT_PARAMS: CadParams = {
  bgStdMultiplier: 5,
  bgMinOffset: 6,
  breastBlurSigma: 1.5,
  skinFrac: 0.012,
  nippleRadiusFrac: 0.05,
  pectoralMinContrast: 5,
  massSigmaFracs: [0.009, 0.013, 0.019, 0.027, 0.039, 0.056, 0.08, 0.115],
  massZ: 4,
  massFcmSigma: 1.5,
  massMinContrast: 18,
  massMinAreaFrac: 0.002,
  massMaxAreaFrac: 0.35,
  massMinAxisRatio: 0.35,
  massMinSolidity: 0.6,
  calcSigma: 0.7,
  calcRadii: [2, 3, 5],
  calcLocalRadius: 24,
  calcZ: 4.5,
  calcTAbs: 6,
  calcMinArea: 2,
  calcMaxArea: 80,
  calcMaxElong: 3.5,
  calcMaxCandidates: 300,
  clusterMinPts: 5,
  clusterEpsFrac: 0.04,
  pixelSpacingMm: 0,
};

/* ------------------------------------------------------------------ */
/* Result types                                                        */
/* ------------------------------------------------------------------ */

export interface MassFeatures {
  areaPx: number;
  equivDiameterPx: number;
  axisRatio: number;
  circularity: number;
  solidity: number;
  meanIntensity: number;
  surroundIntensity: number;
  contrastDiff: number;
  contrastRatio: number;
  heterogeneity: number;
  marginSharpness: number;
  radialRoughness: number;
  internalLucentFraction: number;
  blobZ: number;
  excludedOverlapFraction: number;
}

export interface MassFinding {
  id: string;
  mask: Mask;
  centroid: [number, number];
  bbox: [number, number, number, number];
  confidence: number;
  features: MassFeatures;
  descriptors: { shape: string; margin: string; density: string };
}

export interface RejectedMass {
  center: [number, number];
  radius: number;
  z: number;
  reason: string;
}

export interface CalcCandidate {
  id: string;
  x: number;
  y: number;
  areaPx: number;
  peakTophat: number;
  peakZ: number;
  elongation: number;
  score: number;
  clusterId: string | null;
  insideMass: boolean;
}

export interface CalcCluster {
  id: string;
  hull: Array<[number, number]>;
  memberIds: string[];
  count: number;
  hullAreaPx: number;
  centroid: [number, number];
}

export interface CadResult {
  width: number;
  height: number;
  projection: { view: string; chestWall: "left" | "right"; flipped: boolean };
  masks: {
    breast: Mask;
    pectoral: Mask;
    skin: Mask;
    nipple: Mask;
    marker: Mask;
    excluded: Mask;
  };
  pectoral: { found: boolean; line?: { a: number; b: number }; inliers: number; contrast: number; note: string };
  nipple: { apex: [number, number]; inProfile: boolean; bump: number; wide: number; radiusPx: number };
  markers: Array<{ id: string; centroid: [number, number]; areaPx: number; kind: string }>;
  density: { fibroglandularFraction: number; category: string; meanIntensity: number };
  noise: { backgroundLevel: number; backgroundStd: number; breastThreshold: number };
  masses: MassFinding[];
  rejectedMasses: RejectedMass[];
  calcs: CalcCandidate[];
  clusters: CalcCluster[];
  calcInfo: { thresholdZ: number; rawCount: number; keptCount: number; rejectedByReason: Record<string, number>; epsPx: number };
  timings: Record<string, number>;
  log: string[];
  params: CadParams;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

function union(w: number, h: number, ...ms: Mask[]): Mask {
  const out = new Uint8Array(w * h);
  for (const m of ms) for (let i = 0; i < out.length; i++) if (m[i]) out[i] = 1;
  return out;
}

function disc(w: number, h: number, cx: number, cy: number, r: number): Mask {
  const out = new Uint8Array(w * h);
  const x0 = Math.max(0, Math.floor(cx - r)),
    x1 = Math.min(w - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r)),
    y1 = Math.min(h - 1, Math.ceil(cy + r));
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) out[y * w + x] = 1;
  return out;
}

/* ------------------------------------------------------------------ */
/* Stage 1/2: breast mask, orientation                                 */
/* ------------------------------------------------------------------ */

function segmentBreast(gray: Img, w: number, h: number, P: CadParams) {
  const hist = new Float64Array(256);
  for (let i = 0; i < gray.length; i++) hist[Math.max(0, Math.min(255, Math.round(gray[i])))]++;
  let bg = 0;
  for (let i = 1; i <= 60; i++) if (hist[i] > hist[bg]) bg = i;
  const bgVals: number[] = [];
  for (let i = 0; i < gray.length; i += 3) if (gray[i] <= bg + 10) bgVals.push(gray[i]);
  const { std } = meanStd(bgVals);
  const thr = bg + Math.max(P.bgMinOffset, P.bgStdMultiplier * std);
  const sm = gaussian(gray, w, h, P.breastBlurSigma);
  const raw = new Uint8Array(w * h);
  for (let i = 0; i < raw.length; i++) raw[i] = sm[i] > thr ? 1 : 0;
  let m = binOpen(raw, w, h, 2);
  m = largestComponent(m, w, h);
  m = fillHoles(m, w, h);
  m = binClose(m, w, h, 3);
  m = fillHoles(m, w, h);
  return { mask: m, bg, std, thr };
}

function rowLeftmost(m: Mask, w: number, h: number): Int32Array {
  const xb = new Int32Array(h).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++)
      if (m[y * w + x]) {
        xb[y] = x;
        break;
      }
  }
  return xb;
}

/* ------------------------------------------------------------------ */
/* Pectoral muscle (MLO)                                               */
/* ------------------------------------------------------------------ */

function detectPectoral(N: Img, breast: Mask, xb: Int32Array, w: number, h: number, P: CadParams) {
  const none = (note: string) => ({ found: false, mask: new Uint8Array(w * h) as Mask, inliers: 0, contrast: 0, note, line: undefined as undefined | { a: number; b: number } });
  let touchesTop = false;
  for (let x = 0; x < w; x++) if (breast[x]) touchesTop = true;
  if (!touchesTop) return none("breast does not reach the top image border: no pectoral expected (CC-like)");

  const G = gaussian(N, w, h, 3);
  const win = Math.max(6, Math.round(0.02 * w));
  const yMax = Math.round(0.75 * h);
  const pts: Array<{ x: number; y: number; d: number }> = [];
  for (let y = 0; y < yMax; y += 2) {
    if (xb[y] < 0) continue;
    const xmin = Math.max(win + 1, xb[y] + Math.round(0.06 * w));
    const xmax = w - 1 - win;
    let best = -1e9,
      bx = -1;
    for (let x = xmin; x <= xmax; x++) {
      let r = 0,
        l = 0;
      for (let k = 1; k <= win; k++) {
        r += G[y * w + x + k];
        l += G[y * w + x - k];
      }
      const d = (r - l) / win;
      if (d > best) {
        best = d;
        bx = x;
      }
    }
    if (bx > 0 && best > 0) pts.push({ x: bx, y, d: best });
  }
  if (pts.length < 12) return none("too few edge points");
  const ds = pts.map((p) => p.d).sort((a, b) => a - b);
  const p90 = ds[Math.floor(0.9 * (ds.length - 1))];
  const strong = pts.filter((p) => p.d >= Math.max(4, 0.4 * p90));
  if (strong.length < 12) return none("edge response too weak for a muscle boundary");

  const rand = rng(12345);
  let bestInl: typeof strong = [];
  let bestScore = 0;
  for (let it = 0; it < 300; it++) {
    const p = strong[Math.floor(rand() * strong.length)];
    const q = strong[Math.floor(rand() * strong.length)];
    if (Math.abs(p.y - q.y) < 0.1 * h) continue;
    const a = (q.x - p.x) / (q.y - p.y);
    const b = p.x - a * p.y;
    if (a < 0.1 || a > 2) continue;
    const inl = strong.filter((s) => Math.abs(s.x - (a * s.y + b)) <= 8);
    const score = inl.reduce((s, t) => s + t.d, 0);
    if (score > bestScore) {
      bestScore = score;
      bestInl = inl;
    }
  }
  if (bestInl.length < 12) return none("no consistent oblique line (RANSAC inliers < 12)");
  // least squares refit x = a*y + b
  let sy = 0,
    sx = 0,
    syy = 0,
    sxy = 0;
  const n = bestInl.length;
  for (const p of bestInl) {
    sy += p.y;
    sx += p.x;
    syy += p.y * p.y;
    sxy += p.x * p.y;
  }
  const den = n * syy - sy * sy;
  if (Math.abs(den) < 1e-6) return none("degenerate fit");
  const a = (n * sxy - sy * sx) / den;
  const b = (sx - a * sy) / n;
  const yEnd = (w - 1 - b) / a;
  const ySpan = Math.max(...bestInl.map((p) => p.y)) - Math.min(...bestInl.map((p) => p.y));
  if (a < 0.1 || a > 2 || b < 0.2 * w || yEnd < 0.12 * h || ySpan < 0.15 * h)
    return none(`fitted line implausible for a pectoral edge (a=${a.toFixed(2)}, b=${b.toFixed(0)}, y_end=${yEnd.toFixed(0)})`);

  // contrast validation
  let inSum = 0,
    outSum = 0,
    cnt = 0;
  const yLim = Math.min(h - 1, Math.round(yEnd * 0.9));
  for (let y = 0; y < yLim; y += 4) {
    const xe = Math.round(a * y + b);
    if (xe - 20 < 0 || xe + 20 >= w) continue;
    let si = 0,
      so = 0;
    for (let k = 6; k < 20; k++) {
      si += N[y * w + xe + k];
      so += N[y * w + xe - k];
    }
    inSum += si / 14;
    outSum += so / 14;
    cnt++;
  }
  const contrast = cnt ? (inSum - outSum) / cnt : 0;
  if (contrast < P.pectoralMinContrast) return none(`triangle not brighter than adjacent tissue (Δ=${contrast.toFixed(1)})`);

  const mask = new Uint8Array(w * h);
  const yTop = Math.min(h - 1, Math.ceil(yEnd));
  for (let y = 0; y <= yTop; y++) {
    const xe = Math.max(0, Math.round(a * y + b));
    for (let x = xe; x < w; x++) if (breast[y * w + x]) mask[y * w + x] = 1;
  }
  return { found: true, mask, inliers: bestInl.length, contrast, note: `edge x = ${a.toFixed(3)}·y + ${b.toFixed(1)}; meets chest wall at y≈${yEnd.toFixed(0)}`, line: { a, b } };
}

/* ------------------------------------------------------------------ */
/* Nipple + markers                                                    */
/* ------------------------------------------------------------------ */

function detectNipple(breast: Mask, xb: Int32Array, w: number, h: number, P: CadParams) {
  const ys: number[] = [];
  for (let y = 0; y < h; y++) if (xb[y] >= 0) ys.push(y);
  const y0 = ys[0] ?? 0,
    y1 = ys[ys.length - 1] ?? h - 1;
  const span = y1 - y0;
  const lo = y0 + Math.round(0.12 * span),
    hi = y1 - Math.round(0.1 * span);
  const sm = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0,
      c = 0;
    for (let k = -3; k <= 3; k++) {
      const yy = y + k;
      if (yy >= 0 && yy < h && xb[yy] >= 0) {
        s += xb[yy];
        c++;
      }
    }
    sm[y] = c ? s / c : w;
  }
  let minX = w;
  for (let y = lo; y <= hi; y++) if (sm[y] < minX) minX = sm[y];
  const rows: number[] = [];
  for (let y = lo; y <= hi; y++) if (sm[y] <= minX + 2) rows.push(y);
  const apexY = Math.round(median(rows.length ? rows : [Math.round((lo + hi) / 2)]));
  const apexX = xb[apexY] >= 0 ? xb[apexY] : Math.round(minX);
  const val = (y: number) => (xb[Math.max(0, Math.min(h - 1, y))] >= 0 ? xb[Math.max(0, Math.min(h - 1, y))] : apexX);
  const d = Math.round(0.04 * h),
    D = Math.round(0.12 * h);
  const bump = (val(apexY - d) + val(apexY + d)) / 2 - apexX;
  const wide = (val(apexY - D) + val(apexY + D)) / 2 - apexX;
  const inProfile = bump > Math.max(0.008 * w, 0.3 * wide);
  const breastW = w - 1 - apexX;
  const radius = Math.max(8, P.nippleRadiusFrac * breastW) * (inProfile ? 1.3 : 1);
  const m = disc(w, h, apexX, apexY, radius);
  for (let i = 0; i < m.length; i++) m[i] = m[i] && breast[i] ? 1 : 0;
  return { mask: m, apex: [apexX, apexY] as [number, number], inProfile, bump, wide, radius };
}

function detectMarkers(N: Img, breast: Mask, dist: Float32Array, apex: [number, number], nipR: number, skinPx: number, w: number, h: number) {
  const vals = maskedValues(N, breast);
  const hi = percentile(vals, 99.8);
  const cand = new Uint8Array(w * h);
  for (let i = 0; i < cand.length; i++) cand[i] = breast[i] && N[i] >= Math.max(250, hi - 1) ? 1 : 0;
  const { comps } = labelComponents(cand, w, h, 8);
  const mask = new Uint8Array(w * h);
  const list: Array<{ id: string; centroid: [number, number]; areaPx: number; kind: string }> = [];
  let k = 0;
  for (const c of comps) {
    const bw = c.maxX - c.minX + 1,
      bh = c.maxY - c.minY + 1;
    const fill = c.area / (bw * bh);
    const ax = ellipseAxes(c);
    const elong = ax.major / Math.max(ax.minor, 1e-6);
    const d = dist[Math.round(c.cy) * w + Math.round(c.cx)];
    if (c.area < 3 || c.area > 700 || fill < 0.5 || elong > 3 || d > 4 * skinPx) continue;
    k++;
    const nearNipple = Math.hypot(c.cx - apex[0], c.cy - apex[1]) < 2 * nipR;
    list.push({ id: `K${k}`, centroid: [c.cx, c.cy], areaPx: c.area, kind: nearNipple ? "nipple-marker candidate" : "skin-marker / skin-artifact candidate" });
    const dm = disc(w, h, c.cx, c.cy, Math.max(bw, bh) / 2 + 3);
    for (let i = 0; i < mask.length; i++) if (dm[i]) mask[i] = 1;
  }
  return { mask, list };
}

/* ------------------------------------------------------------------ */
/* Mass candidates                                                     */
/* ------------------------------------------------------------------ */

interface MassCand {
  x: number;
  y: number;
  r: number;
  z: number;
}

function massCandidates(N: Img, eligible: Mask, w: number, h: number, P: CadParams): MassCand[] {
  const maxDim = Math.max(w, h);
  const f = Math.max(1, Math.round(maxDim / 200));
  const NM = new Float32Array(w * h);
  for (let i = 0; i < NM.length; i++) NM[i] = eligible[i] ? N[i] : 0;
  const dn = downsample(NM, w, h, f);
  const de = downsample(Float32Array.from(eligible), w, h, f);
  const lw = dn.w,
    lh = dn.h;
  const elig = new Uint8Array(lw * lh);
  for (let i = 0; i < elig.length; i++) elig[i] = de.img[i] > 0.6 ? 1 : 0;
  const globalMed = median(maskedValues(N, eligible));
  const sigmas = P.massSigmaFracs.map((fr) => Math.max(1.2, (fr * maxDim) / f));

  const smooth = (s: number): Img => {
    const A = gaussian(dn.img, lw, lh, s);
    const B = gaussian(de.img, lw, lh, s);
    const out = new Float32Array(lw * lh);
    for (let i = 0; i < out.length; i++) out[i] = B[i] > 0.05 ? A[i] / B[i] : globalMed;
    return out;
  };
  const levels = sigmas.map(smooth);
  const levels16 = sigmas.map((s) => smooth(s * 1.6));
  const D: Img[] = levels.map((L, i) => {
    const o = new Float32Array(lw * lh);
    for (let j = 0; j < o.length; j++) o[j] = L[j] - levels16[i][j];
    return o;
  });
  // z-score per scale (median / MAD over eligible low-res pixels)
  const Z: Img[] = D.map((d) => {
    const v = maskedValues(d, elig);
    const med = median(v);
    const dev = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) dev[i] = Math.abs(v[i] - med);
    const mad = Math.max(median(dev) * 1.4826, 0.25);
    const z = new Float32Array(lw * lh);
    for (let i = 0; i < z.length; i++) z[i] = (d[i] - med) / mad;
    return z;
  });
  const cands: MassCand[] = [];
  for (let s = 0; s < Z.length; s++) {
    for (let y = 1; y < lh - 1; y++)
      for (let x = 1; x < lw - 1; x++) {
        const i = y * lw + x;
        const v = Z[s][i];
        if (v < P.massZ || !elig[i]) continue;
        let isMax = true;
        for (let ds = -1; ds <= 1 && isMax; ds++) {
          const ss = s + ds;
          if (ss < 0 || ss >= Z.length) continue;
          for (let dy = -1; dy <= 1 && isMax; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              if (!ds && !dy && !dx) continue;
              if (Z[ss][i + dy * lw + dx] > v) {
                isMax = false;
                break;
              }
            }
        }
        if (!isMax) continue;
        const rLow = sigmas[s] * Math.SQRT2;
        // fraction of the candidate disc that is eligible tissue
        let tot = 0,
          ok = 0;
        const rr = Math.ceil(rLow);
        for (let dy = -rr; dy <= rr; dy++)
          for (let dx = -rr; dx <= rr; dx++) {
            if (dx * dx + dy * dy > rLow * rLow) continue;
            tot++;
            const xx = x + dx,
              yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < lw && yy < lh && elig[yy * lw + xx]) ok++;
          }
        if (tot && ok / tot < 0.7) continue;
        cands.push({ x: x * f + f / 2, y: y * f + f / 2, r: rLow * f, z: v });
      }
  }
  cands.sort((a, b) => b.z - a.z);
  const kept: MassCand[] = [];
  for (const c of cands) {
    if (kept.some((k) => Math.hypot(k.x - c.x, k.y - c.y) < 0.7 * Math.max(k.r, c.r))) continue;
    kept.push(c);
    if (kept.length >= 6) break;
  }
  return kept;
}

function segmentMass(Ns: Img, N: Img, eligible: Mask, breast: Mask, w: number, h: number, c: MassCand, P: CadParams) {
  const half = Math.ceil(2.2 * c.r);
  const x0 = Math.max(0, Math.round(c.x) - half),
    y0 = Math.max(0, Math.round(c.y) - half);
  const x1 = Math.min(w - 1, Math.round(c.x) + half),
    y1 = Math.min(h - 1, Math.round(c.y) + half);
  const ww = x1 - x0 + 1,
    wh = y1 - y0 + 1;
  const cx = Math.round(c.x) - x0,
    cy = Math.round(c.y) - y0;
  const loc = new Float32Array(ww * wh);
  const brl = new Uint8Array(ww * wh);
  const el = new Uint8Array(ww * wh);
  for (let y = 0; y < wh; y++)
    for (let x = 0; x < ww; x++) {
      const gi = (y + y0) * w + x + x0;
      loc[y * ww + x] = Ns[gi];
      brl[y * ww + x] = breast[gi];
      el[y * ww + x] = eligible[gi];
    }
  const samples: number[] = [];
  const stride = Math.max(1, Math.floor((ww * wh) / 20000));
  for (let i = 0; i < loc.length; i += stride) if (el[i]) samples.push(loc[i]);
  if (samples.length < 200) return null;
  const { centers } = fcm2(Float32Array.from(samples), 2, 40, 1e-3);
  if (centers[1] - centers[0] < 8) return null;
  const m0 = new Uint8Array(ww * wh);
  const dsc = disc(ww, wh, cx, cy, 1.7 * c.r);
  for (let i = 0; i < m0.length; i++) {
    const a = Math.abs(loc[i] - centers[0]) + 1e-6;
    const b = Math.abs(loc[i] - centers[1]) + 1e-6;
    const u = 1 / (1 + (b / a) ** 2);
    m0[i] = u > 0.5 && brl[i] && dsc[i] ? 1 : 0;
  }
  const ro = Math.max(2, Math.round(0.06 * c.r)),
    rc = Math.max(2, Math.round(0.1 * c.r));
  let m1 = binClose(binOpen(m0, ww, wh, ro), ww, wh, rc);
  const preFill = m1.slice();
  m1 = fillHoles(m1, ww, wh);
  const { labels, comps } = labelComponents(m1, ww, wh, 8);
  if (!comps.length) return null;
  let pick = labels[cy * ww + cx];
  if (!pick) {
    const inner = disc(ww, wh, cx, cy, 0.5 * c.r);
    const ov = new Map<number, number>();
    for (let i = 0; i < labels.length; i++) if (labels[i] && inner[i]) ov.set(labels[i], (ov.get(labels[i]) ?? 0) + 1);
    let bestL = 0,
      bestV = 0;
    for (const [l, v] of ov) if (v > bestV) (bestV = v), (bestL = l);
    if (!bestL) return null;
    pick = bestL;
  }
  let m2 = new Uint8Array(ww * wh);
  for (let i = 0; i < m2.length; i++) m2[i] = labels[i] === pick ? 1 : 0;
  // curvature-flow surrogate: Gaussian smoothing of the indicator, threshold 0.5
  const f = new Float32Array(ww * wh);
  for (let i = 0; i < f.length; i++) f[i] = m2[i];
  const sf = gaussian(f, ww, wh, Math.max(1.5, 0.04 * c.r));
  const compArea = m2.reduce((s, v) => s + v, 0);
  for (let i = 0; i < m2.length; i++) m2[i] = sf[i] > 0.5 ? 1 : 0;
  const withElig = new Uint8Array(ww * wh);
  for (let i = 0; i < withElig.length; i++) withElig[i] = m2[i] && el[i] ? 1 : 0;
  const finalMask = largestComponent(withElig, ww, wh);
  const area = finalMask.reduce((s, v) => s + v, 0);
  if (area < 20) return null;
  const excludedFrac = compArea ? Math.max(0, 1 - area / Math.max(compArea, 1)) : 0;
  let holes = 0;
  for (let i = 0; i < m2.length; i++) if (m2[i] && !preFill[i]) holes++;
  const full = new Uint8Array(w * h);
  for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) if (finalMask[y * ww + x]) full[(y + y0) * w + x + x0] = 1;
  return { full, excludedFrac, holeFrac: holes / Math.max(1, area), centers, win: { x0, y0, x1, y1 } };
}

function massFeatures(mask: Mask, Ns: Img, eligible: Mask, w: number, h: number, win: { x0: number; y0: number; x1: number; y1: number }, z: number, excludedFrac: number, holeFrac: number): { f: MassFeatures; centroid: [number, number]; bbox: [number, number, number, number] } {
  const x0 = Math.max(0, win.x0 - 4),
    y0 = Math.max(0, win.y0 - 4);
  const ww = Math.min(w, win.x1 + 5) - x0,
    wh = Math.min(h, win.y1 + 5) - y0;
  const m = new Uint8Array(ww * wh),
    el = new Uint8Array(ww * wh),
    L = new Float32Array(ww * wh);
  for (let y = 0; y < wh; y++)
    for (let x = 0; x < ww; x++) {
      const gi = (y + y0) * w + x + x0;
      m[y * ww + x] = mask[gi];
      el[y * ww + x] = eligible[gi];
      L[y * ww + x] = Ns[gi];
    }
  const { comps } = labelComponents(m, ww, wh, 8);
  const c = comps[0];
  const ax = ellipseAxes(c);
  const bd = boundaryPixels(m, ww, wh);
  const pts: Array<[number, number]> = [];
  let edges = 0;
  for (let y = 0; y < wh; y++)
    for (let x = 0; x < ww; x++) {
      const i = y * ww + x;
      if (!m[i]) continue;
      if (x === 0 || !m[i - 1]) edges++;
      if (x === ww - 1 || !m[i + 1]) edges++;
      if (y === 0 || !m[i - ww]) edges++;
      if (y === wh - 1 || !m[i + ww]) edges++;
      if (bd[i]) pts.push([x, y]);
    }
  const perim = (edges * Math.PI) / 4;
  const circ = Math.min(1, (4 * Math.PI * c.area) / (perim * perim));
  const hull = convexHull(pts);
  let hullPer = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i],
      b = hull[(i + 1) % hull.length];
    hullPer += Math.hypot(a[0] - b[0], a[1] - b[1]);
  }
  const hullArea = polygonArea(hull) + hullPer / 2 + 1;
  const solidity = Math.min(1, c.area / hullArea);
  const mv = maskedValues(L, m);
  const { mean, std } = meanStd(mv);
  const ringW = Math.max(4, Math.round(0.12 * Math.sqrt(c.area / Math.PI)));
  const inner = binDilate(m, ww, wh, 2);
  const outer = binDilate(m, ww, wh, ringW + 2);
  const ring = new Uint8Array(ww * wh);
  for (let i = 0; i < ring.length; i++) ring[i] = outer[i] && !inner[i] && el[i] ? 1 : 0;
  const rv = maskedValues(L, ring);
  const surround = rv.length ? meanStd(rv).mean : mean;
  // margin sharpness: mean gradient magnitude on the boundary / expected step-edge gradient
  let gsum = 0,
    gn = 0;
  const sig = 1.5;
  for (const [x, y] of pts) {
    if (x < 1 || y < 1 || x >= ww - 1 || y >= wh - 1) continue;
    const gx = (L[y * ww + x + 1] - L[y * ww + x - 1]) / 2;
    const gy = (L[(y + 1) * ww + x] - L[(y - 1) * ww + x]) / 2;
    gsum += Math.hypot(gx, gy);
    gn++;
  }
  const dN = Math.max(mean - surround, 1);
  const sharp = gn ? gsum / gn / ((dN / (sig * Math.sqrt(2 * Math.PI))) * 1.0) : 0;
  // radial roughness
  const bins = 36;
  const rs: number[][] = Array.from({ length: bins }, () => []);
  for (const [x, y] of pts) {
    const ang = Math.atan2(y - c.cy, x - c.cx);
    const b = Math.min(bins - 1, Math.floor(((ang + Math.PI) / (2 * Math.PI)) * bins));
    rs[b].push(Math.hypot(x - c.cx, y - c.cy));
  }
  const rm = rs.map((a) => (a.length ? median(a) : NaN));
  const valid = rm.filter((v) => !Number.isNaN(v));
  const mr = valid.reduce((s, v) => s + v, 0) / Math.max(1, valid.length);
  let dev = 0,
    dn = 0;
  for (let b = 0; b < bins; b++) {
    if (Number.isNaN(rm[b])) continue;
    const nb: number[] = [];
    for (let k = -2; k <= 2; k++) {
      const v = rm[(b + k + bins) % bins];
      if (!Number.isNaN(v)) nb.push(v);
    }
    const sm = nb.reduce((s, v) => s + v, 0) / nb.length;
    dev += (rm[b] - sm) ** 2;
    dn++;
  }
  const rough = dn ? Math.sqrt(dev / dn) / Math.max(mr, 1e-6) : 0;
  return {
    f: {
      areaPx: c.area,
      equivDiameterPx: 2 * Math.sqrt(c.area / Math.PI),
      axisRatio: ax.minor / ax.major,
      circularity: circ,
      solidity,
      meanIntensity: mean,
      surroundIntensity: surround,
      contrastDiff: mean - surround,
      contrastRatio: mean / Math.max(surround, 1),
      heterogeneity: std / Math.max(mean, 1),
      marginSharpness: sharp,
      radialRoughness: rough,
      internalLucentFraction: holeFrac,
      blobZ: z,
      excludedOverlapFraction: excludedFrac,
    },
    centroid: [c.cx + x0, c.cy + y0],
    bbox: [c.minX + x0, c.minY + y0, c.maxX + x0, c.maxY + y0],
  };
}

function describeMass(f: MassFeatures, refHigh: number) {
  const shape =
    f.solidity < 0.85 || f.circularity < 0.6
      ? "irregular"
      : f.axisRatio >= 0.78 && f.circularity > 0.85
        ? "round"
        : f.axisRatio >= 0.5
          ? "oval"
          : "elongated";
  const margin =
    f.marginSharpness < 0.3
      ? "indistinct / ill-defined (edge blur)"
      : f.radialRoughness > 0.12 && f.marginSharpness < 0.5
        ? "possibly spiculated / irregular (needs radiologist review)"
        : f.radialRoughness > 0.06
          ? "possibly lobulated / microlobulated"
          : f.marginSharpness >= 0.45
            ? "circumscribed-like (sharp, smooth edge)"
            : "partly obscured";
  const density = f.meanIntensity > refHigh ? "higher than the fibroglandular reference (p90)" : f.meanIntensity > 0.8 * refHigh ? "comparable to dense glandular tissue" : "lower than dense glandular tissue";
  return { shape, margin, density };
}

/* ------------------------------------------------------------------ */
/* Calcifications                                                      */
/* ------------------------------------------------------------------ */

function dbscan(pts: Array<[number, number]>, eps: number, minPts: number): number[] {
  const n = pts.length;
  const lab = new Array<number>(n).fill(-2); // -2 unvisited, -1 noise
  const nb = (i: number) => {
    const r: number[] = [];
    for (let j = 0; j < n; j++) if (Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= eps) r.push(j);
    return r;
  };
  let cid = 0;
  for (let i = 0; i < n; i++) {
    if (lab[i] !== -2) continue;
    const ns = nb(i);
    if (ns.length < minPts) {
      lab[i] = -1;
      continue;
    }
    lab[i] = cid;
    const q = ns.slice();
    while (q.length) {
      const j = q.pop()!;
      if (lab[j] === -1) lab[j] = cid;
      if (lab[j] !== -2) continue;
      lab[j] = cid;
      const nj = nb(j);
      if (nj.length >= minPts) for (const k of nj) q.push(k);
    }
    cid++;
  }
  return lab;
}

function detectCalcs(N: Img, eligible: Mask, massUnion: Mask, w: number, h: number, P: CadParams) {
  const S = gaussian(N, w, h, P.calcSigma);
  const T = new Float32Array(w * h);
  for (const r of P.calcRadii) {
    const t = whiteTophat(S, w, h, r);
    for (let i = 0; i < T.length; i++) if (t[i] > T[i]) T[i] = t[i];
  }
  const T2 = new Float32Array(w * h);
  for (let i = 0; i < T2.length; i++) T2[i] = T[i] * T[i];
  const mu = boxMean(T, w, h, P.calcLocalRadius);
  const mu2 = boxMean(T2, w, h, P.calcLocalRadius);
  const Zm = new Float32Array(w * h);
  for (let i = 0; i < Zm.length; i++) Zm[i] = (T[i] - mu[i]) / (Math.sqrt(Math.max(mu2[i] - mu[i] * mu[i], 0)) + 1.0);
  const border = 3;
  const elig = eligible.slice();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (x < border || y < border || x >= w - border || y >= h - border) elig[y * w + x] = 0;

  const maxDim = Math.max(w, h);
  let kz = P.calcZ;
  let comps: ReturnType<typeof labelComponents>["comps"] = [];
  let labels: Int32Array = new Int32Array(0);
  let raw = 0;
  for (let attempt = 0; attempt < 8; attempt++) {
    const cand = new Uint8Array(w * h);
    for (let i = 0; i < cand.length; i++) cand[i] = elig[i] && Zm[i] >= 0.5 * kz && T[i] >= 0.5 * P.calcTAbs ? 1 : 0; // hysteresis: low mask, seeds checked below
    const lc = labelComponents(cand, w, h, 8);
    comps = lc.comps;
    labels = lc.labels;
    const pk = new Float32Array(comps.length + 1);
    for (let i = 0; i < labels.length; i++) if (labels[i] && Zm[i] > pk[labels[i]]) pk[labels[i]] = Zm[i];
    const plausible = comps.filter((c) => pk[c.label] >= kz && c.area >= P.calcMinArea && c.area <= P.calcMaxArea).length;
    raw = comps.length;
    if (plausible <= P.calcMaxCandidates) break;
    kz += 0.5;
  }
  const reasons: Record<string, number> = { "too small (<min area)": 0, "too large (>max area: mass/vessel/tissue)": 0, "elongated (vessel / trabecula / skin line)": 0, "low peak local-contrast": 0 };
  const cands: CalcCandidate[] = [];
  let k = 0;
  // per-component peak statistics
  const peakT = new Float32Array(comps.length + 1);
  const peakZ = new Float32Array(comps.length + 1);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i];
    if (!l) continue;
    if (T[i] > peakT[l]) peakT[l] = T[i];
    if (Zm[i] > peakZ[l]) peakZ[l] = Zm[i];
  }
  for (const c of comps) {
    if (c.area < P.calcMinArea) {
      reasons["too small (<min area)"]++;
      continue;
    }
    if (c.area > P.calcMaxArea) {
      reasons["too large (>max area: mass/vessel/tissue)"]++;
      continue;
    }
    const ax = ellipseAxes(c);
    const elong = ax.major / Math.max(ax.minor, 1e-6);
    if (c.area >= 4 && elong > P.calcMaxElong) {
      reasons["elongated (vessel / trabecula / skin line)"]++;
      continue;
    }
    if (peakZ[c.label] < kz || peakT[c.label] < P.calcTAbs) {
      reasons["low peak local-contrast"]++;
      continue;
    }
    k++;
    const cxr = Math.round(c.cx),
      cyr = Math.round(c.cy);
    cands.push({
      id: `c${k}`,
      x: c.cx,
      y: c.cy,
      areaPx: c.area,
      peakTophat: peakT[c.label],
      peakZ: peakZ[c.label],
      elongation: elong,
      score: Math.min(1, peakZ[c.label] / (2 * kz)) * (elong <= 2 ? 1 : 0.7),
      clusterId: null,
      insideMass: !!massUnion[cyr * w + cxr],
    });
  }
  const eps = P.pixelSpacingMm > 0 ? 5 / P.pixelSpacingMm : P.clusterEpsFrac * maxDim;
  const clusters: CalcCluster[] = [];
  if (cands.length >= P.clusterMinPts) {
    const lab = dbscan(cands.map((c) => [c.x, c.y] as [number, number]), eps, P.clusterMinPts);
    const groups = new Map<number, number[]>();
    lab.forEach((l, i) => {
      if (l >= 0) groups.set(l, [...(groups.get(l) ?? []), i]);
    });
    let ci = 0;
    for (const [, idxs] of groups) {
      if (idxs.length < P.clusterMinPts) continue;
      ci++;
      const id = `CL${ci}`;
      const pts = idxs.map((i) => [cands[i].x, cands[i].y] as [number, number]);
      const hull = convexHull(pts);
      idxs.forEach((i) => (cands[i].clusterId = id));
      clusters.push({
        id,
        hull,
        memberIds: idxs.map((i) => cands[i].id),
        count: idxs.length,
        hullAreaPx: polygonArea(hull),
        centroid: [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length],
      });
    }
  }
  return { cands, clusters, info: { thresholdZ: kz, rawCount: raw, keptCount: cands.length, rejectedByReason: reasons, epsPx: eps } };
}

/* ------------------------------------------------------------------ */
/* Orchestrator                                                        */
/* ------------------------------------------------------------------ */

export function runPipeline(grayIn: Img, w: number, h: number, user: Partial<CadParams> = {}): CadResult {
  const P: CadParams = { ...DEFAULT_PARAMS, ...user };
  const timings: Record<string, number> = {};
  const log: string[] = [];
  const tick = (name: string, t0: number) => (timings[name] = Math.round(now() - t0));

  // ---- breast + orientation
  let t0 = now();
  let { mask: breast, bg, std, thr } = segmentBreast(grayIn, w, h, P);
  let cntL = 0,
    cntR = 0;
  for (let y = 0; y < h; y++) {
    if (breast[y * w]) cntL++;
    if (breast[y * w + w - 1]) cntR++;
  }
  const chestRight = cntR >= cntL;
  const flipped = !chestRight;
  const gray = flipped ? flipImgX(grayIn, w, h) : grayIn;
  if (flipped) breast = flipMaskX(breast, w, h);
  log.push(`Background level ${bg}, σ=${std.toFixed(2)} → breast threshold ${thr.toFixed(1)}; chest wall on image ${chestRight ? "right" : "left"} (${flipped ? "image mirrored internally" : "no mirroring"}).`);
  tick("breast_mask", t0);

  // ---- normalisation
  t0 = now();
  const bv = maskedValues(gray, breast);
  const lo = percentile(bv, 1),
    hi = percentile(bv, 99.8);
  const N = new Float32Array(w * h);
  for (let i = 0; i < N.length; i++) N[i] = breast[i] ? Math.max(0, Math.min(255, ((gray[i] - lo) / Math.max(hi - lo, 1)) * 255)) : 0;
  tick("normalise", t0);

  // ---- anatomy
  t0 = now();
  const xb = rowLeftmost(breast, w, h);
  const pect = detectPectoral(N, breast, xb, w, h, P);
  log.push(`Pectoral: ${pect.found ? "detected" : "not detected"} – ${pect.note}`);
  const dist = distanceToBackground(breast, w, h);
  const skinPx = Math.max(3, Math.round(P.skinFrac * w));
  const skin = new Uint8Array(w * h);
  for (let i = 0; i < skin.length; i++) skin[i] = breast[i] && dist[i] <= skinPx ? 1 : 0;
  const nip = detectNipple(breast, xb, w, h, P);
  log.push(`Nipple apex at (${nip.apex[0]}, ${nip.apex[1]}); profile ${nip.inProfile ? "visible" : "NOT clearly visible"} (bump ${nip.bump.toFixed(1)} px vs wide ${nip.wide.toFixed(1)} px). Exclusion disc radius ${nip.radius.toFixed(0)} px applied as a precaution.`);
  const mk = detectMarkers(N, breast, dist, nip.apex, nip.radius, skinPx, w, h);
  log.push(mk.list.length ? `${mk.list.length} compact saturated object(s) near the skin boundary flagged as marker/artifact candidates.` : "No marker-like saturated compact object near the skin boundary.");
  const pectDil = binDilate(pect.mask, w, h, 3);
  const excluded = union(w, h, pectDil, skin, nip.mask, mk.mask);
  const eligible = new Uint8Array(w * h);
  for (let i = 0; i < eligible.length; i++) eligible[i] = breast[i] && !excluded[i] ? 1 : 0;
  tick("anatomy", t0);

  // ---- masses
  t0 = now();
  const Ns = gaussian(N, w, h, P.massFcmSigma);
  const cands = massCandidates(N, eligible, w, h, P);
  const breastArea = breast.reduce((s, v) => s + v, 0);
  const refHigh = percentile(maskedValues(N, eligible), 90);
  const masses: MassFinding[] = [];
  const rejected: RejectedMass[] = [];
  for (const c of cands) {
    const reject = (reason: string) => rejected.push({ center: [c.x, c.y], radius: c.r, z: c.z, reason });
    if (masses.some((m) => m.mask[Math.round(c.y) * w + Math.round(c.x)])) {
      reject("inside an already segmented mass");
      continue;
    }
    const seg = segmentMass(Ns, N, eligible, breast, w, h, c, P);
    if (!seg) {
      reject("FCM produced no separable bright class / no coherent region");
      continue;
    }
    const mf = massFeatures(seg.full, Ns, eligible, w, h, seg.win, c.z, seg.excludedFrac, seg.holeFrac);
    const f = mf.f;
    if (f.areaPx < P.massMinAreaFrac * breastArea) reject(`area ${f.areaPx} px below ${(P.massMinAreaFrac * 100).toFixed(1)}% of breast`);
    else if (f.areaPx > P.massMaxAreaFrac * breastArea) reject("area too large – likely a parenchymal/density gradient, not a focal mass");
    else if (f.axisRatio < P.massMinAxisRatio) reject(`elongated (minor/major ${f.axisRatio.toFixed(2)}) – vessel/duct/fold-like`);
    else if (f.solidity < P.massMinSolidity) reject(`low solidity ${f.solidity.toFixed(2)} – tissue pattern rather than focal mass`);
    else if (f.contrastDiff < P.massMinContrast) reject(`contrast ΔN=${f.contrastDiff.toFixed(1)} < ${P.massMinContrast} – normal glandular tissue`);
    else if (f.excludedOverlapFraction > 0.4) reject(`${(f.excludedOverlapFraction * 100).toFixed(0)}% of region lies in excluded anatomy (pectoral/skin/nipple/marker)`);
    else {
      const conf = Math.cbrt(Math.min(1, c.z / 8) * Math.min(1, f.contrastDiff / 40) * Math.min(1, f.axisRatio / 0.6));
      masses.push({
        id: `M${masses.length + 1}`,
        mask: seg.full,
        centroid: mf.centroid,
        bbox: mf.bbox,
        confidence: conf,
        features: f,
        descriptors: describeMass(f, refHigh),
      });
    }
  }
  tick("mass", t0);

  // ---- calcifications
  t0 = now();
  const massUnion = union(w, h, ...masses.map((m) => m.mask));
  const cal = detectCalcs(N, eligible, massUnion, w, h, P);
  tick("calcification", t0);

  // ---- density (excluding masses)
  const denseMask = new Uint8Array(w * h);
  const massDil = binDilate(massUnion, w, h, 4);
  for (let i = 0; i < denseMask.length; i++) denseMask[i] = eligible[i] && !massDil[i] ? 1 : 0;
  const dv = maskedValues(N, denseMask);
  const ot = otsu(dv);
  let above = 0;
  for (let i = 0; i < dv.length; i++) if (dv[i] > ot) above++;
  const frac = dv.length ? above / dv.length : 0;
  const category = frac < 0.25 ? "a (fatty)" : frac < 0.5 ? "b (scattered)" : frac < 0.75 ? "c (heterogeneously dense)" : "d (extremely dense)";

  // ---- map back to original orientation
  const W = w;
  const fx = (x: number) => (flipped ? W - 1 - x : x);
  const mm = (m: Mask) => (flipped ? flipMaskX(m, w, h) : m);
  const result: CadResult = {
    width: w,
    height: h,
    projection: { view: pect.found ? "MLO (pectoral muscle found)" : "CC or undetermined (no pectoral edge validated)", chestWall: chestRight ? "right" : "left", flipped },
    masks: { breast: mm(breast), pectoral: mm(pect.mask), skin: mm(skin), nipple: mm(nip.mask), marker: mm(mk.mask), excluded: mm(excluded) },
    pectoral: { found: pect.found, line: pect.line, inliers: pect.inliers, contrast: pect.contrast, note: pect.note },
    nipple: { apex: [fx(nip.apex[0]), nip.apex[1]], inProfile: nip.inProfile, bump: nip.bump, wide: nip.wide, radiusPx: nip.radius },
    markers: mk.list.map((m) => ({ ...m, centroid: [fx(m.centroid[0]), m.centroid[1]] as [number, number] })),
    density: { fibroglandularFraction: frac, category, meanIntensity: meanStd(dv).mean },
    noise: { backgroundLevel: bg, backgroundStd: std, breastThreshold: thr },
    masses: masses.map((m) => ({
      ...m,
      mask: mm(m.mask),
      centroid: [fx(m.centroid[0]), m.centroid[1]] as [number, number],
      bbox: flipped ? ([W - 1 - m.bbox[2], m.bbox[1], W - 1 - m.bbox[0], m.bbox[3]] as [number, number, number, number]) : m.bbox,
    })),
    rejectedMasses: rejected.map((r) => ({ ...r, center: [fx(r.center[0]), r.center[1]] as [number, number] })),
    calcs: cal.cands.map((c) => ({ ...c, x: fx(c.x) })),
    clusters: cal.clusters.map((c) => ({
      ...c,
      hull: c.hull.map(([x, y]) => [fx(x), y] as [number, number]),
      centroid: [fx(c.centroid[0]), c.centroid[1]] as [number, number],
    })),
    calcInfo: cal.info,
    timings,
    log,
    params: P,
  };
  if (!masses.length) log.push("No reliable mass candidate passed all gates (see rejected candidates).");
  if (!cal.cands.length) log.push("No calcification candidate passed size/shape/contrast gates.");
  else if (!cal.clusters.length) log.push(`${cal.cands.length} isolated calcification-like candidate(s); none forms a cluster (≥${P.clusterMinPts} within ε=${cal.info.epsPx.toFixed(0)} px).`);
  return result;
}

/** Re-export for tests. */
export { greyDilate, greyErode, binErode };
