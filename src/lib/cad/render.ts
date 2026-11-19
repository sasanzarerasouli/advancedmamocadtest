import { boundaryPixels, type Mask } from "./imgops";
import type { CadResult } from "./pipeline";

export interface Layers {
  anatomy: boolean;
  masses: boolean;
  calcs: boolean;
  clusters: boolean;
  labels: boolean;
  rejected: boolean;
}

export type ViewKey = "original" | "anatomy" | "masses" | "calcs" | "combined";

export const VIEW_PRESETS: Record<Exclude<ViewKey, "combined">, Layers> = {
  original: { anatomy: false, masses: false, calcs: false, clusters: false, labels: false, rejected: false },
  anatomy: { anatomy: true, masses: false, calcs: false, clusters: false, labels: true, rejected: false },
  masses: { anatomy: false, masses: true, calcs: false, clusters: false, labels: true, rejected: true },
  calcs: { anatomy: false, masses: false, calcs: true, clusters: true, labels: true, rejected: false },
};

type RGB = [number, number, number];

function blend(data: Uint8ClampedArray, m: Mask, c: RGB, a: number) {
  for (let i = 0; i < m.length; i++) {
    if (!m[i]) continue;
    const p = i * 4;
    data[p] = data[p] * (1 - a) + c[0] * a;
    data[p + 1] = data[p + 1] * (1 - a) + c[1] * a;
    data[p + 2] = data[p + 2] * (1 - a) + c[2] * a;
  }
}

function stroke(data: Uint8ClampedArray, m: Mask, w: number, h: number, c: RGB) {
  const b = boundaryPixels(m, w, h);
  for (let i = 0; i < b.length; i++) {
    if (!b[i]) continue;
    const p = i * 4;
    data[p] = c[0];
    data[p + 1] = c[1];
    data[p + 2] = c[2];
  }
}

const COL = {
  breast: [60, 220, 120] as RGB,
  pectoral: [60, 120, 255] as RGB,
  skin: [250, 210, 40] as RGB,
  nipple: [240, 70, 220] as RGB,
  marker: [255, 140, 20] as RGB,
  mass: [255, 50, 60] as RGB,
  calc: [40, 230, 240] as RGB,
  cluster: [255, 240, 60] as RGB,
};

export function renderView(canvas: HTMLCanvasElement, base: ImageData, res: CadResult | null, L: Layers) {
  const { width: w, height: h } = base;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = new ImageData(new Uint8ClampedArray(base.data), w, h);
  const d = img.data;
  if (res) {
    if (L.anatomy) {
      blend(d, res.masks.skin, COL.skin, 0.3);
      blend(d, res.masks.pectoral, COL.pectoral, 0.3);
      blend(d, res.masks.nipple, COL.nipple, 0.4);
      blend(d, res.masks.marker, COL.marker, 0.6);
      stroke(d, res.masks.breast, w, h, COL.breast);
      stroke(d, res.masks.pectoral, w, h, COL.pectoral);
      stroke(d, res.masks.nipple, w, h, COL.nipple);
    }
    if (L.masses) {
      for (const m of res.masses) blend(d, m.mask, COL.mass, 0.28);
      for (const m of res.masses) stroke(d, m.mask, w, h, COL.mass);
    }
  }
  ctx.putImageData(img, 0, 0);
  if (!res) return;

  const s = Math.max(1, Math.max(w, h) / 700);
  const fs = Math.round(13 * s);
  ctx.font = `600 ${fs}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  const label = (txt: string, x: number, y: number, c: RGB) => {
    const tw = ctx.measureText(txt).width;
    const px = Math.max(2, Math.min(w - tw - 6, x));
    const py = Math.max(2, Math.min(h - fs - 6, y));
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(px - 2, py - 1, tw + 6, fs + 4);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillText(txt, px + 1, py + 1);
  };

  if (L.rejected) {
    ctx.setLineDash([4 * s, 4 * s]);
    ctx.lineWidth = Math.max(1, s);
    ctx.strokeStyle = "rgba(200,200,200,0.8)";
    res.rejectedMasses.forEach((r, i) => {
      if (r.reason.startsWith("inside")) return;
      ctx.beginPath();
      ctx.arc(r.center[0], r.center[1], r.radius, 0, Math.PI * 2);
      ctx.stroke();
      if (L.labels) label(`r${i + 1}`, r.center[0] + r.radius * 0.7, r.center[1] - r.radius, [210, 210, 210]);
    });
    ctx.setLineDash([]);
  }
  if (L.clusters) {
    ctx.setLineDash([6 * s, 4 * s]);
    ctx.lineWidth = Math.max(1.5, 1.5 * s);
    ctx.strokeStyle = `rgb(${COL.cluster.join(",")})`;
    for (const c of res.clusters) {
      const hull = c.hull;
      ctx.beginPath();
      if (hull.length >= 3) {
        // inflate hull a little around centroid so members stay visible
        hull.forEach(([x, y], i) => {
          const dx = x - c.centroid[0],
            dy = y - c.centroid[1];
          const n = Math.hypot(dx, dy) || 1;
          const px = x + (dx / n) * 8 * s,
            py = y + (dy / n) * 8 * s;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
      }
      ctx.stroke();
      if (L.labels) {
        const top = Math.min(...hull.map((p) => p[1]));
        label(`${c.id}  n=${c.count}`, c.centroid[0] - 20 * s, top - 26 * s, COL.cluster);
      }
    }
    ctx.setLineDash([]);
  }
  if (L.calcs) {
    ctx.lineWidth = Math.max(1, 1.2 * s);
    ctx.strokeStyle = `rgb(${COL.calc.join(",")})`;
    for (const c of res.calcs) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 5 * s, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (L.labels && res.calcs.length <= 40)
      for (const c of res.calcs) label(c.id, c.x + 6 * s, c.y - 4 * s, COL.calc);
  }
  if (L.labels) {
    if (L.masses)
      for (const m of res.masses)
        label(`${m.id}  ${(m.confidence * 100).toFixed(0)}%  ⌀${m.features.equivDiameterPx.toFixed(0)}px`, m.bbox[0], m.bbox[1] - fs - 8, COL.mass);
    if (L.anatomy) {
      if (res.pectoral.found) {
        const ys: number[] = [];
        const xs: number[] = [];
        for (let y = 0; y < h; y += 4)
          for (let x = 0; x < w; x += 4)
            if (res.masks.pectoral[y * w + x]) {
              xs.push(x);
              ys.push(y);
            }
        if (xs.length) label("Pectoral (excluded)", xs[Math.floor(xs.length / 2)] - 60 * s, ys[Math.floor(ys.length / 2)], COL.pectoral);
      }
      label(res.nipple.inProfile ? "Nipple" : "Apex / nipple zone", res.nipple.apex[0] + 4, res.nipple.apex[1] - 28 * s, COL.nipple);
      res.markers.forEach((k) => label(k.id + " marker?", k.centroid[0] + 10 * s, k.centroid[1] - 6 * s, COL.marker));
      label("skin band", 6, h - fs - 12, COL.skin);
    }
  }
}
