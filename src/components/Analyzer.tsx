"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_PARAMS, runPipeline, type CadResult } from "@/lib/cad/pipeline";
import { makePhantom } from "@/lib/cad/phantom";
import { renderView, VIEW_PRESETS, type Layers, type ViewKey } from "@/lib/cad/render";

interface HistoryRow {
  id: number;
  filename: string;
  width: number;
  height: number;
  projection: string;
  massCount: number;
  calcCandidateCount: number;
  calcClusterCount: number;
  runtimeMs: number;
  createdAt: string;
}

const VIEWS: Array<{ key: ViewKey; label: string }> = [
  { key: "original", label: "1 · Original" },
  { key: "anatomy", label: "2 · Anatomical segmentation" },
  { key: "masses", label: "3 · Mass detection / segmentation" },
  { key: "calcs", label: "4 · Calcification detection" },
  { key: "combined", label: "5 · Final combined" },
];

const MAX_SIDE = 1400;

export default function Analyzer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [base, setBase] = useState<ImageData | null>(null);
  const [gray, setGray] = useState<Float32Array | null>(null);
  const [name, setName] = useState("");
  const [source, setSource] = useState<"upload" | "phantom" | null>(null);
  const [res, setRes] = useState<CadResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<ViewKey>("combined");
  const [layers, setLayers] = useState<Layers>({ anatomy: true, masses: true, calcs: true, clusters: true, labels: true, rejected: false });
  const [spacing, setSpacing] = useState(0);
  const [calcZ, setCalcZ] = useState(DEFAULT_PARAMS.calcZ);
  const [massZ, setMassZ] = useState(DEFAULT_PARAMS.massZ);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [saveMsg, setSaveMsg] = useState("");
  const [scaleNote, setScaleNote] = useState("");

  const loadHistory = useCallback(async () => {
    try {
      const r = await fetch("/api/analyses");
      const j = (await r.json()) as { rows: HistoryRow[] };
      setHistory(j.rows ?? []);
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const setFromImageData = (img: ImageData, nm: string, src: "upload" | "phantom") => {
    const g = new Float32Array(img.width * img.height);
    for (let i = 0; i < g.length; i++) g[i] = 0.299 * img.data[i * 4] + 0.587 * img.data[i * 4 + 1] + 0.114 * img.data[i * 4 + 2];
    setBase(img);
    setGray(g);
    setName(nm);
    setSource(src);
    setRes(null);
    setSaveMsg("");
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    const url = URL.createObjectURL(f);
    const im = new Image();
    im.onload = () => {
      const sc = Math.min(1, MAX_SIDE / Math.max(im.naturalWidth, im.naturalHeight));
      const w = Math.round(im.naturalWidth * sc),
        h = Math.round(im.naturalHeight * sc);
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(im, 0, 0, w, h);
      setScaleNote(sc < 1 ? `Image downscaled ${im.naturalWidth}×${im.naturalHeight} → ${w}×${h} (small calcifications may be lost).` : `Native resolution ${w}×${h} kept (no resampling).`);
      setFromImageData(ctx.getImageData(0, 0, w, h), f.name, "upload");
      URL.revokeObjectURL(url);
    };
    im.src = url;
  };

  const usePhantom = () => {
    const w = 480,
      h = 680;
    const { gray: g } = makePhantom(w, h);
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < g.length; i++) {
      data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = g[i];
      data[i * 4 + 3] = 255;
    }
    setScaleNote("Synthetic phantom with known ground truth (software self-test only; not a clinical image).");
    setFromImageData(new ImageData(data, w, h), "synthetic-phantom.png", "phantom");
  };

  const run = () => {
    if (!gray || !base) return;
    setBusy(true);
    setSaveMsg("");
    setTimeout(() => {
      try {
        const t0 = performance.now();
        const r = runPipeline(gray, base.width, base.height, { calcZ, massZ, pixelSpacingMm: spacing });
        r.timings.total = Math.round(performance.now() - t0);
        setRes(r);
      } finally {
        setBusy(false);
      }
    }, 30);
  };

  const activeLayers: Layers = view === "combined" ? layers : VIEW_PRESETS[view];
  useEffect(() => {
    if (base && canvasRef.current) renderView(canvasRef.current, base, res, activeLayers);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, res, view, layers]);

  const download = (key: ViewKey) => {
    if (!base) return;
    const c = document.createElement("canvas");
    renderView(c, base, res, key === "combined" ? layers : VIEW_PRESETS[key]);
    const a = document.createElement("a");
    a.download = `${name.replace(/\.[^.]+$/, "")}_${key}.png`;
    a.href = c.toDataURL("image/png");
    a.click();
  };

  const save = async () => {
    if (!res) return;
    const body = {
      filename: name,
      width: res.width,
      height: res.height,
      projection: res.projection.view,
      pectoralFound: res.pectoral.found,
      massCount: res.masses.length,
      calcCandidateCount: res.calcs.length,
      calcClusterCount: res.clusters.length,
      runtimeMs: res.timings.total ?? 0,
      params: res.params,
      findings: {
        source,
        density: res.density,
        nipple: res.nipple,
        markers: res.markers,
        masses: res.masses.map((m) => ({ id: m.id, centroid: m.centroid, bbox: m.bbox, confidence: m.confidence, features: m.features, descriptors: m.descriptors })),
        calcs: res.calcs,
        clusters: res.clusters.map((c) => ({ id: c.id, count: c.count, centroid: c.centroid, hullAreaPx: c.hullAreaPx })),
        log: res.log,
      },
    };
    const r = await fetch("/api/analyses", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    setSaveMsg(r.ok ? "Summary saved (no pixel data stored)." : "Save failed.");
    void loadHistory();
  };

  const tog = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
        <b>Research / CAD-development tool – not a diagnostic device.</b> Everything below is computed by the deterministic algorithms documented in
        the Methodology tab, running locally in your browser. Output is a <i>candidate</i> list for radiologist review. Un-validated thresholds; no neural network is executed here.
      </div>

      <section className="grid gap-4 rounded-xl bg-white p-5 shadow-sm lg:grid-cols-[1fr_auto]">
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">1 · Load a mammogram</h2>
          <div className="flex flex-wrap items-center gap-3">
            <label className="cursor-pointer rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700">
              Upload image (PNG / JPEG)
              <input type="file" accept="image/png,image/jpeg" className="hidden" onChange={(e) => void onFile(e.target.files?.[0])} />
            </label>
            <button onClick={usePhantom} className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium hover:bg-slate-50">
              Load synthetic phantom (self-test)
            </button>
            <span className="text-sm text-slate-600">{name && `Loaded: ${name}`}</span>
          </div>
          <p className="text-xs text-slate-500">{scaleNote || "DICOM must be window/levelled and exported to 8-bit PNG first; for calcification work use full-resolution (≈50–100 µm/pixel) exports."}</p>
          <div className="flex flex-wrap items-end gap-4 text-sm">
            <label className="flex flex-col gap-1">
              Pixel spacing (mm, 0 = unknown)
              <input type="number" step="0.001" min="0" value={spacing} onChange={(e) => setSpacing(Number(e.target.value))} className="w-40 rounded border border-slate-300 px-2 py-1" />
            </label>
            <label className="flex flex-col gap-1">
              Calcification Z threshold
              <input type="number" step="0.1" min="2" value={calcZ} onChange={(e) => setCalcZ(Number(e.target.value))} className="w-32 rounded border border-slate-300 px-2 py-1" />
            </label>
            <label className="flex flex-col gap-1">
              Mass blob Z threshold
              <input type="number" step="0.1" min="2" value={massZ} onChange={(e) => setMassZ(Number(e.target.value))} className="w-32 rounded border border-slate-300 px-2 py-1" />
            </label>
            <button disabled={!base || busy} onClick={run} className="rounded-lg bg-teal-700 px-5 py-2 font-medium text-white hover:bg-teal-600 disabled:opacity-40">
              {busy ? "Running pipeline…" : "Run CAD pipeline"}
            </button>
          </div>
        </div>
        {res && (
          <div className="min-w-60 rounded-lg bg-slate-50 p-3 text-xs text-slate-700">
            <div className="mb-1 font-semibold">Stage timings (ms)</div>
            {Object.entries(res.timings).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span>{k}</span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {base && (
        <section className="rounded-xl bg-white p-5 shadow-sm">
          <div className="mb-3 flex flex-wrap gap-2">
            {VIEWS.map((v) => (
              <button key={v.key} onClick={() => setView(v.key)} className={`rounded-full px-3 py-1.5 text-sm ${view === v.key ? "bg-slate-900 text-white" : "bg-slate-100 hover:bg-slate-200"}`}>
                {v.label}
              </button>
            ))}
            <button onClick={() => download(view)} className="ml-auto rounded-full border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
              Download this view (PNG)
            </button>
            <button
              onClick={() => VIEWS.forEach((v, i) => setTimeout(() => download(v.key), i * 250))}
              className="rounded-full border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"
            >
              Download all 5
            </button>
          </div>
          {view === "combined" && (
            <div className="mb-3 flex flex-wrap gap-4 text-sm">
              {(Object.keys(layers) as Array<keyof Layers>).map((k) => (
                <label key={k} className="flex items-center gap-1.5">
                  <input type="checkbox" checked={layers[k]} onChange={() => tog(k)} /> {k}
                </label>
              ))}
            </div>
          )}
          <div className="flex justify-center rounded-lg bg-black p-2">
            <canvas ref={canvasRef} className="max-h-[85vh] max-w-full object-contain" />
          </div>
          {!res && <p className="mt-2 text-sm text-slate-500">Press “Run CAD pipeline” to generate the layers. The original is never modified; overlays are drawn on a copy.</p>}
          {res && (
            <p className="mt-2 text-xs text-slate-500">
              Legend – green: breast boundary · blue: pectoral · yellow band: skin · magenta: nipple zone · orange: marker/artifact candidate · red: mass mask · cyan circles: calcification candidates · yellow dashed: cluster hull · grey dashed: rejected mass candidates.
            </p>
          )}
        </section>
      )}

      {res && <Results res={res} onSave={save} saveMsg={saveMsg} />}

      <section className="rounded-xl bg-white p-5 shadow-sm">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Saved run summaries</h2>
          <button
            className="text-xs text-red-600 underline"
            onClick={async () => {
              await fetch("/api/analyses", { method: "DELETE" });
              void loadHistory();
            }}
          >
            clear
          </button>
        </div>
        {history.length === 0 ? (
          <p className="text-sm text-slate-500">No saved runs yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3">When</th>
                  <th className="pr-3">File</th>
                  <th className="pr-3">Size</th>
                  <th className="pr-3">View</th>
                  <th className="pr-3">Masses</th>
                  <th className="pr-3">Calc cand.</th>
                  <th className="pr-3">Clusters</th>
                  <th>ms</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.id} className="border-t border-slate-100">
                    <td className="py-1 pr-3">{new Date(h.createdAt).toLocaleString()}</td>
                    <td className="pr-3">{h.filename}</td>
                    <td className="pr-3">
                      {h.width}×{h.height}
                    </td>
                    <td className="pr-3">{h.projection}</td>
                    <td className="pr-3">{h.massCount}</td>
                    <td className="pr-3">{h.calcCandidateCount}</td>
                    <td className="pr-3">{h.calcClusterCount}</td>
                    <td>{Math.round(h.runtimeMs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Results({ res, onSave, saveMsg }: { res: CadResult; onSave: () => void; saveMsg: string }) {
  const f = (v: number, d = 2) => v.toFixed(d);
  return (
    <section className="space-y-5 rounded-xl bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Findings (algorithm output)</h2>
        <div className="flex items-center gap-3">
          {saveMsg && <span className="text-xs text-teal-700">{saveMsg}</span>}
          <button onClick={onSave} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
            Save summary to database
          </button>
        </div>
      </div>

      <div className="grid gap-3 text-sm md:grid-cols-3">
        <Card title="Projection / orientation">
          {res.projection.view}
          <br />
          Chest wall on image {res.projection.chestWall}
        </Card>
        <Card title="Density (area heuristic)">
          Fibroglandular fraction {f(res.density.fibroglandularFraction * 100, 0)}% → ACR-like category {res.density.category}
          <br />
          <span className="text-xs text-slate-500">Not a validated BI-RADS density.</span>
        </Card>
        <Card title="Nipple">
          Apex ({res.nipple.apex[0]}, {res.nipple.apex[1]}) – {res.nipple.inProfile ? "in profile" : "not clearly in profile"}; excluded radius {f(res.nipple.radiusPx, 0)} px
        </Card>
      </div>

      <div>
        <h3 className="mb-1 font-semibold">Masses</h3>
        {res.masses.length === 0 ? (
          <p className="rounded bg-slate-50 p-3 text-sm">No reliable mass candidate detected.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3">ID</th>
                  <th className="pr-3">Centroid</th>
                  <th className="pr-3">Area px</th>
                  <th className="pr-3">⌀ px</th>
                  <th className="pr-3">minor/major</th>
                  <th className="pr-3">Circ.</th>
                  <th className="pr-3">Solid.</th>
                  <th className="pr-3">ΔN</th>
                  <th className="pr-3">Sharp.</th>
                  <th className="pr-3">Rough.</th>
                  <th className="pr-3">Lucent frac.</th>
                  <th className="pr-3">Conf.</th>
                  <th>Descriptors (suggestions)</th>
                </tr>
              </thead>
              <tbody>
                {res.masses.map((m) => (
                  <tr key={m.id} className="border-t border-slate-100 align-top">
                    <td className="py-1 pr-3 font-semibold text-red-600">{m.id}</td>
                    <td className="pr-3">
                      {f(m.centroid[0], 0)}, {f(m.centroid[1], 0)}
                    </td>
                    <td className="pr-3">{m.features.areaPx}</td>
                    <td className="pr-3">{f(m.features.equivDiameterPx, 0)}</td>
                    <td className="pr-3">{f(m.features.axisRatio)}</td>
                    <td className="pr-3">{f(m.features.circularity)}</td>
                    <td className="pr-3">{f(m.features.solidity)}</td>
                    <td className="pr-3">{f(m.features.contrastDiff, 0)}</td>
                    <td className="pr-3">{f(m.features.marginSharpness)}</td>
                    <td className="pr-3">{f(m.features.radialRoughness, 3)}</td>
                    <td className="pr-3">{f(m.features.internalLucentFraction)}</td>
                    <td className="pr-3">{f(m.confidence * 100, 0)}%</td>
                    <td>
                      shape: {m.descriptors.shape}; margin: {m.descriptors.margin}; density: {m.descriptors.density}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {res.rejectedMasses.length > 0 && (
          <details className="mt-2 text-xs text-slate-600">
            <summary className="cursor-pointer">Rejected mass candidates ({res.rejectedMasses.length})</summary>
            <ul className="ml-4 mt-1 list-disc">
              {res.rejectedMasses.map((r, i) => (
                <li key={i}>
                  r{i + 1} @ ({f(r.center[0], 0)}, {f(r.center[1], 0)}) r≈{f(r.radius, 0)} z={f(r.z, 1)} – {r.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <div>
        <h3 className="mb-1 font-semibold">Calcification candidates</h3>
        {res.calcs.length === 0 ? (
          <p className="rounded bg-slate-50 p-3 text-sm">No reliable calcification candidate detected.</p>
        ) : (
          <p className="text-sm">
            {res.calcs.length} candidate(s) (Z ≥ {f(res.calcInfo.thresholdZ, 1)}); {res.clusters.length} cluster(s) with ≥{res.params.clusterMinPts} members within ε = {f(res.calcInfo.epsPx, 0)} px.
            {res.clusters.map((c) => (
              <span key={c.id} className="ml-2 rounded bg-yellow-100 px-2 py-0.5 text-xs">
                {c.id}: n={c.count}, hull {f(c.hullAreaPx, 0)} px², @ ({f(c.centroid[0], 0)}, {f(c.centroid[1], 0)})
              </span>
            ))}
          </p>
        )}
        <details className="mt-1 text-xs text-slate-600">
          <summary className="cursor-pointer">Candidate funnel & rejections</summary>
          <p className="mt-1">
            Connected components after hysteresis: {res.calcInfo.rawCount}; kept: {res.calcInfo.keptCount}.
          </p>
          <ul className="ml-4 list-disc">
            {Object.entries(res.calcInfo.rejectedByReason).map(([k, v]) => (
              <li key={k}>
                {k}: {v}
              </li>
            ))}
          </ul>
          {res.calcs.length > 0 && (
            <table className="mt-2 text-left">
              <thead>
                <tr className="text-slate-500">
                  <th className="pr-3">ID</th>
                  <th className="pr-3">x,y</th>
                  <th className="pr-3">area</th>
                  <th className="pr-3">tophat</th>
                  <th className="pr-3">Z</th>
                  <th className="pr-3">elong.</th>
                  <th className="pr-3">cluster</th>
                  <th>in mass</th>
                </tr>
              </thead>
              <tbody>
                {res.calcs.slice(0, 80).map((c) => (
                  <tr key={c.id}>
                    <td className="pr-3">{c.id}</td>
                    <td className="pr-3">
                      {f(c.x, 0)},{f(c.y, 0)}
                    </td>
                    <td className="pr-3">{c.areaPx}</td>
                    <td className="pr-3">{f(c.peakTophat, 0)}</td>
                    <td className="pr-3">{f(c.peakZ, 1)}</td>
                    <td className="pr-3">{f(c.elongation, 1)}</td>
                    <td className="pr-3">{c.clusterId ?? "isolated"}</td>
                    <td>{c.insideMass ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </details>
      </div>

      <div>
        <h3 className="mb-1 font-semibold">Anatomical exclusion</h3>
        <ul className="ml-4 list-disc text-sm">
          <li>Pectoral: {res.pectoral.found ? `excluded (RANSAC inliers ${res.pectoral.inliers}, contrast Δ ${f(res.pectoral.contrast, 0)})` : "not detected"} – {res.pectoral.note}</li>
          <li>Markers/artifacts flagged: {res.markers.length ? res.markers.map((m) => `${m.id} (${m.kind}, ${m.areaPx}px)`).join("; ") : "none"}</li>
        </ul>
      </div>

      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer font-semibold">Execution log</summary>
        <ul className="ml-4 mt-1 list-disc">
          {res.log.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</div>
      {children}
    </div>
  );
}
