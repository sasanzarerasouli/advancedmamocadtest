import fs from "node:fs";
import path from "node:path";
import { Callout, Code, M, Page, Section, Table } from "@/components/ui";

export const metadata = { title: "Python implementation design" };
export const dynamic = "force-dynamic";

const FILES = [
  "image_preprocessing.py",
  "breast_segmentation.py",
  "anatomical_segmentation.py",
  "mass_detection.py",
  "mass_segmentation.py",
  "calcification_detection.py",
  "candidate_classification.py",
  "visualization.py",
  "pipeline.py",
];

function read(rel: string) {
  try {
    return fs.readFileSync(path.join(process.cwd(), "python", rel), "utf8");
  } catch {
    return "# file not found in this deployment";
  }
}

export default function PythonPage() {
  return (
    <Page
      title="Implementation-ready Python design (Part 4)"
      intro={
        <>
          Package <M>python/mammocad</M>. Dependencies: numpy, scipy, opencv-python(-headless), scikit-image, scikit-learn (PyTorch / Ultralytics only for the optional detector, see Integration). Each module is
          independent: pure functions, NumPy in / NumPy out, no global state, so they can be unit-tested and called from an existing PyQt6 application.
        </>
      }
    >
      <Callout kind="ok">
        <b>What was executed:</b> <M>python python/selftest.py</M> ran the whole package on a synthetic phantom in this sandbox (OpenCV 5.0 / scikit-image / scikit-learn): pectoral edge recovered (a = 0.448, b = 298; truth 0.45, 298), one mass segmented, calcification recall 11/14 on the phantom.
        <b> What was not:</b> the code has never been run on the uploaded mammogram or any clinical data; <M>candidate_classification.py</M> trainers are untrained scaffolds.
      </Callout>

      <Section title="Module contract table">
        <Table
          head={["Module · function", "Input → output", "Algorithm / mathematical basis", "Recommended parameters", "Cost · known limits"]}
          rows={[
            [<M key="a">image_preprocessing.normalize_in_mask</M>, "gray f32 HxW, mask → N f32 in [0,255]", "Percentile stretch inside breast", "P1 / P99.8", "O(HW) · fails if mask wrong"],
            [<M key="b">breast_segmentation.segment_breast</M>, "gray → mask u8, info", "Background-mode threshold T = bg + max(6, 5σ), CC, fill", "k=5, blur 1.5", "O(HW) · very noisy backgrounds / grids"],
            [<M key="c">anatomical_segmentation.detect_pectoral</M>, "N, breast, xb → mask, info", "Edge scan + RANSAC line + contrast check", "k 2%W, tol 8 px, Δ≥5", "~50 ms · curved or faint muscle edge; use U-Net"],
            [<M key="d">anatomical_segmentation.skin_band / detect_nipple / detect_markers</M>, "breast, N → masks", "Distance transform; boundary profile curvature; saturation + shape", "1.2%W; 5% depth; N≥250", "O(HW) · nipple in profile is not guaranteed; real skin calcs may be dropped"],
            [<M key="e">mass_detection.mass_candidates</M>, "N, eligible → [{x,y,r,z}]", "DoG scale-space with normalised convolution, robust z", "8 scales, Z≥4, top 6", "~100 ms at 200 px · diffuse asymmetries, architectural distortion not captured"],
            [<M key="f">mass_segmentation.segment_mass</M>, "N, eligible, breast, cand → mask, info", "Local FCM (c=2, m=2) + morphology + curvature smoothing", "2.2 r window, 1.7 r cap", "~50 ms · needs bright-vs-surround contrast; overlapping dense tissue leaks"],
            [<M key="g">calcification_detection.detect_calcifications</M>, "N, eligible → candidates, clusters", "Multi-scale white top-hat + local Z + hysteresis + DBSCAN", "r {2,3,5}, Z 4.5, area 2–80, eps 5 mm, minPts 5", "O(HW·#radii) · resolution-dependent, must rescale radii to mm"],
            [<M key="h">candidate_classification.*</M>, "feature dict/table → bool / prob", "Rule gates; GLCM; calibrated HGB; RandomForest", "see file", "Needs annotated data"],
            [<M key="i">visualization.render</M>, "gray8, results → BGR image", "Alpha-blended layers + contours", "alpha 0.3", "—"],
            [<M key="j">pipeline.run_pipeline</M>, "image → results dict", "Orchestrates all stages, mirrors if chest wall left", "PipelineConfig", "~1 s for 0.3 MP"],
          ]}
        />
      </Section>

      <Section title="Usage">
        <Code>{`pip install numpy scipy opencv-python-headless scikit-image scikit-learn
python -m mammocad.pipeline image.png --out out --pixel-mm 0.07   # run from python/
python python/selftest.py                                            # phantom smoke test`}</Code>
      </Section>

      {FILES.map((f) => (
        <Section key={f} title={`python/mammocad/${f}`}>
          <Code>{read(`mammocad/${f}`)}</Code>
        </Section>
      ))}
      <Section title="python/selftest.py">
        <Code>{read("selftest.py")}</Code>
      </Section>
    </Page>
  );
}
