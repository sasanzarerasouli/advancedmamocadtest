import { Callout, Code, M, Page, Section, Table } from "@/components/ui";

export const metadata = { title: "MammographyCAD integration (Part 5)" };

export default function IntegrationPage() {
  return (
    <Page
      title="Integration into the existing MammographyCAD (PyQt6 · OpenCV · scikit-image · FCM · GLCM · YOLO11s)"
      intro="Design only: no files of your project were modified. The guiding rule is additive integration: keep the GUI, viewer and annotation layer, and insert the new stages behind a thin service interface."
    >
      <Section title="1 · Reuse / replace / add">
        <Table
          head={["Existing component", "Decision", "Reason"]}
          rows={[
            ["PyQt6 GUI, image viewer, annotation/ROI drawing", "Reuse unchanged", "Only needs to consume a new result object (layers + findings)"],
            ["OpenCV / scikit-image utilities", "Reuse", "New modules use the same libraries"],
            ["YOLO11s detector", "Reuse as proposal generator; add anatomy-aware filtering + fusion", "It supplies learned localisation; the new gate supplies explainable rejection"],
            ["Fuzzy C-Means", "Reuse, but run locally (window 2.2 r around a candidate, c = 2) instead of whole-image", "Whole-image FCM clusters tissue classes, not lesions; local FCM gives a boundary"],
            ["GLCM texture analysis", "Reuse as classifier features (`candidate_classification.glcm_features`)", "Texture is a useful FP-reduction feature, not a localiser"],
            ["Any fixed-ROI/ellipse or hard-coded coordinate region logic", "Replace", "Violates image-based localisation; replaced by blob → FCM contour"],
            ["Global-threshold or global-top-hat calcification step (if present)", "Replace", "Use multi-scale top-hat + local Z + hysteresis + DBSCAN"],
            ["Pectoral / nipple / marker handling", "Add", "`anatomical_segmentation.py` (+ optional U-Net pectoral model)"],
          ]}
        />
      </Section>

      <Section title="2 · New modules and layout">
        <Code>{`mammography_cad/
  gui/ (existing PyQt6)           <- unchanged; subscribes to AnalysisResult
  detectors/yolo11s_detector.py   <- existing, wrapped: detect(image) -> [Box(x1,y1,x2,y2,score)]
  pipeline/                       <- NEW (python/mammocad)
    image_preprocessing.py  breast_segmentation.py  anatomical_segmentation.py
    mass_detection.py  mass_segmentation.py  calcification_detection.py
    candidate_classification.py  fusion.py (new, below)  visualization.py
  services/analysis_service.py    <- NEW: QThread worker, owns run_pipeline + yolo, emits AnalysisResult
  storage/results_store.py        <- NEW: sqlite/JSON sidecar + PNG/NPZ masks`}</Code>
      </Section>

      <Section title="3 · Data flow with YOLO11s and the exclusion masks">
        <ol className="ml-5 list-decimal space-y-2">
          <li>
            <b>Preprocess + anatomy first</b> (cheap, deterministic): breast, pectoral, skin, nipple, markers → <M>eligible</M> map. Run on the image fed to YOLO as well, to log which boxes land on excluded anatomy.
          </li>
          <li>
            <b>YOLO11s</b> runs on the full image (or tiles at native resolution for calcification-size objects; imgsz 1280). Each box passes through the <i>anatomy gate</i>: reject if IoU(box, excluded) &gt; 0.5 or box centre ∈ excluded;
            move such boxes into a <M>review_excluded_zone</M> list (so the axillary-node-like opacity in your RMLO example is surfaced instead of silently dropped).
          </li>
          <li>
            <b>Classical candidates</b> (DoG blobs) are generated independently. <b>Fusion</b>: match box ↔ blob by centre distance &lt; 0.7·max(r, box radius). Matched → seed FCM with the YOLO box (window = box × 1.3) — the more reliable of the two localisations;
            YOLO-only → run FCM at the box centre and keep if shape/contrast gates pass; blob-only → keep as “classical-only, low confidence”.
          </li>
          <li>
            <b>Segmentation</b>: local FCM + morphology (mass_segmentation) → mask ∩ eligible. Optionally refine with morphological Chan–Vese (`skimage.segmentation.morphological_chan_vese`, 35 iterations, smoothing 2) initialised from the FCM mask.
          </li>
          <li>
            <b>Classification</b>: features (shape, margin, ΔN, GLCM) → rule gates → calibrated model if trained. Final score = 0.5·YOLO conf + 0.3·classifier prob + 0.2·rule score (weights to be fitted by logistic regression on a validation set, not fixed).
          </li>
          <li>
            <b>Calcifications</b>: run the top-hat/Z detector on native-resolution N ∩ eligible; DBSCAN clusters; optionally verify with a patch classifier. YOLO boxes of class “calcification” are only accepted if they contain ≥ 1 detector candidate.
          </li>
        </ol>
      </Section>

      <Section title="4 · Result object, display and storage">
        <Code>{`@dataclass
class AnalysisResult:
    image_id: str; orientation: dict; masks: dict[str, np.ndarray]   # breast, pectoral, skin, nipple, marker, excluded
    masses: list[MassFinding]       # id, mask(RLE), bbox, features, descriptors, yolo_score, clf_prob, source
    calcs: list[CalcCandidate]; clusters: list[CalcCluster]
    review_excluded_zone: list[Box]; rejected: list[Rejected(reason)]
    params: dict; versions: dict; timings: dict`}</Code>
        <ul className="ml-5 list-disc space-y-1">
          <li>
            <b>Display</b>: one QGraphicsScene with independent QGraphicsPixmapItem layers (anatomy / masses / calcs / clusters / labels / rejected), each with an alpha slider and visibility checkbox; the original pixmap item is never altered. Layer rendering
            is the logic in <M>visualization.render</M>.
          </li>
          <li>
            <b>Storage</b>: masks as lossless PNG or compressed NPZ (uint8), findings as JSON (versioned schema, parameter snapshot, software and model hashes), plus DICOM-SEG/GSPS export if you need PACS interoperability. Keep the original untouched.
          </li>
          <li>
            <b>Threading</b>: run in a QThread; emit progress per stage. Expected cost on a 0.3 MP image ≈ 1 s on CPU for the classical path (≈ 5–10 s on a 3–6 MP full-field image), YOLO11s adds ≈ 20–100 ms on GPU.
          </li>
          <li>
            <b>Reader-in-the-loop</b>: allow the radiologist to accept/reject/edit each finding; log decisions. These become labelled training data for Stage-5 classifiers.
          </li>
        </ul>
        <Callout kind="warn">Pectoral exclusion must never suppress review: anything flagged inside an excluded zone appears in a separate “review” list with the reason, as in the RMLO example (upper-right round opacity).</Callout>
      </Section>
    </Page>
  );
}
