import Link from "next/link";
import { Callout, Page, Section, Table } from "@/components/ui";

export const metadata = { title: "Image report – RMLO" };

export default function ReportPage() {
  return (
    <Page
      title="Analysis of the uploaded image (burnt-in label “RMLO”, ≈446 × 655 px as displayed)"
      intro="Parts 1, 2 and 6 of the request. This page separates what was observed visually from what was computed, because those are not the same thing."
    >
      <Callout kind="warn">
        <b>Disclosure (read first).</b> The chat attachment reached me as an image for visual (multimodal) inspection only. It was <b>not</b> saved as a file in the execution sandbox, so
        <b> no image-processing algorithm and no neural network was run on those pixels</b> by me. Consequently: (1) every observation in the next section is <i>visual reasoning by a multimodal model</i>,
        with coordinates that are approximate (±10–15 px, in the 446×655 frame I saw) and <b>are not segmentation output</b>; (2) I cannot honestly hand you pixel-accurate mask files for that specific upload.
        What I did instead: I implemented and executed the full algorithmic pipeline (TypeScript in the browser, Python reference) and verified it on a synthetic phantom with known ground truth.
        <b> Upload the same image in the <Link href="/" className="underline">Analyzer</Link></b> and the five requested outputs (original, anatomy, mass, calcification, combined) are computed from its real pixels, with downloadable PNGs.
      </Callout>

      <Section title="A · Findings observed visually (multimodal reading, not executed algorithms)">
        <Table
          head={["Item", "Observation", "Confidence / caveat"]}
          rows={[
            ["Projection & laterality", "Burnt-in “RMLO”. Chest wall and pectoral muscle on the image right, nipple direction to the left: consistent with a right-breast mediolateral oblique, displayed in standard orientation.", "High"],
            ["Breast boundary / skin", "Smooth convex anterior-inferior contour, thin uniform skin line; posterior-superior region extends into the axillary tail up to the top border. Dark (fat) subcutaneous layer is wide in the upper breast.", "High for outline; skin thickness not assessable at this resolution"],
            ["Pectoral muscle", "Homogeneous brighter band at the upper right, bounded by an oblique, slightly convex edge running from the top border (≈ x 300) down-right towards the right border (lower tip ≈ y 380–420).", "Medium-high; the lower tip is faint"],
            ["Nipple / nipple marker", "No clearly projecting nipple in profile and no radiopaque nipple marker identified. The anterior contour near y ≈ 430 is rounded.", "Medium – a nipple may be en-face/retracted or simply not resolved"],
            ["Skin marker / artefact candidate", "Small, very bright compact object at/just inside the lower-anterior skin line (≈ x 135, y 597), possibly a skin marker or artefact. The “RMLO” bracketed text (top-left) is a burnt-in annotation in the black background.", "Low-medium – nature uncertain"],
            ["Overall density", "Predominantly fatty breast with scattered fibroglandular densities, denser anteriorly/inferiorly: subjective impression ACR a–b. The large mass confounds area-based estimates.", "Low-medium; density is a radiologist call"],
            [
              "Mass-like finding (single dominant)",
              "Large, round-to-oval, high-attenuation mass in the central-anterior breast, approx. extents x 165–275, y 340–450 (≈110 px across). Predominantly smooth, well-circumscribed margin with a thin partial lucent halo; heterogeneous internal density with a small ring-/crescent-like lower-density focus at its upper-right edge (≈ x 240, y 355) and a thin linear density leaving the lower-right margin (≈ x 270, y 420; vessel, ligament or overlapping tissue).",
              "Detection: high. Margin/shape descriptors: medium. Nature: NOT determinable (see E).",
            ],
            ["Other round opacities", "Several small, well-defined round opacities: ≈ (330, 295), (330, 345), (275, 150) (≈6–10 px) and a larger ≈20 px one at ≈ (395, 70) in the upper-right pectoral/axillary region (appearance compatible with lymph node or overlapping nodule).", "Low – not characterised; the upper-right one lies in the pectoral exclusion zone (see limitation below)"],
            ["Microcalcifications", "No reliable microcalcification or calcification cluster could be identified at the displayed resolution. Punctate sub-millimetre calcifications are ~1–3 px at native pitch and would not be resolved in a 446×655 rendering. No candidate is therefore reported.", "A negative at this resolution is NOT evidence of absence"],
            ["Noise / artefacts", "Image looks like a processed (screen-capture-scale) JPEG/PNG with low noise; no obvious motion blur, grid lines, or skin folds. Compression/rescaling would degrade calcification conspicuity.", "Medium"],
          ]}
        />
        <p>
          Radiologic <b>descriptors</b> (not a BI-RADS assessment): shape round/oval; margin predominantly circumscribed with partly obscured portions; density high relative to surrounding fibroglandular tissue;
          focal internal lucency. A circumscribed, high-density mass has a broad differential (e.g. cyst-like or solid benign lesion, fibroadenoma, fat-containing lesion if the lucency is fat, phyllodes tumour, circumscribed carcinoma). It is
          <b> not</b> established as benign or malignant from this single projection and warrants standard diagnostic work-up by a radiologist (comparison with priors, additional views, ultrasound).
        </p>
        <Callout kind="warn">
          Limitation of the exclusion design that this image exposes: a pectoral exclusion mask removes the axillary/pectoral zone from mass and calcification search. The round ≈20 px opacity at the top right would be
          excluded automatically. The CAD should therefore emit a separate “excluded-zone nodule/node review” flag instead of silently discarding it (see Integration).
        </Callout>
      </Section>

      <Section title="B · Computational methods actually executed">
        <ul className="ml-5 list-disc space-y-1">
          <li>
            <b>On the uploaded image: none.</b> (No file access, see disclosure.)
          </li>
          <li>
            <b>On a synthetic phantom (480×680, known ground truth):</b> the TypeScript pipeline (<code>src/lib/cad</code>) and the Python reference (<code>python/mammocad</code>) were executed. Observed in my runs: pectoral edge recovered as x = 0.448·y + 298 (truth
            0.45·y + 298) in both implementations; mass centroid error ≈ 1 px; saturated skin marker flagged and excluded; calcification cluster found (TS: 10 of 12 candidates clustered; Python: 11/14 truth points recalled). This only demonstrates that the code runs and is self-consistent. It is
            <b> not</b> a validation of diagnostic accuracy.
          </li>
          <li>The Analyzer runs the same TypeScript code on any image you load, entirely client-side.</li>
        </ul>
      </Section>

      <Section title="C · Proposed (implementable) methods not executed here">
        <ul className="ml-5 list-disc space-y-1">
          <li>CLAHE for display, GLCM texture features and learned classifiers (provided as Python code, executed only through the import/self-test path; not trained).</li>
          <li>YOLO11s detection fusion, U-Net refinement, active-contour (Chan–Vese / morphological Chan–Vese) refinement: described in Methodology/Integration.</li>
        </ul>
      </Section>

      <Section title="D · Requires training data or expert annotation">
        <ul className="ml-5 list-disc space-y-1">
          <li>Any learned mass/not-mass or calcification true/false classifier (needs pixel-level and lesion-level labels, e.g. INbreast, CBIS-DDSM, VinDr-Mammo or in-house data with patient-level splits).</li>
          <li>YOLO11s fine-tuning and its operating-point calibration; U-Net mass or pectoral segmenter; breast-density deep model.</li>
          <li>Calibration of all hand-set thresholds (Z = 4.5, ΔN ≥ 18, etc.) against a reference-annotated set for the specific detector/vendor.</li>
        </ul>
      </Section>

      <Section title="E · Not reliably establishable from a single mammographic projection">
        <ul className="ml-5 list-disc space-y-1">
          <li>Three-dimensional position (depth, quadrant), true lesion size and whether it is a true mass or a summation of tissue (needs the CC view, tomosynthesis or ultrasound).</li>
          <li>Cystic vs solid composition, whether the lucent focus is fat or an overlapping structure, and malignancy likelihood (needs ultrasound/biopsy).</li>
          <li>Margin class at the sub-millimetre level (spot compression / magnification views).</li>
          <li>Calcification morphology and distribution (needs magnification views at native resolution).</li>
          <li>Change over time and multiplicity (needs priors and the contralateral breast).</li>
        </ul>
      </Section>
    </Page>
  );
}
