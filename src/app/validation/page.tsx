import { Callout, Code, M, Page, Section, Table } from "@/components/ui";

export const metadata = { title: "Scientific validation (Part 6)" };

export default function ValidationPage() {
  return (
    <Page
      title="Scientific validation plan"
      intro="A good-looking overlay is not validation. Nothing in this project has been validated on clinical data; the only quantitative checks were on a synthetic phantom."
    >
      <Callout kind="warn">
        Current evidence level: software verification on one synthetic phantom (known truth): mass centroid error ≈ 1 px, pectoral line parameters recovered, calcification recall 10–11 of 12–14 phantom dots.
        The phantom was authored by the same person as the algorithms (circularity risk), so it proves execution, not performance.
      </Callout>

      <Section title="Metrics and where they apply">
        <Table
          head={["Metric", "Definition", "Mass segmentation", "Mass detection", "Calcification detection", "Notes"]}
          rows={[
            ["Dice", "2|A∩B| / (|A|+|B|)", "Primary", "—", "Not meaningful per dot; optional on cluster masks", "Compare with ≥2 readers; report inter-reader Dice as the ceiling"],
            ["IoU / Jaccard", "|A∩B| / |A∪B|", "Primary (with Dice)", "Box IoU ≥ 0.25–0.5 (or centre-in-lesion) defines a hit", "Cluster-level IoU of hulls", "Report distributions (median, IQR), not only means"],
            ["Sensitivity = recall", "TP / (TP+FN)", "Pixel-level secondary", "Lesion-level, at fixed FP/image", "Per-calcification and per-cluster", "Always report at an operating point"],
            ["Precision", "TP / (TP+FP)", "Pixel-level", "Lesion-level", "Per-candidate", "Depends on prevalence"],
            ["Specificity", "TN / (TN+FP)", "Not informative (huge background TN)", "Only at image/case level", "Case level", "Avoid pixel specificity"],
            ["FP per image", "mean FP detections / image", "—", "Primary", "Primary (per image and per cluster)", "Report separately for normals and abnormals"],
            ["FROC", "Lesion sensitivity vs FPPI curve; partial AUC at 0.25–4 FPPI", "—", "Primary", "Primary", "Bootstrap CIs at case level; JAFROC for reader comparison"],
            ["ROC/AUC (case level)", "—", "—", "Secondary (malignant vs not)", "Secondary", "Needs pathology-proven labels"],
            ["Hausdorff-95 / boundary F1", "Surface distance", "Recommended: margin-sensitive", "—", "—", "Dice hides boundary errors on large masses"],
          ]}
        />
        <p>
          Segmentation of large masses: Dice/IoU + HD95. Detection of masses and calcifications: lesion-level sensitivity vs FPPI (FROC). Calcifications additionally need cluster-level matching (a cluster is a TP if its hull overlaps a reference cluster hull, or ≥ 50% of its members fall in it).
        </p>
      </Section>

      <Section title="Protocol">
        <ol className="ml-5 list-decimal space-y-1.5">
          <li>Datasets: INbreast (dense annotations incl. calcifications), CBIS-DDSM (masses/calcs; known annotation noise), VinDr-Mammo, plus a local retrospective set from the target vendor. Check licence and de-identification.</li>
          <li>Splits at the <b>patient</b> level (never per image/breast/view); thresholds, Z values and any trained model tuned only on train/validation; the test set is touched once.</li>
          <li>Reference standard: ≥2 breast radiologists with consensus/adjudication; biopsy/follow-up for malignancy labels; annotate calcifications individually and as clusters on full-resolution images.</li>
          <li>Ablations: with/without each exclusion mask; FCM vs thresholding vs U-Net; top-hat radii; Z and hysteresis; resolution (native vs ÷2) to quantify the calcification loss.</li>
          <li>Stratify by density, view (CC/MLO), vendor, lesion size, and margin class; report failures (e.g. pectoral detection failure rate, exclusion-induced misses).</li>
          <li>Anatomy sub-tasks: pectoral Dice/boundary distance, nipple localisation error, marker detection recall/precision against manual masks.</li>
          <li>Statistics: bootstrap (case-level resampling, ≥1000×) CIs; paired tests (McNemar, DeLong, JAFROC) versus the existing MammographyCAD; pre-register the primary endpoint (e.g. sensitivity at 1 FPPI).</li>
          <li>Reader study if clinical claims are intended; regulatory pathway (e.g. FDA/CE) is a separate effort.</li>
        </ol>
        <Code>{`# minimal evaluation helpers
def dice(a, b): return 2*np.sum(a & b) / max(1, a.sum() + b.sum())
def iou(a, b):  return np.sum(a & b) / max(1, np.sum(a | b))
# FROC: sort detections by score; for each threshold count lesion hits and false marks / image;
# a detection hits a lesion if its centre lies in the reference mask (or box IoU >= 0.25).`}</Code>
      </Section>

      <Section title="Honest limitations of the proposed system">
        <ul className="ml-5 list-disc space-y-1">
          <li>Thresholds (Z = 4.5, ΔN ≥ 18, solidity, axis ratio) are hand-set priors, not fitted; margin descriptors and the confidence score are heuristics, not BI-RADS.</li>
          <li>Single-projection analysis cannot establish 3-D position, composition or malignancy; architectural distortion, asymmetries and diffuse/segmental calcifications are not targeted.</li>
          <li>Screen-capture-scale JPEG/PNG input lacks calcification information; DICOM at native pitch is required for the calcification stage.</li>
          <li>Exclusion masks can hide real findings (axillary nodes in the pectoral zone, skin calcifications): these must be surfaced as “review” items.</li>
          <li>Percentile normalisation and pectoral RANSAC are fragile for implants, post-surgical breasts, very small breasts or unusual positioning; CC views have no pectoral expected.</li>
        </ul>
        <p>
          Source of truth for everything here: <M>src/lib/cad/*.ts</M> (live in the Analyzer) and <M>python/mammocad/*.py</M>.
        </p>
      </Section>
    </Page>
  );
}
