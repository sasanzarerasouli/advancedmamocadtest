import { Callout, Code, M, Page, Section, Table } from "@/components/ui";

export const metadata = { title: "Methodology – Stages 1-5" };

export default function MethodologyPage() {
  return (
    <Page
      title="Algorithm reconstruction (Stages 1–5)"
      intro={
        <>
          Every parameter below is the value used in the executable code (<M>src/lib/cad/pipeline.ts</M> and <M>python/mammocad</M>). I chose methods from the image: a dark, fatty breast in MLO with one large, bright,
          well-delimited mass, a pectoral triangle, and no resolvable microcalcifications at display resolution. That drives: background-statistics thresholding (Otsu fails on dark breasts),
          DoG scale-space + local FCM (a large high-contrast blob with partial halo: no need for a heavy CNN to <i>segment</i>), and top-hat + local-Z for calcifications (small bright, locally salient objects).
        </>
      }
    >
      <Section title="Stage 1 – Image preprocessing" id="s1">
        <Table
          head={["Step", "Operation / formula", "Parameters", "Why"]}
          rows={[
            ["Grayscale", "G = 0.299 R + 0.587 G + 0.114 B (8-bit); 16-bit DICOM → scale by max", "—", "Mammograms are single-channel; screenshots may be RGB"],
            ["Background model", "bg = mode of histogram in [0,60]; σ_bg = std(pixels ≤ bg+10); T = bg + max(6, 5 σ_bg)", "k = 5, min offset 6", "Dark fatty breasts have Otsu thresholds that cut through the breast; background is a sharp mode"],
            ["Smoothing for mask", "Gaussian σ = 1.5 before thresholding", "σ=1.5", "Suppress noise-driven holes"],
            ["Intensity normalisation (in-breast)", "N = clip((I − P₁)/(P₉₉.₈ − P₁), 0, 1)·255, percentiles over breast pixels only", "P₁, P₉₉.₈", "Makes thresholds vendor/exposure independent; label text and black background cannot compress range"],
            ["CLAHE", "Display / margin viewing only: clipLimit 2.0, tile 8×8", "—", "Non-linear, amplifies noise → breaks Z-score calibration; NOT used for calcification detection"],
            ["Noise reduction", "Mass: Gaussian σ = 1.5 (FCM) ; Calcification: σ = 0.7 (keeps 2–3 px objects)", "σ 1.5 / 0.7", "Mass needs low noise; microcalcs need minimal blur (σ ≥ 1 would halve a 2-px object's peak)"],
            ["Resizing", "Native resolution preserved up to 1400 px longer side (browser) / 3000 px (Python). INTER_AREA for any shrink", "max side", "A 0.2 mm calcification is ~3 px at 70 µm; ×0.5 downsizing → 1.5 px and top-hat SNR collapses. Mass scale-space runs on a ÷(max/200) copy since blobs are large"],
            ["Edge/border guard", "3-px frame zeroed for calcification search", "3 px", "Morphological border artefacts"],
          ]}
        />
      </Section>

      <Section title="Stage 2 – Anatomical segmentation and exclusion" id="s2">
        <p>
          All anatomical masks are derived from pixels. Images with the chest wall on the left are mirrored internally (side = image border touched over more rows by the breast mask) and mirrored back at the end. Nothing is hard-coded to coordinates.
        </p>
        <Table
          head={["Structure", "Algorithm", "Parameters", "How it is excluded downstream"]}
          rows={[
            ["Breast region", "Threshold T (above) → open r=2 → largest 8-connected component (drops burnt-in “RMLO”, labels, markers in the background) → fill holes → close r=3", "see Stage 1", "Everything outside is invalid; also defines percentile normalisation"],
            [
              "Pectoral muscle (MLO)",
              "Per row y < 0.75H (step 2): x*(y) = argmax_x [mean G(y,x+1..x+k) − mean G(y,x−k..x−1)], G = Gaussian(N,3), k = 2% W, search from xb(y)+6%W to W−k. Keep points with response ≥ max(4, 0.4·P90). RANSAC line x = a·y + b (300 iterations, tol 8 px, a ∈ [0.1,2], seeded PRNG). Accept only if b ≥ 0.2W, y_end = (W−1−b)/a ≥ 0.12H, inliers ≥ 12, y-span ≥ 0.15H, and mean N over 6–20 px inside the edge exceeds the outside by ≥ 5. Mask = {x ≥ a·y + b, y ≤ y_end} ∩ breast",
              "k=2%W, tol 8, Δ≥5",
              "Dilated by 3 px and removed from the eligible map. In CC views the breast does not touch the top border or the validation fails → empty mask. A U-Net can replace this where the muscle edge is curved/faint",
            ],
            ["Skin boundary", "Chamfer/L2 distance d(x) to the background (not to the image border); skin band = {d ≤ max(3, 1.2%W)}", "1.2%W", "Removed: stops skin lines, folds and skin calcifications from becoming mass/calcification candidates"],
            [
              "Nipple",
              "Smoothed leftmost-boundary profile xb(y); apex = median row where xb ≤ min+2 within 12–90% of the breast height. In-profile test: bump = mean(xb(y*±4%H)) − xb(y*), wide = mean(xb(y*±12%H)) − xb(y*); in profile if bump > max(0.008W, 0.3·wide) (a circular contour gives ≈ wide/9)",
              "radius = 5% of breast depth (×1.3 if in profile)",
              "Disc ∩ breast is excluded as a precaution even when the nipple is not in profile; the report states honestly whether it is visible",
            ],
            [
              "Nipple / skin marker",
              "Pixels with N ≥ 250 (top ≈0.2% of breast), 8-connected components with 3 ≤ area ≤ 700, bbox fill ≥ 0.5, major/minor ≤ 3 and centre within 4 skin-widths of the skin line. Labelled “nipple-marker” if within 2 nipple radii of the apex, else “skin marker/artefact”. Mask = disc(bbox/2 + 3 px)",
              "area 3–700",
              "Removed from the eligible map. Limitation: a genuine dense calcification on the skin is also removed – flagged for review instead of silent discard in the integrated tool",
            ],
            ["Eligible map", "E = breast ∧ ¬(pectoral⊕3px ∪ skin ∪ nipple ∪ marker)", "—", "All mass and calcification detectors receive only E (normalised convolution is used so excluded pixels do not leak into blob responses)"],
          ]}
        />
      </Section>

      <Section title="Stage 3 – Mass localisation and segmentation" id="s3">
        <Callout>
          Method selection: the finding is a single large, bright, well-contrasted blob. YOLO needs trained weights (not available here) and gives boxes only; U-Net needs pixel labels. Region growing and watershed leak into dense glandular tissue
          whose intensity is close to the mass; active contours need a good initial curve. So: <b>scale-space blob localisation (candidate) → local 2-class Fuzzy C-Means (boundary) → morphological + curvature-flow surrogate refinement</b>, with YOLO11s as an optional second proposal source.
        </Callout>
        <p>
          <b>Candidate localisation.</b> On an image down-sampled to ≈200 px: <M>D_s = G_s * Ñ − G_(1.6 s) * Ñ</M> (scale-normalised LoG approximation, blob radius r = √2·s), where <M>G*Ñ = (G*(N·E)) / (G*E)</M> is normalised convolution over the eligible map.
          σ_s = {"{0.9, 1.3, 1.9, 2.7, 3.9, 5.6, 8, 11.5}"}% of the longer side. Robust z-score <M>Z_s = (D_s − median)/(1.4826·MAD)</M> over eligible pixels. Keep 3×3×3 maxima with Z ≥ 4 whose disc is ≥ 70% eligible; NMS at 0.7·max(r₁,r₂); top 6.
        </p>
        <p>
          <b>Boundary.</b> Window 2.2·r. 2-class FCM on Gaussian(N,1.5) of eligible window pixels (≤20k samples), <M>J = ΣΣ u_ik^m (v_i − c_k)²</M>, m = 2, ε = 1e-3, 40 iterations,
          <M> u_ik = 1/Σ_j (|v_i−c_k|/|v_i−c_j|)^(2/(m−1))</M>, init centres P20/P85, rejected if c₂ − c₁ &lt; 8. Mask = u_bright &gt; 0.5 ∩ breast ∩ disc(1.7 r) → open r = max(2, 0.06 r) → close r = max(2, 0.1 r) → keep component under the blob centre → fill holes
          → Gaussian(σ = max(1.5, 0.04 r)) of the indicator &gt; 0.5 (mean-curvature-flow surrogate; a real morphological Chan–Vese pass can replace it) → ∩ eligible, largest component.
        </p>
        <Table
          head={["Feature", "Definition", "Gate / use"]}
          rows={[
            ["Area, equivalent diameter", "A, 2√(A/π)", "0.2% ≤ A/A_breast ≤ 35%"],
            ["Axis ratio", "minor/major from second central moments (4√λ)", "≥ 0.35 (below: vessel, duct, fold)"],
            ["Solidity, circularity", "A/A_hull; 4πA/P² (P from exposed pixel edges ×π/4)", "solidity ≥ 0.6; shape descriptor"],
            ["Contrast ΔN", "mean_in − mean_ring (ring = 12% of radius, outside the mask, eligible only)", "≥ 18 grey levels (0–255 scale)"],
            ["Margin sharpness", "mean |∇Ñ| on contour / (ΔN/(σ√2π))", "<0.3 indistinct; ≥0.45 circumscribed-like"],
            ["Radial roughness", "std(r(θ) − smooth₅(r(θ)))/mean r over 36 angular bins (median radius per bin)", ">0.12 with low sharpness → ‘possibly spiculated’; >0.06 → ‘possibly lobulated’"],
            ["Internal lucent fraction", "pixels filled by hole-filling / area", "descriptor (flags fat-/fluid-like internal lucency; never a diagnosis)"],
            ["Excluded overlap", "fraction of the region lying in excluded anatomy", "> 40% → rejected (pectoral/skin/nipple artefact)"],
            ["Confidence", "(min(1,Z/8) · min(1,ΔN/40) · min(1,ratio/0.6))^(1/3)", "Un-calibrated ranking score, not a probability"],
          ]}
        />
        <p>
          <b>I/O.</b> Input: float32 N (H×W), eligible mask uint8. Output per mass: uint8 mask (H×W), centroid, bbox, feature dict, descriptors. <b>FP reduction</b>: eligible-map restriction, disc-eligibility test, axis-ratio/solidity/contrast gates,
          excluded-overlap gate, ‘inside an already segmented mass’ suppression and logging of every rejected candidate with the reason. A <i>learned</i> second stage is in Stage 5.
        </p>
      </Section>

      <Section title="Stage 4 – Calcification detection" id="s4">
        <Code>{`S  = Gaussian(N, 0.7)
T  = max_{r in {2,3,5}}  [ S - open_r(S) ]          # white top-hat, disk SE (multi-scale)
μ,σ = box statistics of T, window 49x49
Z  = (T - μ) / (σ + 1)                              # locally adaptive, handles dense/noisy zones
low  = Z >= 0.5 kz  and T >= 0.5 tabs               # hysteresis (kz=4.5, tabs=6)
keep component if max Z >= kz and max T >= tabs
gates: 2 <= area <= 80 px; elongation (major/minor) <= 3.5 (area >= 4); not in excluded map
if #plausible > 300: kz += 0.5 (≤ 8 times)           # noise guard, reported in the log
cluster: DBSCAN eps = 5 mm / pixel_mm  (unknown spacing: 0.04*max(H,W)), minPts = 5 -> convex hull
isolated candidates are reported as isolated, never as clusters`}</Code>
        <ul className="ml-5 list-disc space-y-1">
          <li>
            <b>Vessels / trabeculae / skin lines</b>: elongated, so rejected by the elongation gate; the top-hat radius (≤5 px SE) removes anything wider. <b>Glandular tissue / mass / pectoral</b>: wider than the SE → suppressed by the opening, and the mask excludes them.
            <b> Noise</b>: Z is relative to the local top-hat distribution; hysteresis requires a strong seed. <b>Calcifications inside a mass</b> are kept but flagged <M>inside_mass</M>.
          </li>
          <li>
            <b>Why no per-pixel “bright” threshold</b>: bright pixels are mostly tissue; the top-hat + local Z makes salience relative to the immediate neighbourhood.
          </li>
          <li>Resolution dependence: SE radii and area limits are in pixels, so they must be rescaled as r_px = r_mm / pixel_mm (e.g. SE 0.15–0.5 mm; at 70 µm → r ≈ 2–7 px).</li>
          <li>Learned stage (when data exist): Random Forest / gradient boosting on candidate features (below) or a patch CNN; evaluated per candidate and per cluster.</li>
        </ul>
      </Section>

      <Section title="Stage 5 – Candidate classification and false-positive reduction" id="s5">
        <Table
          head={["Class", "Discriminating features", "Rule gate (implemented)", "Learned model (needs data)"]}
          rows={[
            ["True mass candidate", "blob Z, ΔN, solidity, margin sharpness/roughness, GLCM contrast/homogeneity/energy/correlation, heterogeneity", "area, axis ratio, solidity, ΔN, excluded-overlap", "Calibrated gradient boosting on features; or YOLO11s score fused (Integration); patient-level GroupKFold"],
            ["Normal glandular tissue", "Low ΔN, low sharpness, low solidity, large area, texture similar to surroundings", "ΔN < 18, solidity < 0.6, area > 35%", "Same classifier; add local-density-normalised texture"],
            ["Vessel / duct", "Elongation (minor/major < 0.35), constant width, ridge strength (Hessian eigen-ratio), curvilinear continuity", "axis ratio gate; calcification elongation gate", "Frangi/ridge features + classifier"],
            ["Microcalcification", "Peak top-hat, local Z, area 2–80, circularity, nearest-neighbour distance, local density within 5 mm, ridge strength", "Hysteresis Z, area, elongation", "Random Forest on candidate vectors; cluster-level features (count, hull area, linearity, pleomorphism)"],
            ["Noise", "Z just above threshold, 1-px area, no spatial neighbours", "area ≥ 2, hysteresis seed, noise guard", "—"],
            ["Nipple structures", "Located at the boundary apex; in-profile test", "Apex disc excluded", "—"],
            ["Pectoral muscle", "Oblique, long, brighter-than-neighbour step to the chest wall, homogeneous texture", "Line + contrast validation", "U-Net / nnU-Net (best when edge is curved)"],
          ]}
        />
      </Section>
    </Page>
  );
}
