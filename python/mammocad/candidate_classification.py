"""Stage 5 - false-positive reduction.  Rule gates (always on) + optional learned classifiers.

No trained model ships with this repository: training needs annotated data (INbreast, CBIS-DDSM,
VinDr-Mammo, or in-house).  The functions below define the feature vectors and the training/inference API."""
from __future__ import annotations
import numpy as np
from skimage.feature import graycomatrix, graycoprops

MASS_FEATURES = ["area", "equiv_diam", "axis_ratio", "circularity", "solidity", "contrast", "heterogeneity",
                 "margin_sharpness", "radial_roughness", "blob_z", "glcm_contrast", "glcm_homogeneity",
                 "glcm_energy", "glcm_correlation", "dist_to_skin", "dist_to_pectoral"]
CALC_FEATURES = ["area", "peak_tophat", "peak_z", "elongation", "circularity", "local_density_5mm",
                 "nn_dist", "dist_to_skin", "ridge_strength", "inside_mass"]


def glcm_features(N, mask, levels: int = 32):
    """GLCM on the lesion bbox quantised to `levels` grey levels; d=1,3 ; angles 0,45,90,135; averaged."""
    ys, xs = np.nonzero(mask)
    crop = N[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = np.clip((crop / 256 * levels).astype(np.uint8), 0, levels - 1)
    g = graycomatrix(q, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=True, normed=True)
    return {f"glcm_{p}": float(graycoprops(g, p).mean()) for p in ("contrast", "homogeneity", "energy", "correlation")}


def rule_gate_mass(f: dict, breast_area: int) -> tuple[bool, str]:
    if f["area"] < 0.002 * breast_area: return False, "too small"
    if f["area"] > 0.35 * breast_area: return False, "too large (parenchymal gradient)"
    if f["axis_ratio"] < 0.35: return False, "elongated (vessel/duct/fold)"
    if f["solidity"] < 0.6: return False, "tissue pattern"
    if f["contrast"] < 18: return False, "low contrast (normal glandular tissue)"
    return True, "ok"


def train_mass_classifier(X, y):
    """Gradient boosting on handcrafted features; calibrated; evaluate with patient-level GroupKFold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=300,
                                  class_weight="balanced"), method="isotonic", cv=5).fit(X, y)


def train_calc_classifier(X, y):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1).fit(X, y)
