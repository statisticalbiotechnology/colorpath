"""
diagnostics.py — checks to run before trusting a decomposition (CLAUDE.md §Diagnostics).

1. variance_vs_mean      : choose the Route 2 loss (variance ∝ mean^2 -> IS;
                           variance ∝ mean -> KL).
2. compensation_artifact : confirm a saturating ion's reappearance in later components
                           is a measurement nonlinearity, not biology.
3. component_recovery     : confirm masking/IS collapses the spurious compensating
                           component relative to a Frobenius baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VarianceMeanResult:
    means: np.ndarray            # (M,) per-ion mean across pixels
    variances: np.ndarray        # (M,) per-ion variance across pixels
    slope: float                 # slope of log(var) vs log(mean)
    recommended_loss: str        # "is" (slope ~ 2), "kl" (slope ~ 1), else "frobenius"


def variance_vs_mean(X: np.ndarray, min_mean: float = 1e-6) -> VarianceMeanResult:
    """Fit ``log(variance) ~ slope * log(mean)`` across ions to pick the noise model.

    slope ≈ 2  -> variance ∝ mean^2 -> multiplicative Gamma noise -> ``loss="is"``.
    slope ≈ 1  -> variance ∝ mean    -> Poisson / shot noise       -> ``loss="kl"``.
    slope ≈ 0  -> constant variance  -> additive Gaussian          -> ``loss="frobenius"``.
    """
    X = np.asarray(X, dtype=float)
    means = X.mean(axis=0)
    variances = X.var(axis=0)
    ok = (means > min_mean) & (variances > 0)
    if ok.sum() < 2:
        return VarianceMeanResult(means, variances, np.nan, "frobenius")
    slope = float(np.polyfit(np.log(means[ok]), np.log(variances[ok]), 1)[0])
    if slope >= 1.5:
        rec = "is"
    elif slope >= 0.5:
        rec = "kl"
    else:
        rec = "frobenius"
    return VarianceMeanResult(means, variances, slope, rec)


def compensation_artifact_check(
    U: np.ndarray,
    ion_index: int,
    V: np.ndarray,
    primary_component: int = 0,
    suspect_components: tuple[int, ...] | None = None,
) -> dict:
    """Test whether a saturating ion's loading in later components is a clipping artifact.

    For a genuinely saturating ion, its spatial image in the *primary* component is
    clipped (flat-topped) exactly where intensity is highest; the ion then reappears in
    later components with a spatial pattern that is *anti-correlated* with that primary
    image (high where the primary clips). We report, per suspect component, the
    correlation between the suspect component's spatial image and the primary image,
    weighted by how strongly the ion loads on the suspect component.

    Returns a dict with per-component loading and spatial correlation; strongly negative
    correlation + non-trivial loading is evidence of a compensation artifact.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = U.shape[1]
    if suspect_components is None:
        suspect_components = tuple(k for k in range(K) if k != primary_component)

    primary_img = U[:, primary_component]
    out = {}
    for k in suspect_components:
        img = U[:, k]
        if primary_img.std() > 0 and img.std() > 0:
            corr = float(np.corrcoef(primary_img, img)[0, 1])
        else:
            corr = np.nan
        out[k] = {
            "ion_loading": float(V[k, ion_index]),
            "spatial_corr_with_primary": corr,
            "artifact_suspected": bool(
                corr < -0.3 and V[k, ion_index] > V[:, ion_index].mean()
            ),
        }
    return out


def component_recovery(
    baseline_V: np.ndarray,
    masked_V: np.ndarray,
    ion_index: int,
    primary_component: int = 0,
) -> dict:
    """Quantify how much masking/IS collapses an ion's spurious off-component loadings.

    Compares the fraction of an ion's total loading that sits *outside* its primary
    component, between a baseline (e.g. Frobenius, no mask) and the masked/IS fit. A
    drop indicates the compensating components were absorbed measurement nonlinearity.
    """
    def off_fraction(V):
        col = np.abs(V[:, ion_index])
        total = col.sum()
        if total == 0:
            return np.nan
        return float((total - col[primary_component]) / total)

    base = off_fraction(baseline_V)
    masked = off_fraction(masked_V)
    return {
        "baseline_off_component_fraction": base,
        "masked_off_component_fraction": masked,
        "collapsed": bool(np.isfinite(base) and np.isfinite(masked) and masked < base),
    }
