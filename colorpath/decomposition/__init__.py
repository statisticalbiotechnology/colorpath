"""
colorpath.decomposition — pathway-activity decomposition engine for imaging MS.

Factorises a non-negative IMS matrix ``X (P pixels x M ions) ~= U V`` into K rank-1
components, each interpreted as a pathway:

    V[k, :]  pathway activity graph  (which metabolites load, how strongly)
    U[:, k]  pathway activity image  (where in the tissue the pathway is active)

These two vectors per component feed the existing colorpath illustration layer
(``colorpath.illustration``).

Two complementary routes resolve the error-model vs coupling-model tension (see the
project CLAUDE.md):

* **Route 2 (primary)** — :class:`LinearNMF`: stay in linear space (so ``UV`` *is* the
  multiplicative pathway coupling) and obtain the multiplicative-error property from a
  scale-invariant Itakura-Saito (or KL) loss, with a masked loss for detector
  saturation.
* **Route 1 (secondary)** — :class:`LogLevelNMF`: factorise ``asinh(X/c)`` with each
  loading constrained toward sparse, equal-level membership; multiplicative coupling
  reappears as an additive equal-loading component in transformed space.
"""

from .losses import LOSSES, frobenius_loss, is_loss, kl_loss, get_loss
from .nmf_linear import LinearNMF, LinearNMFResult, fit_linear_nmf
from .nmf_independent import (
    IndependentNMF,
    IndependentNMFResult,
    fit_independent_nmf,
    normalised_hsic_matrix,
)
from .nmf_loglevel import (
    LogLevelNMF,
    LogLevelNMFResult,
    asinh_transform,
    fit_loglevel_nmf,
    inverse_asinh,
)
from .saturation import (
    SaturationReport,
    build_saturation_mask,
    detect_saturation_ceiling,
)
from .diagnostics import (
    VarianceMeanResult,
    compensation_artifact_check,
    component_recovery,
    variance_vs_mean,
)
from .selection import KSelectionResult, select_k
from .contributions import variation_explained, spatial_variation_explained

__all__ = [
    # losses
    "LOSSES", "frobenius_loss", "is_loss", "kl_loss", "get_loss",
    # Route 2
    "LinearNMF", "LinearNMFResult", "fit_linear_nmf",
    # Route 2 + ICA-style independence
    "IndependentNMF", "IndependentNMFResult", "fit_independent_nmf",
    "normalised_hsic_matrix",
    # Route 1
    "LogLevelNMF", "LogLevelNMFResult", "fit_loglevel_nmf",
    "asinh_transform", "inverse_asinh",
    # saturation
    "SaturationReport", "build_saturation_mask", "detect_saturation_ceiling",
    # diagnostics
    "VarianceMeanResult", "variance_vs_mean",
    "compensation_artifact_check", "component_recovery",
    # selection
    "KSelectionResult", "select_k",
    # interpretation / illustration scaling
    "variation_explained", "spatial_variation_explained",
]
