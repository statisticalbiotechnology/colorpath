"""
saturation.py — detector-saturation ceiling detection and mask construction.

High-abundance ions saturate the detector; the clipped intensity then reappears in
components 2/3 of a naive factorisation as spurious "compensation". gait handles
this by treating saturated ``(pixel, ion)`` entries as *right-censored* via a binary
weight matrix ``W in {0, 1}^{P x M}`` passed to the masked NMF (see ``nmf_linear``).

Two mechanisms are distinguished (CLAUDE.md §Diagnostics):

* **Hard clipping** — a pile-up spike at a per-ion ceiling. Mask those entries.
* **Soft compression** — a smoothly bending tail with no spike. Masking optional;
  Route 1's ``asinh`` transform absorbs this instead.

Detection defaults to *off*: :func:`build_saturation_mask` returns an all-ones weight
matrix unless a ceiling is supplied or auto-detection is explicitly requested, and a
warning is emitted when a pile-up is detected so the user can opt in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class SaturationReport:
    """Per-ion saturation findings."""

    ceilings: np.ndarray            # (M,) detected/assigned ceiling per ion (inf = none)
    pile_up_fraction: np.ndarray    # (M,) fraction of pixels in the top histogram bin
    flagged: np.ndarray             # (M,) bool, True where hard clipping is suspected
    n_masked: int                   # total censored (pixel, ion) entries


def detect_saturation_ceiling(
    x: np.ndarray,
    n_bins: int = 50,
    pile_up_threshold: float = 0.05,
) -> tuple[float, float]:
    """Detect a hard-clipping ceiling for a single ion's intensities.

    A pile-up spike in the *top* histogram bin (relative to its neighbours) indicates
    hard clipping at the detector maximum. Returns ``(ceiling, pile_up_fraction)``;
    ``ceiling`` is ``inf`` when no clipping is detected.

    Parameters
    ----------
    x                 : (P,) intensities for one ion across pixels.
    n_bins            : histogram resolution.
    pile_up_threshold : minimum fraction of pixels in the top bin (and excess over the
                        second-to-last bin) to declare hard clipping.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0 or finite.max() <= finite.min():
        return np.inf, 0.0

    counts, edges = np.histogram(finite, bins=n_bins)
    frac = counts / counts.sum()
    top = frac[-1]
    # Reference level = median of the interior bins (robust to the tail).
    interior = frac[:-1]
    ref = np.median(interior[interior > 0]) if np.any(interior > 0) else 0.0
    pile_up = top >= pile_up_threshold and top > 3 * ref
    if pile_up:
        # Censor at the lower edge of the pile-up bin.
        return float(edges[-2]), float(top)
    return np.inf, float(top)


def build_saturation_mask(
    X: np.ndarray,
    ceilings: np.ndarray | float | None = None,
    saturation_quantile: float | None = None,
    auto_detect: bool = False,
    pile_up_threshold: float = 0.05,
    warn: bool = True,
) -> tuple[np.ndarray, SaturationReport]:
    """Build a binary weight matrix censoring saturated entries.

    Parameters
    ----------
    X                   : (P, M) non-negative data (pixels x ions).
    ceilings            : explicit per-ion ceiling(s); entries ``>= ceiling`` are masked.
                          Scalar broadcasts to all ions. Overrides other detection.
    saturation_quantile : if given (e.g. 0.999), mask entries above this per-ion quantile.
    auto_detect         : if True, run histogram pile-up detection per ion and mask
                          flagged ions at their detected ceiling.
    pile_up_threshold   : sensitivity of pile-up detection.
    warn                : emit a warning when a pile-up is detected but not masked.

    Returns
    -------
    (W, report) : ``W`` is (P, M) in {0, 1} (1 = keep). When no censoring applies, ``W``
    is all ones (the masked loss then reduces to the unmasked loss).
    """
    X = np.asarray(X, dtype=float)
    P, M = X.shape
    W = np.ones_like(X)

    # Always profile for the report / warning.
    detected = np.full(M, np.inf)
    pile = np.zeros(M)
    for j in range(M):
        c, f = detect_saturation_ceiling(X[:, j], pile_up_threshold=pile_up_threshold)
        detected[j], pile[j] = c, f
    flagged = np.isfinite(detected)

    if ceilings is not None:
        ceil = np.broadcast_to(np.asarray(ceilings, dtype=float), (M,)).copy()
    elif saturation_quantile is not None:
        ceil = np.quantile(X, saturation_quantile, axis=0)
    elif auto_detect:
        ceil = detected.copy()
    else:
        ceil = np.full(M, np.inf)
        if warn and np.any(flagged):
            warnings.warn(
                f"Detected likely hard-clipping pile-up in {int(flagged.sum())} ion(s) "
                f"but masking is off. Pass auto_detect=True, saturation_quantile=, or "
                f"explicit ceilings= to censor saturated entries.",
                stacklevel=2,
            )

    for j in range(M):
        if np.isfinite(ceil[j]):
            W[:, j] = (X[:, j] < ceil[j]).astype(float)

    report = SaturationReport(
        ceilings=ceil,
        pile_up_fraction=pile,
        flagged=flagged,
        n_masked=int((W == 0).sum()),
    )
    return W, report
