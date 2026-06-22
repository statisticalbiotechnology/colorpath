"""
selection.py — choosing the number of components K and best-of-n_init restarts.

K selection on IMS is unavoidably heuristic. We provide a reconstruction-error sweep
(masked divergence vs K) and an "elbow" pick via the maximum-curvature heuristic on the
normalised error curve. The same masked loss used for fitting is used for scoring, so
saturated entries do not drive K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nmf_linear import LinearNMF, LinearNMFResult


@dataclass
class KSelectionResult:
    ks: list[int]
    errors: list[float]
    best_k: int
    results: dict[int, LinearNMFResult]


def _elbow(ks, errors) -> int:
    """Pick K at the point of maximum curvature of the normalised error curve."""
    ks = np.asarray(ks, dtype=float)
    err = np.asarray(errors, dtype=float)
    if len(ks) < 3:
        return int(ks[int(np.argmin(err))])
    x = (ks - ks.min()) / (np.ptp(ks) or 1)
    y = (err - err.min()) / (np.ptp(err) or 1)
    # Distance from each point to the line joining the first and last points.
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    line = line / (np.linalg.norm(line) or 1)
    pts = np.column_stack([x - x[0], y - y[0]])
    proj = pts @ line
    dist = np.linalg.norm(pts - np.outer(proj, line), axis=1)
    return int(ks[int(np.argmax(dist))])


def select_k(
    X: np.ndarray,
    k_range,
    mask: np.ndarray | None = None,
    loss: str = "is",
    **nmf_kwargs,
) -> KSelectionResult:
    """Sweep ``k_range``, fitting a :class:`LinearNMF` per K, and pick the elbow.

    Extra keyword arguments (``n_init``, ``max_iter``, ``random_state`` ...) are passed
    to :class:`LinearNMF`. Returns every fitted result so callers can inspect them.
    """
    ks = list(k_range)
    errors, results = [], {}
    for k in ks:
        res = LinearNMF(k, loss=loss, **nmf_kwargs).fit(X, mask=mask)
        results[k] = res
        errors.append(res.reconstruction_error)
    best_k = _elbow(ks, errors)
    return KSelectionResult(ks=ks, errors=errors, best_k=best_k, results=results)
