"""
losses.py — divergences for non-negative matrix factorisation, with masked variants.

These implement the three loss functions referenced by the gait decomposition
engine (see CLAUDE.md §"Route 2"):

    frobenius : D(x, y) = (x - y)^2 / 2          (additive Gaussian error)
    kl        : D(x, y) = x log(x/y) - x + y     (Poisson / shot / counting noise)
    is        : D(x, y) = x/y - log(x/y) - 1     (multiplicative Gamma gain noise)

Each accepts an optional non-negative weight matrix ``W`` of the same shape that is
multiplied elementwise into the per-entry divergence before summation. A binary
``W in {0, 1}`` implements the masked / right-censored loss used for detector
saturation: saturated ``(pixel, ion)`` entries are given weight 0 so they do not
contribute to the fit.

The IS divergence is *scale invariant* — ``D_IS(λx, λy) = D_IS(x, y)`` — which is the
property gait wants for multiplicative measurement error. KL has variance ∝ mean
(shot noise); Frobenius has constant variance (the conventional, and for IMS usually
wrong, assumption).
"""

from __future__ import annotations

import numpy as np

# Small floor to keep divisions and logs finite. Applied to reconstructions and to
# data where a ratio/log is taken; it does not change the value of well-scaled entries.
EPS = 1e-9


def _prepare(X: np.ndarray, Y: np.ndarray, W: np.ndarray | None):
    X = np.asarray(X, dtype=float)
    Y = np.maximum(np.asarray(Y, dtype=float), EPS)
    if W is None:
        return X, Y, None
    W = np.asarray(W, dtype=float)
    if W.shape != X.shape:
        raise ValueError(f"weight matrix W{W.shape} must match data X{X.shape}")
    return X, Y, W


def frobenius_loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray | None = None) -> float:
    """Half squared Frobenius distance ``0.5 * sum W (X - Y)^2``."""
    X, Y, W = _prepare(X, Y, W)
    d = 0.5 * (X - Y) ** 2
    if W is not None:
        d = W * d
    return float(d.sum())


def kl_loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray | None = None) -> float:
    """Generalised KL (Poisson) divergence ``sum W (x log(x/y) - x + y)``."""
    X, Y, W = _prepare(X, Y, W)
    Xc = np.maximum(X, 0.0)
    # x log(x/y) -> 0 as x -> 0; guard the log against x = 0.
    xlog = np.where(Xc > 0, Xc * np.log(np.maximum(Xc, EPS) / Y), 0.0)
    d = xlog - Xc + Y
    if W is not None:
        d = W * d
    return float(d.sum())


def is_loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray | None = None) -> float:
    """Itakura-Saito divergence ``sum W (x/y - log(x/y) - 1)`` (scale invariant)."""
    X, Y, W = _prepare(X, Y, W)
    Xc = np.maximum(X, EPS)
    ratio = Xc / Y
    d = ratio - np.log(ratio) - 1.0
    if W is not None:
        d = W * d
    return float(d.sum())


#: Public registry mapping the ``loss=`` keyword used across the decomposition API to
#: the scalar divergence function. The matching multiplicative-update rules live in
#: :mod:`gait.decomposition.nmf_linear`.
LOSSES = {
    "frobenius": frobenius_loss,
    "kl": kl_loss,
    "is": is_loss,
}


def get_loss(name: str):
    """Look up a divergence by name, raising a helpful error for typos."""
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(
            f"unknown loss {name!r}; choose one of {sorted(LOSSES)}"
        ) from None
