"""
nmf_loglevel.py — Route 1 (SECONDARY): equal-loading constrained NMF in asinh space.

Route 2 keeps multiplicative coupling exact but fixes the pathway response exponent at 1
(metabolites A and B scale identically with activity). Route 1 is the better choice when
the deliverable is crisp, near-binary **pathway membership** and we accept a log-style
error model.

Key reframing (CLAUDE.md §Route 1): for a rank-1 pathway ``A = a*s``, ``B = b*s``,

    log A = log a + log s ,   log B = log b + log s

so multiplicative coupling in linear space becomes an **additive, equal-loading**
rank-1 component in log space: the spectral loading is constant over member metabolites
and the spatial score is ``log s``. We therefore:

* transform with ``asinh(X / c)`` (defined at 0, linear near the origin, logarithmic for
  large x; absorbs soft saturation) instead of plain log, and
* factorise ``Y = asinh(X/c) ~= sum_k g_k (x) p_k`` while **constraining each loading
  ``p_k`` toward a sparse, equal-level (near-binary) membership vector** via an
  elastic-net penalty plus a within-component equalisation step.

Initialise from a Route 2 solution so the two routes share component identity and can be
compared component-by-component.

Loadings are on the transformed (asinh) scale, *not* linear abundance — document this
when handing ``p`` to the illustration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .losses import EPS, frobenius_loss


def asinh_transform(X: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Variance-stabilising ``asinh(X / c)`` transform (non-negative for X >= 0)."""
    return np.arcsinh(np.asarray(X, dtype=float) / c)


def inverse_asinh(Y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Inverse of :func:`asinh_transform`."""
    return np.sinh(np.asarray(Y, dtype=float)) * c


@dataclass
class LogLevelNMFResult:
    """Container for a fitted Route 1 model."""

    g: np.ndarray                       # (P, K) log/asinh-activity images
    p: np.ndarray                       # (K, M) membership loadings (transformed scale)
    c: float                            # asinh scale used
    loss_curve: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False

    @property
    def K(self) -> int:
        return self.g.shape[1]

    def multiplicative_activity(self) -> np.ndarray:
        """Recover the (approximate) multiplicative activity image ``sinh(g) * c``."""
        return inverse_asinh(self.g, self.c)

    def membership(self, threshold: float = 0.5) -> np.ndarray:
        """Binary membership matrix: ``p_k`` entries above ``threshold * row_max``."""
        rmax = self.p.max(axis=1, keepdims=True)
        rmax[rmax == 0] = 1.0
        return (self.p >= threshold * rmax).astype(int)

    def component(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.g[:, k], self.p[k, :]


def _equalise(p_row: np.ndarray, alpha: float, active_frac: float) -> np.ndarray:
    """Shrink a loading row's active (member) entries toward their shared mean.

    Active = entries above ``active_frac * row_max``. ``alpha`` in [0, 1] is the
    equalisation strength; alpha=0 leaves the row unchanged, alpha=1 sets every active
    entry to the common mean (a hard equal-loading / binary support).
    """
    if alpha <= 0 or p_row.max() <= 0:
        return p_row
    active = p_row >= active_frac * p_row.max()
    if active.sum() == 0:
        return p_row
    mean_active = p_row[active].mean()
    out = p_row.copy()
    out[active] = (1 - alpha) * p_row[active] + alpha * mean_active
    return out


class LogLevelNMF:
    """asinh-space equal-loading constrained NMF (colorpath Route 1).

    Parameters
    ----------
    n_components  : K.
    c             : asinh scale (set near the noise level / small-signal cut).
    l1, l2        : elastic-net penalties on the loadings ``p`` (sparsity, smoothing).
    equalise      : within-component equalisation strength in [0, 1] (drives members to a
                    shared level -> equal-loading membership).
    active_frac   : fraction of a row's max above which an entry counts as a member.
    max_iter, tol : optimisation controls.
    random_state  : seed for random initialisation when no warm start is given.
    """

    def __init__(
        self,
        n_components: int,
        c: float = 1.0,
        l1: float = 0.1,
        l2: float = 0.0,
        equalise: float = 0.3,
        active_frac: float = 0.25,
        max_iter: int = 500,
        tol: float = 1e-5,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.c = c
        self.l1 = l1
        self.l2 = l2
        self.equalise = equalise
        self.active_frac = active_frac
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        init: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LogLevelNMFResult:
        """Factorise ``asinh(X/c) ~= g p`` with equal-loading constrained ``p``.

        Parameters
        ----------
        X    : (P, M) non-negative data.
        init : optional ``(g0, p0)`` warm start. Pass a Route 2 ``(U, V)`` here to share
               component identity; it is used as-is (non-negativity assumed).
        """
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("LogLevelNMF requires non-negative data")
        Y = asinh_transform(X, self.c)
        P, M = Y.shape
        K = self.n_components

        if init is not None:
            g = np.maximum(np.asarray(init[0], dtype=float).copy(), EPS)
            p = np.maximum(np.asarray(init[1], dtype=float).copy(), EPS)
        else:
            rng = np.random.default_rng(self.random_state)
            scale = np.sqrt(np.mean(Y) / K) if Y.mean() > 0 else 1.0
            g = np.maximum(rng.random((P, K)) * scale, EPS)
            p = np.maximum(rng.random((K, M)) * scale, EPS)

        loss_curve = [self._penalised_loss(Y, g, p)]
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # ---- penalised multiplicative update for p (loadings) ----
            gp = np.maximum(g @ p, EPS)
            num = g.T @ Y
            den = g.T @ gp + self.l1 + self.l2 * p
            p = p * (num / np.maximum(den, EPS))
            p = np.maximum(p, EPS)
            # equalisation toward shared member level (the equal-loading constraint)
            for k in range(K):
                p[k] = _equalise(p[k], self.equalise, self.active_frac)

            # ---- multiplicative update for g (activity images) ----
            gp = np.maximum(g @ p, EPS)
            num = Y @ p.T
            den = gp @ p.T
            g = g * (num / np.maximum(den, EPS))
            g = np.maximum(g, EPS)

            cur = self._penalised_loss(Y, g, p)
            loss_curve.append(cur)
            prev = loss_curve[-2]
            if prev > 0 and abs(prev - cur) / prev < self.tol:
                converged = True
                break

        # Resolve scale ambiguity: normalise each loading to unit max, push scale to g.
        for k in range(K):
            m = p[k].max()
            if m > 0:
                p[k] /= m
                g[:, k] *= m

        return LogLevelNMFResult(
            g=g, p=p, c=self.c, loss_curve=loss_curve, n_iter=n_iter, converged=converged
        )

    def _penalised_loss(self, Y, g, p) -> float:
        data = frobenius_loss(Y, g @ p)
        pen = self.l1 * np.abs(p).sum() + 0.5 * self.l2 * (p**2).sum()
        return float(data + pen)


def fit_loglevel_nmf(X, n_components, init=None, **kwargs) -> LogLevelNMFResult:
    """Functional shortcut for :class:`LogLevelNMF`. ``init`` is a Route 2 ``(U, V)``."""
    return LogLevelNMF(n_components, **kwargs).fit(X, init=init)
