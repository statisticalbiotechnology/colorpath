"""
nmf_linear.py — Route 2 (PRIMARY): masked NMF in linear space.

This is the core gait decomposition engine. It factorises a non-negative
imaging-mass-spectrometry matrix

    X  in  R>=0^{P x M}        P pixels (rows), M metabolites/ions (columns)
    X ~= U V                   U in R>=0^{P x K}   spatial scores  (pathway images)
                               V in R>=0^{K x M}   spectral loadings (pathway graphs)

*in linear space* so that each rank-1 component ``U[:, k] (x) V[k, :]`` is literally the
multiplicative pathway coupling gait models (requirement R2 in CLAUDE.md). The
multiplicative-error property (R1) comes from the *loss*, not from a log transform:

    loss="is"   Itakura-Saito  -> multiplicative Gamma gain noise (variance ∝ mean^2)
    loss="kl"   generalised KL -> Poisson / ion-counting shot noise (variance ∝ mean)
    loss="frobenius"           -> additive Gaussian noise (the conventional baseline)

Pick ``is`` vs ``kl`` empirically with the variance-vs-mean diagnostic
(:func:`gait.decomposition.diagnostics.variance_vs_mean`).

Detector saturation is handled by a binary weight matrix ``W in {0, 1}^{P x M}`` that
zeroes out saturated ``(pixel, ion)`` entries (right-censoring). The mask is carried
*inside both the numerator and denominator* of every multiplicative update, so masked
entries influence neither the fit nor the normalisation.

The optimiser uses multiplicative-update (MU) rules (Lee-Seung for Frobenius/KL,
Fevotte-Bertin-Durrieu for IS). IS-NMF is non-convex and MU can be unstable, so by
default we **warm-start IS from a few KL iterations**, floor every quantity by ``EPS``,
and support best-of ``n_init`` restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .losses import EPS, get_loss


@dataclass
class LinearNMFResult:
    """Container for a fitted Route 2 model, fed directly to the illustration layer."""

    U: np.ndarray                       # (P, K) spatial scores  -> pathway activity images
    V: np.ndarray                       # (K, M) spectral loadings -> pathway activity graphs
    loss: str                           # divergence used
    loss_curve: list[float]             # objective value per iteration (best init)
    n_iter: int                         # iterations run for the winning init
    converged: bool                     # whether tol was reached
    mask: np.ndarray | None = None      # (P, K) weight matrix actually used (or None)
    reconstruction_error: float = np.nan  # final (masked) divergence value
    init_errors: list[float] = field(default_factory=list)  # final loss of each restart

    @property
    def K(self) -> int:
        return self.U.shape[1]

    def reconstruct(self) -> np.ndarray:
        """Return the rank-K reconstruction ``U @ V``."""
        return self.U @ self.V

    def component(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(U[:, k], V[k, :])`` for component ``k`` (image, graph)."""
        return self.U[:, k], self.V[k, :]


def _init_factors(X, K, rng, mask=None):
    """Non-negative random initialisation scaled to the data mean (NNDSVD-lite)."""
    P, M = X.shape
    if mask is not None:
        scale = np.sqrt(np.mean(X[mask > 0]) / K) if np.any(mask > 0) else 1.0
    else:
        scale = np.sqrt(np.mean(X) / K) if X.mean() > 0 else 1.0
    U = np.maximum(rng.random((P, K)) * scale, EPS)
    V = np.maximum(rng.random((K, M)) * scale, EPS)
    return U, V


def _mu_updates(X, U, V, loss, W):
    """One masked multiplicative-update sweep (V then U). Returns updated (U, V).

    With weight matrix ``W`` (ones if unmasked) the rules are:

        frobenius : V *= (U^T (W*X))            / (U^T (W*(UV)))
                    U *= ((W*X) V^T)            / ((W*(UV)) V^T)
        kl        : V *= (U^T (W*X/(UV)))       / (U^T W)
                    U *= ((W*X/(UV)) V^T)       / (W V^T)
        is        : V *= (U^T (W*X/(UV)^2))     / (U^T (W/(UV)))
                    U *= ((W*X/(UV)^2) V^T)     / ((W/(UV)) V^T)
    """
    ones = W if W is not None else None

    def w(A):  # apply mask if present
        return A if ones is None else ones * A

    # ---- update V ----
    UV = np.maximum(U @ V, EPS)
    if loss == "frobenius":
        num = U.T @ w(X)
        den = U.T @ w(UV)
    elif loss == "kl":
        num = U.T @ w(X / UV)
        den = U.T @ (ones if ones is not None else np.ones_like(X))
    else:  # is
        num = U.T @ w(X / UV**2)
        den = U.T @ w(1.0 / UV)
    V = V * (num / np.maximum(den, EPS))
    V = np.maximum(V, EPS)

    # ---- update U ----
    UV = np.maximum(U @ V, EPS)
    if loss == "frobenius":
        num = w(X) @ V.T
        den = w(UV) @ V.T
    elif loss == "kl":
        num = w(X / UV) @ V.T
        den = (ones if ones is not None else np.ones_like(X)) @ V.T
    else:  # is
        num = w(X / UV**2) @ V.T
        den = w(1.0 / UV) @ V.T
    U = U * (num / np.maximum(den, EPS))
    U = np.maximum(U, EPS)

    return U, V


def _fit_single(X, K, loss, W, max_iter, tol, rng, warm_start_iter, init_UV):
    """Run MU from one initialisation; return (U, V, loss_curve, n_iter, converged)."""
    loss_fn = get_loss(loss)

    if init_UV is not None:
        U, V = init_UV[0].copy(), init_UV[1].copy()
    else:
        U, V = _init_factors(X, K, rng, W)
        # Warm-start the non-convex IS objective with a few KL sweeps.
        if loss == "is" and warm_start_iter > 0:
            for _ in range(warm_start_iter):
                U, V = _mu_updates(X, U, V, "kl", W)

    loss_curve = [loss_fn(X, U @ V, W)]
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        U, V = _mu_updates(X, U, V, loss, W)
        cur = loss_fn(X, U @ V, W)
        loss_curve.append(cur)
        prev = loss_curve[-2]
        if prev > 0 and abs(prev - cur) / prev < tol:
            converged = True
            break
    return U, V, loss_curve, n_iter, converged


class LinearNMF:
    """Masked IS / KL / Frobenius NMF in linear space (gait Route 2).

    Parameters
    ----------
    n_components : K, the number of pathway components.
    loss         : one of ``"is"`` (default), ``"kl"``, ``"frobenius"``.
    max_iter     : maximum MU sweeps per initialisation.
    tol          : stop when the relative change in the loss falls below this.
    n_init       : number of random restarts; the lowest-loss fit is kept.
    warm_start_iter : KL sweeps used to initialise an IS fit (ignored otherwise).
    random_state : seed for reproducible initialisation.

    Notes
    -----
    Outputs ``U`` (pathway activity images) and ``V`` (pathway activity graphs) are on
    interpretable linear-abundance units, ready for the existing gait illustration
    layer (see :func:`gait.illustration.bridge.illustrate_component`).
    """

    def __init__(
        self,
        n_components: int,
        loss: str = "is",
        max_iter: int = 500,
        tol: float = 1e-5,
        n_init: int = 1,
        warm_start_iter: int = 10,
        random_state: int | None = None,
    ):
        get_loss(loss)  # validate eagerly
        self.n_components = n_components
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.warm_start_iter = warm_start_iter
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        mask: np.ndarray | None = None,
        init: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LinearNMFResult:
        """Factorise ``X ~= U V``.

        Parameters
        ----------
        X    : (P, M) non-negative data matrix (pixels x ions).
        mask : optional (P, M) weight matrix; binary {0,1} censors saturated entries.
               See :mod:`gait.decomposition.saturation` for construction.
        init : optional ``(U, V)`` warm start (e.g. a previous fit). When given,
               ``n_init`` is forced to 1.
        """
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("LinearNMF requires non-negative data (linear space)")
        if mask is not None:
            mask = np.asarray(mask, dtype=float)
            if mask.shape != X.shape:
                raise ValueError("mask must match X shape")

        n_init = 1 if init is not None else self.n_init
        best = None
        init_errors: list[float] = []
        for i in range(n_init):
            rng = np.random.default_rng(
                None if self.random_state is None else self.random_state + i
            )
            U, V, curve, n_iter, conv = _fit_single(
                X, self.n_components, self.loss, mask,
                self.max_iter, self.tol, rng, self.warm_start_iter, init,
            )
            final = curve[-1]
            init_errors.append(final)
            if best is None or final < best[2][-1]:
                best = (U, V, curve, n_iter, conv)

        U, V, curve, n_iter, conv = best
        return LinearNMFResult(
            U=U, V=V, loss=self.loss, loss_curve=curve, n_iter=n_iter,
            converged=conv, mask=mask, reconstruction_error=curve[-1],
            init_errors=init_errors,
        )


def fit_linear_nmf(X, n_components, **kwargs) -> LinearNMFResult:
    """Functional shortcut: ``LinearNMF(n_components, ...).fit(X, mask=...)``.

    ``mask`` and ``init`` are forwarded to :meth:`LinearNMF.fit`; all other keyword
    arguments configure the estimator.
    """
    mask = kwargs.pop("mask", None)
    init = kwargs.pop("init", None)
    return LinearNMF(n_components, **kwargs).fit(X, mask=mask, init=init)
