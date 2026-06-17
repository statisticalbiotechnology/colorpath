"""
nmf_independent.py — IS/KL NMF with an ICA-style mutual-information penalty.

colorpath's Route 2 keeps the data in linear space (so ``UV`` is genuine multiplicative
pathway coupling, R2) and gets the multiplicative-error model from a scale-invariant
IS / KL loss (R1). But like any NMF it leaves the components *correlated*: nothing pushes
component 2, 3, ... to be statistically independent the way ICA does.

This module adds that independence while keeping both colorpath principles. We do **not**
whiten-and-rotate (which would destroy non-negativity and the multiplicative outer-product
structure); instead we keep the IS/KL-NMF fidelity term and add a penalty that targets
mutual information directly:

    minimise_{U,V >= 0}   D_IS/KL(X || UV)  +  lambda * sum_{i<j} HSIC(U[:,i], U[:,j])

The Hilbert-Schmidt Independence Criterion (HSIC) with a *characteristic* (RBF) kernel is
zero **iff** the two variables are independent (iff their mutual information is zero), so
minimising it drives the components toward MI-sense orthogonality. This is precisely the
kernel dependence contrast used by Kernel ICA (Bach & Jordan, 2002), here used as a
penalty on the non-negative spatial activity maps rather than as a rotation objective.
Plain second-order decorrelation (a *linear* kernel) is available as ``kernel="linear"``
but only removes correlation, not higher-order dependence.

Scalability: the RBF kernel and its gradient are approximated with **random Fourier
features** (Rahimi & Recht, 2007), giving the penalty and gradient in O(P * D) for D
features instead of O(P^2). Optimisation alternates a masked multiplicative update for V
(data term only) with a projected-gradient + backtracking step for U on the deterministic
(data + penalty) objective, warm-started from a plain Route 2 fit.

Penalising ``U`` (spatial maps) is the spatial-ICA analogue; ``target="V"`` penalises the
loadings instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .losses import EPS, get_loss
from .nmf_linear import LinearNMF, LinearNMFResult


# --------------------------------------------------------------------------- #
# Independence measures
# --------------------------------------------------------------------------- #

def _rbf_bandwidths(A: np.ndarray) -> np.ndarray:
    """Per-column RBF bandwidth (std-based; proportional to the median heuristic)."""
    s = A.std(axis=0)
    s[s < EPS] = 1.0
    return s


def _rff(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Random Fourier features for a standardised 1-D variable ``z`` (length P).

    Returns (P, D) features whose inner products approximate the unit-bandwidth RBF
    kernel ``exp(-(z-z')^2 / 2)`` in standardised space.
    """
    D = W.shape[0]
    return np.sqrt(2.0 / D) * np.cos(np.outer(z, W) + b)  # (P, D)


def _centered_features(A: np.ndarray, scales: np.ndarray, W, b, linear: bool):
    """Build centred feature maps for every column of ``A`` (P, K) -> list of (P, D)."""
    P, K = A.shape
    feats = []
    for k in range(K):
        z = A[:, k] / scales[k]
        if linear:
            phi = (z - z.mean())[:, None]            # linear kernel -> covariance only
        else:
            phi = _rff(z, W, b)
            phi = phi - phi.mean(axis=0, keepdims=True)
        feats.append(phi)
    return feats


def hsic_penalty_and_grad(
    A: np.ndarray, scales: np.ndarray, W, b, linear: bool, want_grad: bool = True
):
    """Sum of pairwise (unnormalised) HSIC over columns of ``A`` and its gradient.

    Uses the feature-covariance form ``HSIC(i,j) ≈ ||C_ij||_F^2`` with
    ``C_ij = (1/P) Φ̃_i^T Φ̃_j``. Returns ``(value, grad)`` where ``grad`` has the shape
    of ``A`` (zero if ``want_grad`` is False).
    """
    P, K = A.shape
    feats = _centered_features(A, scales, W, b, linear)
    value = 0.0
    G = [np.zeros((P, feats[0].shape[1])) for _ in range(K)] if want_grad else None
    for i in range(K):
        for j in range(i + 1, K):
            C = (feats[i].T @ feats[j]) / P                # (D, D)
            value += float((C * C).sum())
            if want_grad:
                # d||C||^2 / dΦ̃_i = (2/P) Φ̃_j C^T ;  symmetric term for j
                G[i] += (2.0 / P) * (feats[j] @ C.T)
                G[j] += (2.0 / P) * (feats[i] @ C)
    if not want_grad:
        return value, np.zeros_like(A)

    # chain through the feature map back to A
    grad = np.zeros_like(A)
    for k in range(K):
        Gk = G[k] - G[k].mean(axis=0, keepdims=True)       # transpose of centring
        z = A[:, k] / scales[k]
        if linear:
            grad[:, k] = Gk[:, 0] / scales[k]
        else:
            D = W.shape[0]
            # dΦ/da = -sqrt(2/D) sin(W z + b) * W / scale
            S = -np.sqrt(2.0 / D) * np.sin(np.outer(z, W) + b) * W   # (P, D)
            grad[:, k] = (Gk * S).sum(axis=1) / scales[k]
    return value, grad


def normalised_hsic_matrix(A: np.ndarray, n_sample: int = 1500, random_state=0):
    """Pairwise normalised HSIC (centred kernel alignment) in [0, 1] for columns of ``A``.

    An *independent* evaluation of MI-sense dependence (direct RBF kernels with the median
    heuristic), used to verify that the penalty achieved independence. 0 ≈ independent,
    1 ≈ fully dependent. Subsamples ``n_sample`` rows for tractability.
    """
    A = np.asarray(A, dtype=float)
    P, K = A.shape
    rng = np.random.default_rng(random_state)
    idx = rng.choice(P, size=min(n_sample, P), replace=False)
    As = A[idx]
    m = As.shape[0]
    H = np.eye(m) - np.ones((m, m)) / m

    def kernel(x):
        d2 = (x[:, None] - x[None, :]) ** 2
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        Kx = np.exp(-d2 / (med + EPS))
        return H @ Kx @ H

    HK = [kernel(As[:, k]) for k in range(K)]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            num = float((HK[i] * HK[j]).sum())
            den = np.sqrt(float((HK[i] * HK[i]).sum()) * float((HK[j] * HK[j]).sum()))
            M[i, j] = num / (den + EPS)
    return M


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

@dataclass
class IndependentNMFResult:
    U: np.ndarray
    V: np.ndarray
    loss: str
    lam: float
    target: str
    data_loss_curve: list[float] = field(default_factory=list)
    penalty_curve: list[float] = field(default_factory=list)
    n_iter: int = 0
    reconstruction_error: float = np.nan
    dependence_before: np.ndarray | None = None
    dependence_after: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def K(self) -> int:
        return self.U.shape[1]

    def reconstruct(self) -> np.ndarray:
        return self.U @ self.V

    def component(self, k: int):
        return self.U[:, k], self.V[k, :]


def _data_loss_grad_U(X, U, V, loss, W):
    """Data divergence value and its gradient w.r.t. U (Y = UV)."""
    Y = np.maximum(U @ V, EPS)
    if W is None:
        Wm = 1.0
    else:
        Wm = W
    if loss == "frobenius":
        dY = Wm * (Y - X)
    elif loss == "kl":
        dY = Wm * (1.0 - X / Y)
    else:  # is
        dY = Wm * ((Y - X) / (Y * Y))
    return (dY @ V.T)


class IndependentNMF:
    """IS/KL NMF with an ICA-style HSIC independence penalty (colorpath Route 2+).

    Parameters
    ----------
    n_components : K.
    loss         : data divergence, ``"is"`` (default), ``"kl"`` or ``"frobenius"``.
    lam          : independence penalty weight ``lambda`` (0 reduces to plain Route 2).
    target       : ``"U"`` (spatial maps, default) or ``"V"`` (loadings) — which factor is
                   driven toward independence.
    kernel       : ``"rbf"`` (true MI-sense independence, default) or ``"linear"``
                   (decorrelation / second-order only).
    n_features   : number of random Fourier features for the RBF kernel.
    max_iter     : outer iterations (each: one V update + ``inner_steps`` U steps).
    inner_steps  : projected-gradient steps on U per outer iteration.
    tol          : stop when the relative change in the total objective is below this.
    random_state : seed for warm start and the random features.
    warm_start_iter : iterations of plain Route 2 used to initialise (via ``LinearNMF``).

    Notes
    -----
    HSIC with the RBF kernel is zero iff the penalised factor's components are mutually
    independent (mutual information zero) — the Kernel-ICA dependence contrast — so larger
    ``lam`` trades a little reconstruction for ICA-style independence while preserving
    non-negativity and the multiplicative model.
    """

    def __init__(
        self,
        n_components: int,
        loss: str = "is",
        lam: float = 1.0,
        target: str = "U",
        kernel: str = "rbf",
        n_features: int = 128,
        max_iter: int = 150,
        inner_steps: int = 3,
        tol: float = 1e-6,
        random_state: int | None = None,
        warm_start_iter: int = 300,
    ):
        get_loss(loss)
        if target not in ("U", "V"):
            raise ValueError("target must be 'U' or 'V'")
        if kernel not in ("rbf", "linear"):
            raise ValueError("kernel must be 'rbf' or 'linear'")
        self.n_components = n_components
        self.loss = loss
        self.lam = lam
        self.target = target
        self.kernel = kernel
        self.n_features = n_features
        self.max_iter = max_iter
        self.inner_steps = inner_steps
        self.tol = tol
        self.random_state = random_state
        self.warm_start_iter = warm_start_iter

    def fit(self, X, mask=None, init=None) -> IndependentNMFResult:
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("IndependentNMF requires non-negative data")
        W = None if mask is None else np.asarray(mask, dtype=float)
        loss_fn = get_loss(self.loss)
        linear = self.kernel == "linear"
        rng = np.random.default_rng(self.random_state)

        # ---- warm start from plain Route 2 ----
        if init is not None:
            U, V = np.maximum(init[0].copy(), EPS), np.maximum(init[1].copy(), EPS)
        else:
            base = LinearNMF(
                self.n_components, loss=self.loss, max_iter=self.warm_start_iter,
                n_init=1, random_state=self.random_state,
            ).fit(X, mask=mask)
            U, V = base.U.copy(), base.V.copy()

        # random Fourier feature parameters (fixed for the fit)
        Wf = rng.standard_normal(self.n_features)
        bf = rng.uniform(0, 2 * np.pi, self.n_features)

        def penalised_target():
            return U if self.target == "U" else V.T  # penalise columns -> (samples, K)

        dep_before = normalised_hsic_matrix(penalised_target(), random_state=0)

        def total_obj():
            data = loss_fn(X, U @ V, W)
            A = penalised_target()
            scales = _rbf_bandwidths(A)
            pen, _ = hsic_penalty_and_grad(A, scales, Wf, bf, linear, want_grad=False)
            return data, pen, data + self.lam * pen

        data0, pen0, obj_prev = total_obj()
        data_curve = [data0]
        pen_curve = [pen0]
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # ---- V update: masked MU on the data term only ----
            V = self._mu_V(X, U, V, W)

            # ---- U update: projected gradient on data + lam * penalty ----
            if self.target == "U":
                U = self._prox_grad_U(X, U, V, W, loss_fn, Wf, bf, linear)
            else:
                # penalise V instead: gradient step on V's columns (= rows of V)
                V = self._prox_grad_V(X, U, V, W, loss_fn, Wf, bf, linear)

            data, pen, obj = total_obj()
            data_curve.append(data)
            pen_curve.append(pen)
            if obj_prev > 0 and abs(obj_prev - obj) / obj_prev < self.tol:
                obj_prev = obj
                break
            obj_prev = obj

        dep_after = normalised_hsic_matrix(penalised_target(), random_state=0)
        return IndependentNMFResult(
            U=U, V=V, loss=self.loss, lam=self.lam, target=self.target,
            data_loss_curve=data_curve, penalty_curve=pen_curve, n_iter=n_iter,
            reconstruction_error=data_curve[-1], dependence_before=dep_before,
            dependence_after=dep_after, mask=W,
        )

    # -- updates -----------------------------------------------------------
    def _mu_V(self, X, U, V, W):
        Y = np.maximum(U @ V, EPS)
        if self.loss == "frobenius":
            num = U.T @ (X if W is None else W * X)
            den = U.T @ (Y if W is None else W * Y)
        elif self.loss == "kl":
            num = U.T @ ((X / Y) if W is None else W * (X / Y))
            den = U.T @ (np.ones_like(X) if W is None else W)
        else:
            num = U.T @ ((X / Y**2) if W is None else W * (X / Y**2))
            den = U.T @ ((1.0 / Y) if W is None else W * (1.0 / Y))
        return np.maximum(V * (num / np.maximum(den, EPS)), EPS)

    def _effective_lam(self, gdata, gpen):
        """Dimensionless lambda: balance penalty force against data force.

        Raw HSIC is many orders of magnitude smaller than the divergence, so we scale the
        penalty gradient to the data gradient's norm. ``lam=1`` then means "push
        independence as hard as the data term"; ``lam=0`` recovers plain Route 2.
        """
        gd = np.linalg.norm(gdata)
        gp = np.linalg.norm(gpen)
        return self.lam * gd / (gp + EPS)

    def _prox_grad_U(self, X, U, V, W, loss_fn, Wf, bf, linear):
        scales = _rbf_bandwidths(U)
        _, gpen = hsic_penalty_and_grad(U, scales, Wf, bf, linear, want_grad=True)
        gdata = _data_loss_grad_U(X, U, V, self.loss, W)
        eff = self._effective_lam(gdata, gpen)
        grad = gdata + eff * gpen
        return self._armijo(
            U, grad, lambda Um: self._obj_U(X, Um, V, W, loss_fn, Wf, bf, linear, eff)
        )

    def _prox_grad_V(self, X, U, V, W, loss_fn, Wf, bf, linear):
        scales = _rbf_bandwidths(V.T)
        _, gpenT = hsic_penalty_and_grad(V.T, scales, Wf, bf, linear, want_grad=True)
        gpen = gpenT.T
        Y = np.maximum(U @ V, EPS)
        if self.loss == "frobenius":
            dY = (Y - X) if W is None else W * (Y - X)
        elif self.loss == "kl":
            dY = (1.0 - X / Y) if W is None else W * (1.0 - X / Y)
        else:
            dY = ((Y - X) / Y**2) if W is None else W * ((Y - X) / Y**2)
        gdata = U.T @ dY
        eff = self._effective_lam(gdata, gpen)
        grad = gdata + eff * gpen
        return self._armijo(
            V, grad, lambda Vm: self._obj_V(X, U, Vm, W, loss_fn, Wf, bf, linear, eff)
        )

    def _obj_U(self, X, U, V, W, loss_fn, Wf, bf, linear, eff):
        data = loss_fn(X, U @ V, W)
        scales = _rbf_bandwidths(U)
        pen, _ = hsic_penalty_and_grad(U, scales, Wf, bf, linear, want_grad=False)
        return data + eff * pen

    def _obj_V(self, X, U, V, W, loss_fn, Wf, bf, linear, eff):
        data = loss_fn(X, U @ V, W)
        scales = _rbf_bandwidths(V.T)
        pen, _ = hsic_penalty_and_grad(V.T, scales, Wf, bf, linear, want_grad=False)
        return data + eff * pen

    def _armijo(self, M, grad, obj, eta0=None, max_back=20):
        """Projected-gradient step with backtracking; projects onto M >= EPS."""
        f0 = obj(M)
        gnorm = np.linalg.norm(grad)
        if gnorm < EPS:
            return M
        eta = eta0 if eta0 is not None else 1.0 / gnorm
        for _ in range(max_back):
            Mn = np.maximum(M - eta * grad, EPS)
            if obj(Mn) <= f0:
                # one extra round of inner refinement at this step size
                return Mn
            eta *= 0.5
        return M


def fit_independent_nmf(X, n_components, **kwargs) -> IndependentNMFResult:
    """Functional shortcut for :class:`IndependentNMF`. ``mask``/``init`` are forwarded."""
    mask = kwargs.pop("mask", None)
    init = kwargs.pop("init", None)
    return IndependentNMF(n_components, **kwargs).fit(X, mask=mask, init=init)
