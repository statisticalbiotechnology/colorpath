"""
benchmark.py — formalising the log-vs-linear comparison.

GAIT's claim is that staying in *linear* space (so a rank-1 component is a genuine
multiplicative outer product, R2) and putting the multiplicative-error model in the loss
(IS/KL, R1) recovers a pathway's structure more faithfully than the conventional
``log-transform then PCA/NMF`` recipe, which converts the multiplicative coupling into an
additive-shared-log structure (a different object).

This module makes that testable. :func:`synthetic_coupling_dataset` plants ``K``
region-structured components, each multiplicatively coupling a group of features whose
loadings span orders of magnitude (so the log's dynamic-range compression bites), and
corrupts them with multiplicative (Gamma) or Poisson noise. :func:`run_synthetic_benchmark`
fits four methods and scores how well each recovers the planted structure:

    GAIT (KL)             LinearNMF(loss="kl")   on linear X      -- R1+R2
    GAIT (IS)             LinearNMF(loss="is")   on linear X      -- R1+R2
    linear NMF (Frob)     LinearNMF(loss="frob") on linear X      -- R2 only (wrong error)
    log1p + NMF           LinearNMF(loss="frob") on log1p(X)      -- neither
    log1p + PCA           top-K PCA              on log1p(X)      -- the conventional baseline

Three scores per method (all in [0, 1], higher is better):
    U-recovery   mean |corr| of each recovered spatial map to its matched true component
    V-recovery   mean |corr| of each recovered loading to its matched true loading
                 (this is the R2 / coupling-fidelity score the log is expected to fail)
    region-acc   accuracy of the per-location dominant-component map vs the planted regions

Only numpy/scipy and the GAIT engine are used (no scikit-learn), so the benchmark has no
extra dependencies. :func:`region_mutual_information` applies the same idea to *real* data
with region labels (e.g. the mouse-brain striatum annotation).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .decomposition import LinearNMF

EPS = 1e-9


def synthetic_coupling_dataset(
    side: int = 24,
    n_components: int = 3,
    per_group: int = 6,
    noise: str = "mult",
    shape: float = 10.0,
    seed: int = 0,
):
    """Plant ``K`` region-structured, multiplicatively-coupled components on a grid.

    Each component is a Gaussian spatial blob (a "region") times a group of ``per_group``
    features whose loadings span three orders of magnitude (``logspace(0, 3)``), so a log
    transform's compression of the dynamic range materially distorts the loadings. Returns
    ``(X, U_true, V_true)`` with ``X = (U_true V_true) * noise``.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    P = side * side
    K = n_components
    # K blob centres spread around the grid.
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    cx = side / 2 + 0.3 * side * np.cos(angles)
    cy = side / 2 + 0.3 * side * np.sin(angles)
    sigma = side / 6.0
    U = np.stack(
        [np.exp(-(((xx - cx[k]) ** 2 + (yy - cy[k]) ** 2) / (2 * sigma ** 2))).ravel()
         for k in range(K)], axis=1,
    ) + 0.02
    M = K * per_group
    V = np.full((K, M), 0.02)
    for k in range(K):
        lo = k * per_group
        V[k, lo:lo + per_group] = np.logspace(0, 3, per_group) * rng.uniform(0.5, 1.5, per_group)
    clean = U @ V + EPS
    if noise == "mult":
        X = clean * rng.gamma(shape, 1.0 / shape, clean.shape)   # variance ∝ mean^2
    elif noise == "pois":
        X = rng.poisson(clean).astype(float)                     # variance ∝ mean
    else:
        raise ValueError("noise must be 'mult' or 'pois'")
    return X, U, V


def _topk_pca(Y: np.ndarray, K: int):
    """Top-K PCA of ``Y`` (no sklearn): returns (scores P×K, loadings K×M)."""
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U_, S_, Vt_ = np.linalg.svd(Yc, full_matrices=False)
    return U_[:, :K] * S_[:K], Vt_[:K]


def _abs_corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|Pearson correlation| between every column of A and every column of B."""
    def z(M):
        M = M - M.mean(0, keepdims=True)
        n = np.linalg.norm(M, axis=0, keepdims=True)
        return M / np.maximum(n, EPS)
    return np.abs(z(A).T @ z(B))


def _match(corr: np.ndarray):
    """Optimal one-to-one assignment maximising |corr| (rows -> columns)."""
    r, c = linear_sum_assignment(-corr)
    return r, c


def _fit_methods(X: np.ndarray, K: int, n_init: int, max_iter: int, seed: int):
    """Return {method: (U_rec P×K, V_rec K×M)} for the five methods."""
    Xlog = np.log1p(X)
    out = {}
    for name, loss, data in [
        ("GAIT (KL)", "kl", X),
        ("GAIT (IS)", "is", X),
        ("linear NMF (Frob)", "frobenius", X),
        ("log1p + NMF", "frobenius", Xlog),
    ]:
        r = LinearNMF(K, loss=loss, n_init=n_init, max_iter=max_iter,
                      random_state=seed).fit(data)
        out[name] = (r.U, r.V)
    out["log1p + PCA"] = _topk_pca(Xlog, K)
    return out


def run_synthetic_benchmark(
    n_components: int = 3,
    noise: str = "mult",
    n_init: int = 4,
    max_iter: int = 600,
    seed: int = 0,
    **data_kwargs,
):
    """Fit all methods on a planted dataset and score recovery. Returns a results dict
    ``{method: {"U_recovery", "V_recovery", "region_acc"}}``."""
    X, U_true, V_true = synthetic_coupling_dataset(
        n_components=n_components, noise=noise, seed=seed, **data_kwargs)
    region_true = U_true.argmax(axis=1)
    results: dict[str, dict[str, float]] = {}
    for name, (U_rec, V_rec) in _fit_methods(X, n_components, n_init, max_iter, seed).items():
        cU = _abs_corr_matrix(U_rec, U_true)
        rows, cols = _match(cU)
        u_score = float(cU[rows, cols].mean())
        cV = _abs_corr_matrix(V_rec.T, V_true.T)
        v_score = float(cV[rows, cols].mean())
        # remap recovered dominant-component labels to true via the same matching
        remap = {int(r): int(c) for r, c in zip(rows, cols)}
        region_rec = np.array([remap[j] for j in U_rec.argmax(axis=1)])
        region_acc = float((region_rec == region_true).mean())
        results[name] = {"U_recovery": u_score, "V_recovery": v_score,
                         "region_acc": region_acc}
    return results


def region_mutual_information(labels: np.ndarray, regions: np.ndarray) -> float:
    """Mutual information (bits) between a per-location component label and a region label.

    The real-data analogue of ``region_acc``: how much the dominant-component partition
    (e.g. :func:`gait.spatial.dominant_component`) tells you about the annotated regions.
    Higher means the pathway components align better with anatomy.
    """
    labels = np.asarray(labels); regions = np.asarray(regions)
    a = {v: i for i, v in enumerate(np.unique(labels))}
    b = {v: i for i, v in enumerate(np.unique(regions))}
    C = np.zeros((len(a), len(b)))
    for l, r in zip(labels, regions):
        C[a[l], b[r]] += 1
    P = C / C.sum()
    pa = P.sum(1, keepdims=True); pb = P.sum(0, keepdims=True)
    nz = P > 0
    return float(np.sum(P[nz] * np.log2(P[nz] / (pa @ pb)[nz])))


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Render a results dict as a fixed-width table."""
    cols = ["U_recovery", "V_recovery", "region_acc"]
    lines = [f"{'method':<20}" + "".join(f"{c:>13}" for c in cols)]
    for name, sc in results.items():
        lines.append(f"{name:<20}" + "".join(f"{sc[c]:>13.3f}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    for nz in ("mult", "pois"):
        print(f"\n=== planted multiplicative coupling, noise={nz!r} ===")
        print(format_table(run_synthetic_benchmark(noise=nz)))
