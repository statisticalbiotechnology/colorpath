"""Tests for the colorpath decomposition engine and illustration bridge."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorpath.decomposition import (
    IndependentNMF,
    LinearNMF,
    LogLevelNMF,
    asinh_transform,
    build_saturation_mask,
    component_recovery,
    fit_linear_nmf,
    frobenius_loss,
    inverse_asinh,
    is_loss,
    kl_loss,
    normalised_hsic_matrix,
    select_k,
    variance_vs_mean,
    variation_explained,
    spatial_variation_explained,
)
from colorpath.decomposition.saturation import detect_saturation_ceiling


# ----------------------------- fixtures -----------------------------

def low_rank_data(P=60, M=20, K=3, seed=0, noise="mult"):
    rng = np.random.default_rng(seed)
    U = rng.random((P, K))
    V = rng.random((K, M))
    X = U @ V + 1e-3
    if noise == "mult":
        X = X * rng.gamma(50.0, 1 / 50.0, X.shape)
    elif noise == "pois":
        X = rng.poisson(X * 100) / 100.0 + 1e-3
    return X, U, V


# ----------------------------- losses -----------------------------

def test_is_scale_invariance():
    rng = np.random.default_rng(1)
    X = rng.random((5, 4)) + 0.1
    Y = rng.random((5, 4)) + 0.1
    assert is_loss(X, Y) == pytest.approx(is_loss(10 * X, 10 * Y), rel=1e-6)


def test_losses_zero_at_perfect_fit():
    rng = np.random.default_rng(2)
    Y = rng.random((4, 4)) + 0.5
    assert frobenius_loss(Y, Y) == pytest.approx(0.0, abs=1e-8)
    assert kl_loss(Y, Y) == pytest.approx(0.0, abs=1e-6)
    assert is_loss(Y, Y) == pytest.approx(0.0, abs=1e-6)


def test_masked_loss_ignores_masked_entries():
    rng = np.random.default_rng(3)
    X = rng.random((4, 4)) + 0.5
    Y = X.copy()
    Y[0, 0] = 99.0  # large error
    W = np.ones_like(X)
    W[0, 0] = 0.0
    assert frobenius_loss(X, Y, W) == pytest.approx(0.0, abs=1e-8)
    assert frobenius_loss(X, Y) > 1.0


# ----------------------------- Route 2 -----------------------------

@pytest.mark.parametrize("loss", ["frobenius", "kl", "is"])
def test_linear_nmf_monotone_decrease(loss):
    X, _, _ = low_rank_data(noise="mult")
    res = LinearNMF(3, loss=loss, max_iter=200, random_state=0).fit(X)
    curve = np.array(res.loss_curve)
    # MU is monotonically non-increasing (allow tiny numerical wiggle).
    assert np.all(np.diff(curve) <= 1e-6 * (np.abs(curve[:-1]) + 1e-9))


@pytest.mark.parametrize("loss", ["frobenius", "kl", "is"])
def test_linear_nmf_recovers_low_rank(loss):
    X, _, _ = low_rank_data(K=3, noise="mult")
    res = LinearNMF(3, loss=loss, n_init=3, max_iter=400, random_state=0).fit(X)
    rel = np.linalg.norm(res.reconstruct() - X) / np.linalg.norm(X)
    assert rel < 0.2
    assert np.all(res.U >= 0) and np.all(res.V >= 0)


def test_linear_nmf_rejects_negative():
    X = np.array([[1.0, -1.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        LinearNMF(1).fit(X)


def test_warm_start_init_used():
    X, _, _ = low_rank_data()
    base = LinearNMF(3, loss="is", max_iter=50, random_state=0).fit(X)
    cont = LinearNMF(3, loss="is", max_iter=50, random_state=0).fit(
        X, init=(base.U, base.V)
    )
    # Continuing from a fit should not be worse than starting it fresh.
    assert cont.reconstruction_error <= base.reconstruction_error + 1e-6


def test_mask_reduces_saturating_ion_off_component_loading():
    # Build data where ion 0 saturates inside component 0's region.
    X, U, V = low_rank_data(P=80, M=12, K=2, noise="mult", seed=5)
    # Make ion 0 load only on component 0, strongly.
    V[:, 0] = [5.0, 0.0]
    X = U @ V + 1e-3
    ceiling = np.quantile(X[:, 0], 0.8)
    Xs = X.copy()
    Xs[:, 0] = np.minimum(Xs[:, 0], ceiling)

    base = LinearNMF(2, loss="is", n_init=3, random_state=0).fit(Xs)
    mask, _ = build_saturation_mask(Xs, saturation_quantile=0.8)
    masked = LinearNMF(2, loss="is", n_init=3, random_state=0).fit(Xs, mask=mask)

    rec = component_recovery(base.V, masked.V, ion_index=0,
                             primary_component=int(np.argmax(base.V[:, 0])))
    # Masking should not increase the off-component spread of the saturating ion.
    assert rec["masked_off_component_fraction"] <= rec["baseline_off_component_fraction"] + 0.05


# --------------------------- saturation ---------------------------

def test_detect_hard_clip_ceiling():
    rng = np.random.default_rng(7)
    x = rng.exponential(1.0, 2000)
    x = np.minimum(x, 2.0)  # hard clip -> pile-up at 2.0
    ceil, frac = detect_saturation_ceiling(x)
    assert np.isfinite(ceil)
    assert frac > 0.05


def test_no_false_clip_on_smooth_data():
    rng = np.random.default_rng(8)
    x = rng.normal(5.0, 1.0, 2000)
    ceil, _ = detect_saturation_ceiling(x)
    assert not np.isfinite(ceil)


def test_build_mask_unmasked_by_default():
    X, _, _ = low_rank_data()
    W, report = build_saturation_mask(X, warn=False)
    assert np.all(W == 1.0)
    assert report.n_masked == 0


def test_build_mask_with_quantile():
    X, _, _ = low_rank_data()
    W, report = build_saturation_mask(X, saturation_quantile=0.9)
    assert report.n_masked > 0
    assert set(np.unique(W)).issubset({0.0, 1.0})


# --------------------------- diagnostics ---------------------------

def test_variance_vs_mean_picks_is_for_multiplicative():
    rng = np.random.default_rng(9)
    means = rng.uniform(1, 100, 40)
    # variance proportional to mean^2 -> IS
    X = np.vstack([
        m * rng.gamma(25.0, 1 / 25.0, 200) for m in means
    ]).T
    res = variance_vs_mean(X)
    assert res.recommended_loss == "is"
    assert res.slope > 1.5


def test_variance_vs_mean_picks_kl_for_poisson():
    rng = np.random.default_rng(10)
    means = rng.uniform(5, 200, 40)
    X = np.vstack([rng.poisson(m, 300) for m in means]).T.astype(float)
    res = variance_vs_mean(X)
    assert res.recommended_loss == "kl"


# --------------------------- selection ---------------------------

def test_select_k_returns_results_per_k():
    X, _, _ = low_rank_data(K=3)
    sel = select_k(X, range(1, 6), loss="frobenius", max_iter=100, random_state=0)
    assert set(sel.results) == set(range(1, 6))
    assert sel.best_k in sel.ks
    # error should decrease as K grows
    assert sel.errors[0] >= sel.errors[-1]


# ----------------------------- Route 1 -----------------------------

def test_asinh_roundtrip():
    x = np.array([0.0, 1.0, 10.0, 100.0])
    assert np.allclose(inverse_asinh(asinh_transform(x, 2.0), 2.0), x)


def test_loglevel_nmf_runs_and_equalises():
    X, _, _ = low_rank_data(K=2, noise="mult")
    res = LogLevelNMF(2, l1=0.05, equalise=0.5, max_iter=300, random_state=0).fit(X)
    assert res.g.shape[0] == X.shape[0]
    assert res.p.shape[1] == X.shape[1]
    mem = res.membership()
    assert mem.shape == res.p.shape
    assert mem.sum() > 0  # some metabolites assigned


def test_loglevel_warm_start_from_route2():
    X, _, _ = low_rank_data(K=2)
    r2 = LinearNMF(2, loss="is", random_state=0).fit(X)
    r1 = LogLevelNMF(2, random_state=0).fit(X, init=(r2.U, r2.V))
    assert r1.loss_curve[-1] <= r1.loss_curve[0] + 1e-6


# ------------------------- independence (ICA-style) -------------------------

def dependent_sources(P=800, M=14, seed=1, noise="mult"):
    """Two strongly (nonlinearly) dependent non-negative spatial sources."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 3, P)
    s1 = t + 0.2 * rng.gamma(2, 1, P)
    s2 = (t - 1.5) ** 2 + 0.2 * rng.gamma(2, 1, P)  # dependent on t, ~uncorrelated
    S = np.column_stack([s1, s2])
    S = S - S.min(0) + 0.05
    V = rng.random((2, M)) + 0.1
    X = S @ V
    if noise == "mult":
        X = X * rng.gamma(40, 1 / 40, X.shape)
    return X


def _offdiag_mean(M):
    return float(M[~np.eye(M.shape[0], dtype=bool)].mean())


def test_normalised_hsic_independent_vs_dependent():
    rng = np.random.default_rng(0)
    a = rng.gamma(2, 1, 1500)
    indep = np.column_stack([a, rng.gamma(2, 1, 1500)])
    dep = np.column_stack([a, a**2 + 0.01 * rng.gamma(2, 1, 1500)])
    assert _offdiag_mean(normalised_hsic_matrix(dep, random_state=0)) > \
           _offdiag_mean(normalised_hsic_matrix(indep, random_state=0))


def test_independent_nmf_lam0_matches_route2_quality():
    X = dependent_sources()
    r = IndependentNMF(2, loss="kl", lam=0.0, max_iter=80, random_state=0).fit(X)
    assert np.all(r.U >= 0) and np.all(r.V >= 0)
    rel = np.linalg.norm(r.reconstruct() - X) / np.linalg.norm(X)
    assert rel < 0.3


def test_independent_nmf_reduces_dependence():
    X = dependent_sources()
    base = IndependentNMF(2, loss="kl", lam=0.0, max_iter=120, inner_steps=2,
                          random_state=0).fit(X)
    indep = IndependentNMF(2, loss="kl", lam=5.0, max_iter=120, inner_steps=2,
                           random_state=0).fit(X)
    dep_base = _offdiag_mean(base.dependence_after)
    dep_indep = _offdiag_mean(indep.dependence_after)
    # The penalty should measurably lower MI-sense dependence.
    assert dep_indep < dep_base - 0.02
    # ...without destroying the reconstruction.
    rel = np.linalg.norm(indep.reconstruct() - X) / np.linalg.norm(X)
    assert rel < 0.4


def test_variation_explained_sums_to_one_per_metabolite():
    rng = np.random.default_rng(0)
    U = rng.random((100, 3))
    V = rng.random((3, 8))
    F = variation_explained(U, V, normalize="sum")
    assert F.shape == (3, 8)
    assert np.all((F >= 0) & (F <= 1))
    assert np.allclose(F.sum(axis=0), 1.0)


def test_variation_explained_scale_invariant():
    rng = np.random.default_rng(1)
    U = rng.random((80, 3)) + 0.1
    V = rng.random((3, 6)) + 0.1
    F = variation_explained(U, V)
    # NMF scale ambiguity: U[:,k]*c, V[k,:]/c must leave the fractions unchanged.
    c = np.array([5.0, 0.2, 3.0])
    F2 = variation_explained(U * c, V / c[:, None])
    assert np.allclose(F, F2)


def test_variation_explained_removes_concentration_imbalance():
    # One metabolite is 1000x more abundant but is "owned" by component 1, not 0.
    U = np.abs(np.random.default_rng(2).standard_normal((200, 2))) + 0.05
    V = np.array([[1.0, 0.0], [0.0, 1000.0]])  # m0 -> comp0, m1 -> comp1 (high conc.)
    F = variation_explained(U, V)
    # Despite m1's huge loading, component 0 explains ~none of its variation.
    assert F[0, 1] < 0.05 and F[1, 1] > 0.95
    assert F[0, 0] > 0.95


def test_variation_explained_max_normalize():
    rng = np.random.default_rng(3)
    F = variation_explained(rng.random((50, 3)), rng.random((3, 5)), normalize="max")
    assert np.allclose(F.max(axis=0), 1.0)


def test_spatial_variation_explained_sums_to_one_per_pixel():
    rng = np.random.default_rng(4)
    U = rng.random((120, 3))
    V = rng.random((3, 9))
    G = spatial_variation_explained(U, V)
    assert G.shape == (120, 3)               # (P, K)
    assert np.all((G >= 0) & (G <= 1))
    assert np.allclose(G.sum(axis=1), 1.0)   # each pixel's shares sum to 1


def test_spatial_variation_explained_is_transpose_dual():
    # Per-pixel share equals the metabolite-share of the transposed factorisation.
    rng = np.random.default_rng(5)
    U = rng.random((40, 3)) + 0.1
    V = rng.random((3, 7)) + 0.1
    G = spatial_variation_explained(U, V)
    assert np.allclose(G, variation_explained(V.T, U.T).T)


def test_independent_nmf_linear_kernel_runs():
    X = dependent_sources()
    r = IndependentNMF(2, loss="frobenius", lam=2.0, kernel="linear",
                       max_iter=60, random_state=0).fit(X)
    assert r.dependence_after.shape == (2, 2)
