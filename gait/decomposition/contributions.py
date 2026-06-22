"""
contributions.py — per-metabolite variation-explained scaling of component loadings.

A component's spectral loading ``V[k, :]`` is on linear-abundance units, so when it is used
directly to colour a pathway activity graph the few high-concentration metabolites saturate
the colour scale and everything else looks uniformly dark (a near-binary appearance). For
illustration it is usually more informative to ask, *per metabolite*, **how much of that
metabolite's variation this component explains** — a quantity that is normalised away from
absolute concentration.

For metabolite ``m`` the rank-1 contribution of component ``k`` is the outer-product term
``U[:, k] * V[k, m]`` over pixels; its variation (variance, or energy) is
``V[k, m]^2 * Var_p(U[:, k])``. Normalising per metabolite gives

    F[k, m] = V[k, m]^2 * s_k  /  sum_k' ( V[k', m]^2 * s_k' ) ,   s_k = Var_p(U[:, k])

which lies in ``[0, 1]`` and, with ``normalize="sum"``, sums to 1 over components for each
metabolite. The product ``V[k, m] * sqrt(s_k)`` is invariant to NMF's scale ambiguity
(``U[:, k] -> cU[:, k]``, ``V[k, :] -> V[k, :]/c``), so ``F`` is well defined regardless of
how the factors are normalised.

Because NMF components are not in general orthogonal, the per-component variances do not
exactly partition the metabolite's total variance (cross terms are dropped); ``F`` is
therefore a normalised *share* among components, exact in the independent-component limit
(see :class:`~gait.decomposition.nmf_independent.IndependentNMF`).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def variation_explained(
    U: np.ndarray,
    V: np.ndarray,
    mode: str = "variance",
    normalize: str = "sum",
) -> np.ndarray:
    """Fraction of each metabolite's variation explained by each component.

    Parameters
    ----------
    U         : (P, K) spatial scores.
    V         : (K, M) spectral loadings.
    mode      : ``"variance"`` weights each component by ``Var_p(U[:, k])`` (spatial
                heterogeneity; the default, matching "variation"); ``"energy"`` weights by
                ``sum_p U[:, k]^2`` (includes the mean level).
    normalize : ``"sum"`` divides each metabolite's column by the total over components
                (columns sum to 1 — "fraction explained"); ``"max"`` divides by the
                largest component (the dominant component reads 1, others relative).

    Returns
    -------
    F : (K, M) array in ``[0, 1]``. ``F[k, m]`` is the share of metabolite ``m``'s
        variation attributable to component ``k``.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if mode == "variance":
        s = U.var(axis=0)              # (K,)
    elif mode == "energy":
        s = (U ** 2).sum(axis=0)       # (K,)
    else:
        raise ValueError("mode must be 'variance' or 'energy'")

    contrib = (V ** 2) * s[:, None]    # (K, M)
    if normalize == "sum":
        denom = contrib.sum(axis=0, keepdims=True)
    elif normalize == "max":
        denom = contrib.max(axis=0, keepdims=True)
    else:
        raise ValueError("normalize must be 'sum' or 'max'")
    return contrib / np.maximum(denom, EPS)


def loading_share(
    U: np.ndarray,
    V: np.ndarray,
    normalize: str = "sum",
) -> np.ndarray:
    """Per-metabolite share of (scale-invariant) loading captured by each component.

    A gentler sibling of :func:`variation_explained`: it is *linear* in the loading
    rather than variance-weighted. For metabolite ``m`` the component weight is
    ``L[k, m] = |V[k, m]| * sd_k`` with ``sd_k = std_p(U[:, k])``, normalised per
    metabolite. As with :func:`variation_explained` the product ``V[k, m] * sd_k`` is
    invariant to NMF's per-component scale ambiguity (``U[:, k] -> cU[:, k]``,
    ``V[k, :] -> V[k, :]/c``), so ``L`` is well defined.

    Because it does **not** square the loading, it does not over-concentrate on the few
    high-variance, high-abundance metabolites the way :func:`variation_explained` can: a
    low-abundance pathway member (e.g. a transcription factor sitting beside hugely
    abundant structural genes) keeps a visible share instead of collapsing to ~0. This
    makes membership graphs far more readable when one component is dominated by a handful
    of very abundant features — use ``graph_value="loading_share"`` in
    :func:`gait.illustration.illustrate_component`.

    Parameters
    ----------
    U         : (P, K) spatial scores.
    V         : (K, M) spectral loadings.
    normalize : ``"sum"`` divides each metabolite's column by the total over components
                (columns sum to 1); ``"max"`` divides by the largest component.

    Returns
    -------
    L : (K, M) array in ``[0, 1]``. ``L[k, m]`` is the share of metabolite ``m``'s
        loading attributable to component ``k``.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    s = U.std(axis=0)                      # (K,) scale-invariance weight
    contrib = np.abs(V) * s[:, None]       # (K, M), linear in V
    if normalize == "sum":
        denom = contrib.sum(axis=0, keepdims=True)
    elif normalize == "max":
        denom = contrib.max(axis=0, keepdims=True)
    else:
        raise ValueError("normalize must be 'sum' or 'max'")
    return contrib / np.maximum(denom, EPS)


def spatial_variation_explained(
    U: np.ndarray,
    V: np.ndarray,
    mode: str = "variance",
    normalize: str = "sum",
) -> np.ndarray:
    """Per-pixel fraction of variation explained by each component.

    The spatial analogue of :func:`variation_explained`, obtained by swapping the roles of
    pixels and metabolites: for pixel ``p``, the across-metabolite variation of component
    ``k``'s rank-1 contribution is ``U[p, k]^2 * Var_m(V[k, :])``, normalised per pixel,

        G[p, k] = U[p, k]^2 * w_k  /  sum_k' ( U[p, k']^2 * w_k' ),   w_k = Var_m(V[k, :]).

    Returns
    -------
    G : (P, K) array in ``[0, 1]``; with ``normalize="sum"`` each pixel's row sums to 1.
        ``G[:, k]`` is the per-pixel share map for component ``k`` and can be rendered with
        :func:`gait.illustration.render_pathway_activity_image` directly.
    """
    return variation_explained(
        np.asarray(V, dtype=float).T, np.asarray(U, dtype=float).T,
        mode=mode, normalize=normalize,
    ).T
