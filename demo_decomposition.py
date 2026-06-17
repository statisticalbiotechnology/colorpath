"""
demo_decomposition.py — end-to-end colorpath pipeline on synthetic IMS data.

Generates a small synthetic imaging-mass-spectrometry dataset with a known
ground-truth pathway structure (two spatial activity regions, each multiplicatively
coupling a set of metabolites), corrupts it with multiplicative noise and detector
saturation, then runs the full pipeline:

    diagnostics  -> noise-model selection (variance vs mean)
    saturation   -> mask construction
    Route 2      -> masked IS-NMF in linear space
    illustration -> pathway activity graph + image per component
    Route 1      -> asinh equal-loading NMF, warm-started from Route 2 (comparison)

Run:
    python demo_decomposition.py
"""

from __future__ import annotations

import numpy as np

from colorpath.decomposition import (
    LinearNMF,
    LogLevelNMF,
    build_saturation_mask,
    variance_vs_mean,
)
from colorpath.illustration import illustrate_component


def make_synthetic_ims(seed: int = 0):
    """Two-pathway synthetic IMS cube with multiplicative noise + one saturating ion."""
    rng = np.random.default_rng(seed)
    H, W = 24, 24
    P = H * W
    yy, xx = np.mgrid[0:H, 0:W]

    # Two spatial activity regions (Gaussian blobs) -> ground-truth U.
    blob1 = np.exp(-(((xx - 7) ** 2 + (yy - 7) ** 2) / 18.0))
    blob2 = np.exp(-(((xx - 16) ** 2 + (yy - 17) ** 2) / 22.0))
    U_true = np.column_stack([blob1.ravel(), blob2.ravel()]) + 0.02

    # Metabolites: first 6 belong to pathway 1, next 6 to pathway 2, last 3 shared/noise.
    metabolites = [f"m{i}" for i in range(15)]
    V_true = np.zeros((2, len(metabolites)))
    V_true[0, 0:6] = rng.uniform(0.6, 1.0, 6)
    V_true[1, 6:12] = rng.uniform(0.6, 1.0, 6)
    V_true[:, 12:] = rng.uniform(0.0, 0.15, (2, 3))  # weak background
    V_true *= 50.0

    clean = U_true @ V_true
    # Multiplicative (Gamma-like) noise -> variance ∝ mean^2 -> favours IS.
    noisy = clean * rng.gamma(shape=20.0, scale=1 / 20.0, size=clean.shape)

    # Detector saturation on the most abundant ion (m0): hard clip with pile-up.
    ceiling = np.quantile(noisy[:, 0], 0.85)
    noisy[:, 0] = np.minimum(noisy[:, 0], ceiling)

    # A simple "pathway" edge set for illustration (chain within each pathway).
    edges = [(f"m{i}", f"m{i+1}") for i in range(5)]          # pathway 1 chain
    edges += [(f"m{i}", f"m{i+1}") for i in range(6, 11)]      # pathway 2 chain
    return noisy, metabolites, edges, (H, W), U_true, V_true


def main():
    X, metabolites, edges, image_shape, U_true, V_true = make_synthetic_ims()

    # 1. Diagnostics: which noise model?
    vm = variance_vs_mean(X)
    print(f"[diagnostics] var~mean^{vm.slope:.2f}  -> recommended loss: {vm.recommended_loss!r}")

    # 2. Saturation mask (auto-detect the clipped ion).
    mask, report = build_saturation_mask(X, auto_detect=True)
    print(f"[saturation] flagged ions: {np.where(report.flagged)[0].tolist()}  "
          f"masked entries: {report.n_masked}")

    # 3. Route 2 — masked IS-NMF in linear space.
    route2 = LinearNMF(
        n_components=2, loss=vm.recommended_loss, n_init=4,
        max_iter=400, random_state=0,
    ).fit(X, mask=mask)
    print(f"[route2] loss={route2.loss} final_div={route2.reconstruction_error:.3f} "
          f"iters={route2.n_iter} converged={route2.converged}")

    # 4. Illustrate each component (pathway activity graph + image).
    for k in range(route2.K):
        paths = illustrate_component(
            route2, component=k,
            metabolite_names=metabolites,
            pathway_edges=edges,
            image_shape=image_shape,
            graph_output=f"route2_component_{k}_graph.svg",
            image_output=f"route2_component_{k}_image.svg",
            graph_colormap="viridis",
        )
        print(f"[illustration] component {k}: {paths}")

    # 5. Route 1 — asinh equal-loading NMF, warm-started from Route 2 for comparison.
    route1 = LogLevelNMF(
        n_components=2, c=np.median(X[X > 0]), l1=0.05, equalise=0.4, random_state=0,
    ).fit(X, init=(route2.U, route2.V))
    print(f"[route1] final_loss={route1.loss_curve[-1]:.3f} iters={route1.n_iter}")
    print(f"[route1] membership (binary):\n{route1.membership()}")


if __name__ == "__main__":
    main()
