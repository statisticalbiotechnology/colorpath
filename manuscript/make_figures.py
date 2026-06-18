"""
make_figures.py — regenerate the manuscript figures from a MALDI-MSI parquet.

Reproduces the `01_pd_51` results shown in `main.tex`: a KL-NMF decomposition (K=5) and,
for the first three components, three views — the raw activity score U[:,k], the per-pixel
spatial fraction of variation explained G[:,k], and the pathway graph coloured by the
per-metabolite fraction of variation explained F[k,:] over the catecholamine/serotonin
network.

Usage:
    python manuscript/make_figures.py /path/to/01_pd_51_raw_by_metabolite_5ppm.parquet

The parquet must contain columns `x`, `y` (pixel coordinates) and one column per
metabolite. Figures are written to `manuscript/figures/`.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorpath.decomposition import (
    LinearNMF,
    spatial_variation_explained,
    variation_explained,
)
from colorpath.illustration import (
    CATECHOLAMINE_SEROTONIN_EDGES,
    CATECHOLAMINE_SEROTONIN_POSITIONS,
    draw_pathway,
    render_pathway_activity_image,
)

N_SHOW = 3      # number of leading components to illustrate
K = 5           # total components fitted


def main(parquet_path: str):
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    mets = [c for c in df.columns if c not in ("sample_name", "x", "y")]
    X = df[mets].values.astype(float)
    Wg, Hg = int(df.x.max()), int(df.y.max())
    pix = (df.y.values - 1) * Wg + (df.x.values - 1)

    res = LinearNMF(K, loss="kl", n_init=6, max_iter=800, tol=1e-7,
                    random_state=0).fit(X)
    rel = np.linalg.norm(res.reconstruct() - X) / np.linalg.norm(X)
    print(f"KL-NMF K={K}: relative reconstruction error = {rel:.3f}")

    # Fraction-of-variation-explained quantities (scale-invariant, so computed from the
    # raw fit). F is per-metabolite (graph); G is per-pixel (spatial map).
    F = variation_explained(res.U, res.V)            # (K, M) in [0, 1]
    G = spatial_variation_explained(res.U, res.V)    # (P, K) in [0, 1]

    # Display-normalise the raw activity maps (unit max) for the top row.
    U, V = res.U.copy(), res.V.copy()
    for k in range(K):
        s = U[:, k].max()
        if s > 0:
            U[:, k] /= s
            V[k] *= s

    # --- One combined figure, three rows per component: (1) raw activity score, (2) the
    #     per-pixel spatial fraction of variation explained, (3) the pathway graph coloured
    #     by the per-metabolite fraction of variation explained. Rows 1 and 2/3 mean
    #     different things and are deliberately both shown (see the manuscript). ---
    fig, axes = plt.subplots(
        3, N_SHOW, figsize=(6.0 * N_SHOW, 14),
        gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.95], hspace=0.06, wspace=0.05),
    )
    rowlab = [
        "activity score  $U_{:,k}$\n(max-normalised)",
        "spatial fraction\nexplained  $G_{:,k}$",
        "graph: fraction\nexplained  $F_{k,:}$",
    ]
    for k in range(N_SHOW):
        top = [mets[i] for i in np.argsort(V[k])[::-1][:3]]
        # row 0: raw activity score (where the component is active)
        render_pathway_activity_image(
            scores=U[:, k], image_shape=(Hg, Wg), pixel_index=pix,
            colormap="magma", ax=axes[0][k], colorbar=True, title_fontsize=11,
            colorbar_label="activity (norm.)", title=f"Component {k}\n" + ", ".join(top),
        )
        # row 1: per-pixel fraction of variation explained (where it dominates)
        render_pathway_activity_image(
            scores=G[:, k], image_shape=(Hg, Wg), pixel_index=pix,
            colormap="magma", ax=axes[1][k], colorbar=True,
            colorbar_label="fraction explained", title="",
        )
        # row 2: pathway graph coloured by per-metabolite fraction of variation explained
        abundance = {m: float(F[k, j]) for j, m in enumerate(mets)}
        draw_pathway(
            pathway=CATECHOLAMINE_SEROTONIN_EDGES, abundance=abundance,
            positions=CATECHOLAMINE_SEROTONIN_POSITIONS, colormap="magma",
            node_size=300, font_size=5.5, ax=axes[2][k], colorbar=False,
            title="", label_halo=True, vmin=0.0, vmax=1.0,
        )
    for r in range(3):
        axes[r][0].text(
            -0.16, 0.5, rowlab[r], transform=axes[r][0].transAxes,
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
        )
    fig.suptitle("01_pd_51 — activity score, spatial fraction explained, and graph "
                 "fraction explained per component (KL-NMF, K=5)", fontsize=13)
    out = os.path.join(figdir, "components_overview.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    print("\nTop-5 metabolites per shown component:")
    for k in range(N_SHOW):
        order = np.argsort(V[k])[::-1][:5]
        print(f"  component {k}: " + ", ".join(mets[i] for i in order))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python manuscript/make_figures.py <parquet_path>")
    main(sys.argv[1])
