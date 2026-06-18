"""
make_figures.py — regenerate the manuscript figures from a MALDI-MSI parquet.

Reproduces the `01_pd_51` results shown in `main.tex`: a KL-NMF
decomposition (K=5) and, for the first three components, the pathway activity image
(spatial score on the tissue grid) and the pathway activity graph (loadings over the
catecholamine/serotonin network).

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

from colorpath.decomposition import LinearNMF, variation_explained
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

    # Per-metabolite fraction of variation explained by each component (scale-invariant,
    # so it is computed from the raw fit before any cosmetic renormalisation).
    F = variation_explained(res.U, res.V)            # (K, M) in [0, 1]

    # Normalise each component (unit-max spatial map; scale folded into loadings).
    U, V = res.U.copy(), res.V.copy()
    for k in range(K):
        s = U[:, k].max()
        if s > 0:
            U[:, k] /= s
            V[k] *= s

    # --- One combined figure: for each of the first N_SHOW components, the pathway
    #     activity image (top row) with its smaller pathway activity graph beneath. ---
    fig, axes = plt.subplots(
        2, N_SHOW, figsize=(6.0 * N_SHOW, 10),
        gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.04, wspace=0.04),
    )
    for k in range(N_SHOW):
        top = [mets[i] for i in np.argsort(V[k])[::-1][:3]]
        # top row: pathway activity image
        render_pathway_activity_image(
            scores=U[:, k], image_shape=(Hg, Wg), pixel_index=pix,
            colormap="magma", ax=axes[0][k], colorbar=True, title_fontsize=11,
            title=f"Component {k}\n" + ", ".join(top),
        )
        # bottom row: smaller pathway activity graph, coloured by the per-metabolite
        # fraction of variation this component explains (0-1), not the raw loading.
        abundance = {m: float(F[k, j]) for j, m in enumerate(mets)}
        draw_pathway(
            pathway=CATECHOLAMINE_SEROTONIN_EDGES, abundance=abundance,
            positions=CATECHOLAMINE_SEROTONIN_POSITIONS, colormap="magma",
            node_size=300, font_size=5.5, ax=axes[1][k], colorbar=False,
            title="", label_halo=True, vmin=0.0, vmax=1.0,
        )
    fig.suptitle("01_pd_51 — pathway activity image (top) and graph (bottom) per component "
                 "(KL-NMF, K=5)", fontsize=13)
    out = os.path.join(figdir, "components_overview.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
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
