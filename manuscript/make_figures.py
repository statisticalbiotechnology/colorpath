"""
make_figures.py — regenerate the manuscript figures from a MALDI-MSI parquet.

Reproduces the `01_pd_51` results shown in `colorpath_manuscript.md`: a KL-NMF
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

from colorpath.decomposition import LinearNMF
from colorpath.illustration import (
    CATECHOLAMINE_SEROTONIN_EDGES,
    CATECHOLAMINE_SEROTONIN_POSITIONS,
    illustrate_component,
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

    # Normalise each component (unit-max spatial map; scale folded into loadings).
    U, V = res.U.copy(), res.V.copy()
    for k in range(K):
        s = U[:, k].max()
        if s > 0:
            U[:, k] /= s
            V[k] *= s
    res.U, res.V = U, V

    # --- Figure: montage of the first N_SHOW pathway activity images ---
    fig, axes = plt.subplots(1, N_SHOW, figsize=(4 * N_SHOW, 6))
    for k in range(N_SHOW):
        img = np.full(Hg * Wg, np.nan)
        img[pix] = U[:, k]
        im = axes[k].imshow(img.reshape(Hg, Wg), cmap="magma", origin="upper")
        top = [mets[i] for i in np.argsort(V[k])[::-1][:3]]
        axes[k].set_title(f"Component {k}\n" + ", ".join(top), fontsize=9)
        axes[k].axis("off")
        fig.colorbar(im, ax=axes[k], shrink=0.6)
    fig.suptitle("01_pd_51 — pathway activity images (KL-NMF, K=5)", fontsize=12)
    fig.tight_layout()
    montage = os.path.join(figdir, "components_images.png")
    fig.savefig(montage, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", montage)

    # --- Per-component pathway activity graphs ---
    for k in range(N_SHOW):
        illustrate_component(
            res, component=k, metabolite_names=mets,
            pathway_edges=CATECHOLAMINE_SEROTONIN_EDGES,
            image_shape=(Hg, Wg), pixel_index=pix,
            positions=CATECHOLAMINE_SEROTONIN_POSITIONS,
            graph_output=os.path.join(figdir, f"component{k}_graph.png"),
            image_output=os.path.join(figdir, f"component{k}_image.png"),
            graph_colormap="magma", image_colormap="magma",
            graph_kwargs=dict(figsize=(18, 10), node_size=2400, font_size=9),
        )

    print("\nTop-5 metabolites per shown component:")
    for k in range(N_SHOW):
        order = np.argsort(V[k])[::-1][:5]
        print(f"  component {k}: " + ", ".join(mets[i] for i in order))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python manuscript/make_figures.py <parquet_path>")
    main(sys.argv[1])
