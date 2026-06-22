"""
demo_visium_plasma.py — gait on a Visium (spatial transcriptomics) breast section.

Reuses the IMS decomposition engine to pull a **plasma-cell / immunoglobulin pathway**
out of a human breast-cancer Visium HD export and render both gait views:

    load export            -> spots x genes matrix + coordinates
    select gene set        -> plasma-cell markers + immunoglobulin loci
    per-gene scale         -> equalise influence so the program consolidates
    KL-NMF (Route 2)       -> U (activity images), V (gene loadings)
    pick plasma component  -> the one most correlated with a plasma marker (IGKC)
    illustrate             -> activity graph (co-expression net, coloured by loading share)
                              + activity image (component score over the tissue section)

The data lives wherever you downloaded it; point the demo at that folder:

    python demo_visium_plasma.py /path/to/tmp_BC_all_genes
    # or:  GAIT_VISIUM_DIR=/path/to/tmp_BC_all_genes python demo_visium_plasma.py

The export is expected to contain gene_names_*.txt, a *.csr.npz (or *.mtx) counts matrix,
spot_barcodes.txt and spatial_coordinates.csv (see :mod:`gait.spatial`).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from gait.decomposition import LinearNMF
from gait.illustration import illustrate_component
from gait.spatial import (
    coexpression_edges,
    load_spatial_export,
    per_gene_scale,
    plasma_cell_gene_set,
    sample_grid,
)


def main(directory: str, n_components: int = 5, marker: str = "IGKC") -> None:
    # 1. Load the export and restrict to a single section for the activity image.
    exp = load_spatial_export(directory)
    print(f"[load] {exp.n_spots} spots x {exp.n_genes} genes; samples={exp.samples()}")
    sample = exp.samples()[0]
    rows = exp.sample_mask(sample)

    # 2. Select the plasma-cell / immunoglobulin gene set.
    gidx, names = plasma_cell_gene_set(exp.genes)
    print(f"[genes] plasma/Ig module: {len(names)} genes present")

    # 3. Restrict to the section's spots + the module's genes, equalise gene influence.
    Xsub = exp.X[rows][:, gidx]
    Xs = per_gene_scale(Xsub)

    # 4. Route 2 — KL-NMF (Poisson/counting-noise divergence) in (normalised) linear space.
    res = LinearNMF(n_components, loss="kl", max_iter=400, random_state=0).fit(Xs)
    print(f"[route2] KL-NMF final_div={res.reconstruction_error:.3f} K={res.K}")

    # 5. Pick the component whose spatial score tracks the plasma marker.
    ref = Xs[:, names.index(marker)] if marker in names else Xs.sum(1)
    corr = [np.corrcoef(res.U[:, k], ref)[0, 1] for k in range(res.K)]
    k = int(np.nanargmax(corr))
    top = [names[j] for j in np.argsort(res.V[k])[::-1][:10]]
    print(f"[select] plasma component = {k}; top genes: {top}")

    # 6. Illustrate. Co-expression edges within the program-active spots (top decile);
    #    graph coloured by loading share so low-abundance machinery genes stay visible.
    active = res.U[:, k] > np.percentile(res.U[:, k], 90)
    edges = coexpression_edges(Xs, names, active=active, threshold=0.30)
    print(f"[graph] {len(edges)} co-expression edges among {len(names)} genes")

    image_shape, pixel_index = sample_grid(exp.x[rows], exp.y[rows])
    vmax = float(np.percentile(res.U[:, k], 99))      # robust colour scale
    paths = illustrate_component(
        res, component=k,
        metabolite_names=names,
        pathway_edges=edges,
        image_shape=image_shape,
        pixel_index=pixel_index,
        graph_value="loading_share",
        graph_colormap="viridis",
        image_colormap="magma",
        title_prefix="Plasma-cell / Ig component",
        graph_output="visium_plasma_graph.svg",
        image_output="visium_plasma_image.svg",
        graph_kwargs=dict(layout="spring"),
        image_kwargs=dict(vmin=0.0, vmax=vmax, background=np.nan),
    )
    print(f"[illustration] {paths}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GAIT_VISIUM_DIR")
    if not path:
        print(__doc__)
        print("error: pass the export directory as an argument or set "
              "GAIT_VISIUM_DIR.")
        sys.exit(0)
    main(path)
