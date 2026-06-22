"""
demo_visium_dopamine.py — region-specific pathway components on a mouse-brain Visium section.

Demonstrates gait on spatial transcriptomics *counts* the multiplicatively-coherent
way (the point of the package): the routine "log1p-normalise then PCA" conflates the
multiplicative error model with the multiplicative coupling model exactly as for IMS, so
instead we keep the counts in **linear** space (a linear library-size normalisation only)
and fit KL/IS NMF directly — no log.

The illustrative question is one the single-section IMS data cannot ask: does a *single
pathway* carry *different dominating sub-programmes in different tissue regions*? We restrict
a mouse-brain section to a neurotransmission (neuroactive-ligand/receptor) pathway, decompose
it, and render the **dominant-component map** (per-spot argmax of the fraction explained):

    load 10x h5      -> raw counts (spots x genes) + coordinates + region labels
    library_normalize-> linear depth correction (no log)
    select pathway   -> neurotransmission gene set (case-insensitive: mouse or human)
    KL/IS-NMF        -> U (activity maps), V (gene loadings), loss chosen by diagnostic
    dominant_component + render_dominant_component  -> tissue segmented by sub-programme

Point it at a Space Ranger / GEO sample (e.g. GSE232910):

    python demo_visium_dopamine.py SAMPLE_filtered_feature_bc_matrix.h5 \
        SAMPLE_tissue_positions_list.csv [SAMPLE_region.csv]
"""

from __future__ import annotations

import sys

import numpy as np

from gait.decomposition import LinearNMF, variance_vs_mean
from gait.illustration import render_dominant_component
from gait.spatial import (
    dominant_component,
    library_normalize,
    load_visium_10x_h5,
    neurotransmission_gene_set,
    per_gene_scale,
    to_dense,
)


def main(matrix_h5: str, positions: str, regions: str | None, n_components: int = 5) -> None:
    # 1. Load raw counts + coordinates (+ region labels if provided).
    exp = load_visium_10x_h5(matrix_h5, positions, regions)
    print(f"[load] {exp.n_spots} spots x {exp.n_genes} genes (raw counts)")

    # 2. Linear library-size normalisation on the FULL matrix (no log), then pathway subset.
    Xnorm = library_normalize(exp.X)
    gidx, names = neurotransmission_gene_set(exp.genes)
    print(f"[genes] neurotransmission pathway: {len(names)} present")

    # 3. Choose the divergence from the counts themselves (variance-vs-mean), then fit.
    vm = variance_vs_mean(to_dense(exp.X[:, gidx]))
    print(f"[diagnostic] var~mean^{vm.slope:.2f} -> loss={vm.recommended_loss!r}")
    Xsub = per_gene_scale(Xnorm[:, gidx])          # linear, equalise gene influence
    res = LinearNMF(n_components, loss=vm.recommended_loss, max_iter=600,
                    random_state=0).fit(Xsub)
    for k in range(res.K):
        print(f"  comp {k}: {[names[i] for i in np.argsort(res.V[k])[::-1][:6]]}")

    # 4. Dominant-component map = which sub-programme dominates each spot.
    dom = dominant_component(res.U, res.V)
    labels = [names[int(np.argmax(res.V[k]))] for k in range(res.K)]
    render_dominant_component(
        dom, exp.x, exp.y, output="visium_dopamine_dominant.svg",
        component_names=labels, n_components=res.K,
        title="Neurotransmission pathway — dominant component",
    )

    # 5. If region labels are present, report which component dominates each region.
    if exp.region is not None:
        for lab in sorted(set(exp.region)):
            mask = exp.region == lab
            comp = np.bincount(dom[mask], minlength=res.K).argmax()
            print(f"[region] {lab!r}: dominated by comp {comp} ({labels[comp]}) "
                  f"in {100 * np.mean(dom[mask] == comp):.0f}% of its spots")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("error: pass MATRIX.h5 and TISSUE_POSITIONS.csv (and optionally REGION.csv).")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
