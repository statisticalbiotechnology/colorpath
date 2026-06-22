# GAIT

**Graph + Activity Image over Tissue** — multiplicatively-coherent **spatial pathway-activity
analysis** for imaging mass spectrometry and spatial transcriptomics.

GAIT decomposes a spatial-omics matrix `X` (locations × features — IMS pixels × m/z, or
Visium spots × genes) into `K` pathway components `X ≈ U V`, and renders each as a **pathway
activity graph** (the loading `V[k,:]` coloured over a feature network) and a **pathway
activity image** (the spatial score `U[:,k]` over the tissue). Where ORA/GSEA score *whether*
a pathway is up or down overall, GAIT resolves **how** a pathway is regulated in space — which
features carry the activity, and where distinct sub-programmes dominate.

It is built around two properties the conventional "log-then-PCA" pipeline gets wrong —
measurement error is *multiplicative* (R1), and pathway co-regulation is *multiplicative* (R2):

- **Route 2 (primary)** — Itakura–Saito / KL NMF in **linear** space, so `UV` is the
  multiplicative coupling and a scale-invariant loss supplies the multiplicative-error
  model, with a masked loss that right-censors detector-saturated entries.
- **Route 1 (secondary)** — `asinh` transform + equal-loading constrained NMF, for crisp
  near-binary pathway membership.

For **spatial transcriptomics**, treat counts like metabolites: keep them **linear** (a linear
`library_normalize`, never `log1p`) and let the KL/IS loss model the error — logging counts
re-introduces the very error-vs-coupling conflation GAIT exists to avoid.

See [`CLAUDE.md`](CLAUDE.md) for the full design and rationale, and `manuscript/` for the
write-up (`supplement.tex` gives the error-vs-coupling argument in full).

## Install

```bash
pip install -e .              # core (numpy, scipy, matplotlib, networkx)
pip install -e ".[visium]"    # + h5py, for 10x Space Ranger .h5 loading
```

(or, without packaging: `pip install -r requirements.txt`)

## Use — imaging mass spectrometry (metabolites)

```python
from gait.decomposition import LinearNMF, build_saturation_mask, variance_vs_mean
from gait.illustration import illustrate_component

mask, _ = build_saturation_mask(X, auto_detect=True)        # censor saturated ions
loss = variance_vs_mean(X).recommended_loss                 # "is" or "kl"
result = LinearNMF(n_components=4, loss=loss, n_init=4).fit(X, mask=mask)

illustrate_component(
    result, component=0,
    metabolite_names=metabolite_names, pathway_edges=edges,
    image_shape=(height, width),
)
```

## Use — spatial transcriptomics (genes)

```python
from gait.spatial import (
    load_visium_10x_h5, library_normalize, neurotransmission_gene_set,
    per_gene_scale, dominant_component,
)
from gait.decomposition import LinearNMF
from gait.illustration import render_dominant_component

exp = load_visium_10x_h5("sample_filtered_feature_bc_matrix.h5",
                         "sample_tissue_positions_list.csv")
Xn = library_normalize(exp.X)                        # linear depth correction (no log)
gidx, names = neurotransmission_gene_set(exp.genes)  # case-insensitive: mouse or human
res = LinearNMF(5, loss="kl").fit(per_gene_scale(Xn[:, gidx]))

dom = dominant_component(res.U, res.V)               # which sub-programme dominates each spot
render_dominant_component(dom, exp.x, exp.y, n_components=res.K)
```

## Demos & tests

```bash
python demo_decomposition.py                    # full pipeline on synthetic IMS data
python demo_visium_plasma.py DIR                # breast Visium: plasma-cell / Ig pathway
python demo_visium_dopamine.py H5 POSITIONS [REGION]  # mouse brain: region-specific components
python pathway_viz.py                           # original dopamine-pathway illustration
python -m pytest tests/ -q
```
