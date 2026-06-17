# colorpath

Pathway-activity analysis of imaging mass spectrometry (IMS / MALDI-MSI).

`colorpath` decomposes an IMS image `X` (pixels × m/z) into `K` pathway components
`X ≈ U V`, then illustrates each component as a **pathway activity graph** (the spectral
loading `V[k,:]` coloured over a metabolite network) and a **pathway activity image**
(the spatial score `U[:,k]` reshaped onto the tissue).

The decomposition is designed around two properties the conventional "log-then-PCA"
pipeline gets wrong — measurement error is *multiplicative*, and pathway co-regulation is
*multiplicative*:

- **Route 2 (primary)** — Itakura–Saito / KL NMF in **linear** space, so `UV` is the
  multiplicative coupling and a scale-invariant loss supplies the multiplicative-error
  model, with a masked loss that right-censors detector-saturated entries.
- **Route 1 (secondary)** — `asinh` transform + equal-loading constrained NMF, for crisp
  near-binary pathway membership.

See [`CLAUDE.md`](CLAUDE.md) for the full design and rationale.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from colorpath.decomposition import LinearNMF, build_saturation_mask, variance_vs_mean
from colorpath.illustration import illustrate_component

mask, _ = build_saturation_mask(X, auto_detect=True)        # censor saturated ions
loss = variance_vs_mean(X).recommended_loss                 # "is" or "kl"
result = LinearNMF(n_components=4, loss=loss, n_init=4).fit(X, mask=mask)

illustrate_component(
    result, component=0,
    metabolite_names=metabolite_names, pathway_edges=edges,
    image_shape=(height, width),
)
```

```bash
python demo_decomposition.py   # full pipeline on synthetic IMS data
python pathway_viz.py          # original dopamine-pathway illustration
python -m pytest tests/ -q
```
