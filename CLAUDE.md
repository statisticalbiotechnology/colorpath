# CLAUDE.md — GAIT

Guidance for working in this repository. `GAIT` does **pathway-activity analysis of
imaging mass spectrometry (IMS / MALDI-MSI)**: it decomposes an IMS image into pathway
components and illustrates each one as a *pathway activity graph* (loadings over the
metabolite network) and a *pathway activity image* (spatial scores over the tissue).

## Repository map

```
gait/
  __init__.py
  decomposition/            # the factorisation engine (this is the analysis core)
    losses.py               # frobenius / kl / is divergences + masked variants
    nmf_linear.py           # Route 2 (PRIMARY): masked IS/KL NMF in linear space, MU + warm start
    nmf_independent.py      # Route 2+: IS/KL NMF with an ICA-style HSIC independence penalty
    nmf_loglevel.py         # Route 1 (SECONDARY): asinh transform + equal-loading constrained NMF
    saturation.py           # detector-ceiling detection, mask construction, histograms
    diagnostics.py          # variance-vs-mean, compensation-artifact, recovery checks
    selection.py            # K selection (elbow), best-of n_init restarts
  illustration/             # the visualisation layer — REUSE, do not reimplement
    pathway_graph.py        # draw_pathway: metabolite network coloured by a loading vector
    pathway_image.py        # render_pathway_activity_image: spatial score -> tissue heatmap;
                            #   render_dominant_component: categorical "which component dominates" map
    bridge.py               # illustrate_component: one component -> both views
  spatial.py                # spatial-transcriptomics (Visium/HD) I/O + helpers: load export or 10x
                            #   Space Ranger .h5, case-insensitive gene-set select, library_normalize
                            #   (LINEAR depth correction, no log), per-gene scale, grid index,
                            #   co-expression edges, dominant_component (reuses the engine)
  benchmark.py              # log-vs-linear recovery benchmark: planted multiplicative coupling,
                            #   GAIT vs linear-NMF vs log1p+NMF/PCA; region mutual information;
                            #   regional_structure_score (coherence x diversity) + direction term
pathway_viz.py              # backward-compatible shim (re-exports draw_pathway) + dopamine example
demo_decomposition.py       # end-to-end synthetic pipeline (diagnostics -> Route 2 -> illustrate -> Route 1)
demo_visium_plasma.py       # end-to-end on a Visium breast section: plasma-cell/Ig pathway
demo_visium_dopamine.py     # mouse-brain Visium: a neurotransmission pathway -> region-specific
                            #   components (linear counts + KL, no log) + dominant-component map
robustness_multisection.py  # run a pathway across all GEO _RAW.tar sections; striatum/lesion AUC per section
scan_pathways.py            # annotation-free screen: rank every pathway in a .gmt (KEGG) by
                            #   regional_structure_score on one or many Visium sections
tests/test_decomposition.py
tests/test_spatial.py
tests/test_benchmark.py
tests/test_scan.py
```

**Spatial transcriptomics counts:** treat counts like the metabolites — stay in **linear**
space (`library_normalize` is a linear depth correction; **do not `log1p`**) and let the
KL/IS loss model the error. Logging counts re-introduces exactly the error-vs-coupling
conflation the package exists to avoid.

## Conventions

- Python ≥ 3.11. Dependencies: `numpy`, `scipy`, `matplotlib`, `networkx` (see
  `requirements.txt`). `matplotlib` uses the `Agg` backend (file output, no display).
- Public API is small and stable: a `fit(...)` returning `U`, `V` (linear, Route 2) or
  `g`, `p` (asinh, Route 1), plus the fitted mask and diagnostics, all feeding the
  existing illustration renderers. Keep new decomposition code under
  `gait/decomposition/`; **reuse `gait/illustration/`, do not rewrite it.**
- Match the surrounding docstring style (NumPy-ish, parameter blocks) and keep modules
  importable without side effects.
- Figures are output artifacts — do not commit `*.svg`/`*.png` produced by demos/tests.

## Commands

```bash
pip install -r requirements.txt
python -m pytest tests/ -q        # unit tests
python demo_decomposition.py      # full pipeline on synthetic IMS data
python pathway_viz.py             # original dopamine-pathway illustration (still works)
```

---

# Pathway-activity decomposition of imaging mass spectrometry

The sections below are the scientific design the decomposition engine implements. The
illustration layer (`gait/illustration/`) is the visualisation layer for the two
vectors each component produces — reuse it.

## 1. Scientific context and goal

We analyse IMS / MALDI-MSI data as a matrix

```
X  ∈  R≥0^{P × M}          P pixels (rows), M metabolites/ions (columns)
X ≈ U V                    U ∈ R≥0^{P × K}  spatial scores  (one image per component)
                           V ∈ R≥0^{K × M}  spectral loadings (one metabolite vector per component)
```

Each rank-1 component `U[:,k] ⊗ V[k,:]` is interpreted as a **pathway**:

- `V[k,:]` — the **pathway activity graph** (which metabolites belong / how they load) →
  `gait.illustration.draw_pathway`,
- `U[:,k]` — the **pathway activity image** (where in the tissue the pathway is active) →
  `gait.illustration.render_pathway_activity_image`.

`illustrate_component` renders both from a fitted result. Because metabolite concentrations
span orders of magnitude, colouring graph nodes by the raw loading `V[k,:]` looks near-binary
(a few abundant ions saturate the scale); pass `graph_value="explained"` to colour each
metabolite instead by the **fraction of its variation the component explains**
(`decomposition.variation_explained`: `V[k,m]²·Var_p(U[:,k])` normalised per metabolite,
∈[0,1], scale-invariant). The factorisation must satisfy
two properties the conventional PCA/ICA/NNMF (or "log-then-PCA") baseline does **not**.

### 1.1 The two requirements that drive the design

**(R1) Errors are multiplicative.** IMS measurement error scales with signal
(`x = μ·ε`, `ε>0`). The conventional fix is to log-transform so the error becomes
additive and MSE/PCA applies. But logging breaks (R2).

**(R2) Pathway coupling is multiplicative.** If compounds A and B are in the same
pathway, doubling pathway activity in a pixel doubles (or halves, with a sign) both A
and B. In **linear** space a rank-1 component `u·vᵀ` encodes exactly this: the outer
product is a product of a per-pixel activity and a per-metabolite loading. **After log,
this is destroyed** — `log(a·s)` and `log(b·s)` share an *additive* term `log s`, not a
multiplicative one. A linear factorisation of logged data therefore studies
additive-shared-log structure, a weaker and different statement than multiplicative
co-regulation.

The crux: log can serve the error model **or** the coupling model, not both. The two
routes below resolve this in complementary ways. **Route 2 is primary; implement/extend
it first, then Route 1.**

## 2. Route 2 (PRIMARY): Itakura–Saito NMF in linear space — `nmf_linear.py`

**Idea.** Do not log. Stay in linear space so the outer product `UV` *is* the
multiplicative pathway coupling (R2 holds by construction). Obtain the
multiplicative-error property (R1) from the *loss function* via a scale-invariant
divergence.

**Loss — Itakura–Saito (IS) divergence**, elementwise and summed:

```
D_IS(x, y) = x/y − log(x/y) − 1            (y = (UV)_{pm})
```

IS is scale-invariant: `D_IS(λx, λy) = D_IS(x, y)`. It penalises the **ratio** `x/y`,
so a factor-of-two error costs the same at high and low abundance — the property we log
for, obtained without leaving linear space. This gives **both R1 and R2 simultaneously**.

**Noise-model selection (decide empirically, §4).** IS ⇔ multiplicative Gamma-type gain
noise (variance ∝ mean²). If the dominant noise is ion-**counting/shot** noise
(variance ∝ mean), use the **KL / Poisson** divergence `D_KL(x,y) = x log(x/y) − x + y`.
Both are implemented behind `loss="is"` / `loss="kl"`; pick via the variance-vs-mean
diagnostic (`diagnostics.variance_vs_mean`).

**Saturation handling.** High-abundance ions saturate the detector; clipped intensity
reappears in components 2/3 as spurious "compensation". Handled by a **masked loss**: a
weight matrix `W ∈ {0,1}^{P×M}` zeroes out saturated `(pixel, ion)` entries so they are
treated as right-censored, not data:

```
minimise   Σ_{p,m}  W_{pm} · D(X_{pm}, (UV)_{pm})      s.t. U,V ≥ 0
```

IS/KL are defined on non-negative linear data, so the mask ports in directly (a reason
Route 2 composes better with saturation handling than a log+MSE route). Mask
construction (`saturation.build_saturation_mask`): flag `X_{pm}` above a per-ion
saturation ceiling, detected from the per-ion intensity histogram (a pile-up spike ⇒
hard clipping ⇒ mask; a smoothly bending tail ⇒ soft compression ⇒ masking optional).
`saturation_quantile` / explicit-ceiling / `auto_detect` options; **default off**, warns
if a ceiling pile-up is detected.

**Optimisation.** Masked multiplicative-update (MU) rules for IS-NMF and KL-NMF
(Févotte–Bertin–Durrieu / Lee–Seung), with the mask `W` carried inside both numerator
and denominator. IS-NMF is non-convex and MU can be unstable, so: **warm-start IS from a
few KL iterations** (`warm_start_iter`), floor by `EPS` to avoid division by zero,
`max_iter`/`tol` on relative loss change, and `n_init` best-of restarts (`random_state`).

**Outputs.** `U` (pathway activity images) and `V` (pathway activity graphs) on
interpretable **linear-abundance** units, ready for the illustration layer.

## 2b. Route 2+ (OPTIONAL): ICA-style independent components — `nmf_independent.py`

Plain NMF (even with the IS/KL loss) leaves components **correlated**; it never enforces
the statistical *independence* ICA targets. `IndependentNMF` adds independence while
keeping both principles — it does **not** whiten/rotate (that would break non-negativity
and the multiplicative outer product). It keeps the masked IS/KL fidelity term and adds a
mutual-information penalty:

```
minimise_{U,V ≥ 0}   D_IS/KL(X ‖ UV)  +  λ · Σ_{i<j} HSIC(U[:,i], U[:,j])
```

**Why HSIC.** The Hilbert–Schmidt Independence Criterion with a *characteristic* (RBF)
kernel is zero **iff** the two variables are independent (iff their mutual information is
zero), so minimising it pushes components to MI-sense orthogonality. This is exactly the
kernel dependence contrast of **Kernel ICA (Bach–Jordan 2002)**, used here as a penalty on
the non-negative factor rather than as a rotation objective. A *linear* kernel
(`kernel="linear"`) recovers plain second-order decorrelation only — **not** independence.

**Implementation.** RBF HSIC and its gradient are approximated with **random Fourier
features** (`n_features`), giving O(P·D) cost. Optimisation warm-starts from a plain Route
2 fit, then alternates a masked-MU update for `V` with a **projected-gradient + Armijo**
step for `U` on the deterministic (data + λ·penalty) objective. `λ` is **dimensionless**:
the penalty gradient is rescaled to the data gradient's norm each step, so `λ=0` is plain
Route 2 and `λ≈1` pushes independence about as hard as the data term. `target="U"`
penalises the spatial maps (spatial-ICA analogue, default); `target="V"` the loadings.

**Diagnostic.** `normalised_hsic_matrix` reports pairwise normalised HSIC (centred kernel
alignment, ∈ [0,1]; 0 ≈ independent) using a direct RBF kernel — an *independent* check
that the penalty achieved independence. Increasing `λ` trades a modest rise in
reconstruction error for components whose off-diagonal dependence falls toward 0.

## 3. Route 1 (SECONDARY): equal-loading constrained log-space NMF — `nmf_loglevel.py`

**Motivation.** Route 2 fixes the pathway response *exponent* at 1 (A and B scale
identically with activity). Route 1 is better when the deliverable is crisp, near-binary
**pathway membership**, and we accept a log-space error model.

**Key reframing.** For a rank-1 pathway `A = a·s`, `B = b·s`:
`log A = log a + log s`, `log B = log b + log s`. The shared multiplicative activity `s`
becomes a **common additive offset across the pathway's members, varying over pixels** —
i.e. multiplicative coupling in linear space ⇔ an **additive, equal-loading** rank-1
component in log space (loading constant over members, spatial score = `log s`).

**Model.** Factorise `Y = asinh(X/c) ≈ Σ_k g_k ⊗ p_k` (multiplicative→additive error,
R1) but **constrain each loading `p_k` toward equal/binary membership** instead of a
free vector. `exp(g_k)` (here `sinh(g_k)·c`) recovers the multiplicative activity (R2).

**Implementation notes.**
- `asinh(x/c)` (not plain log): defined at zero, linear near origin / logarithmic for
  large x; absorbs soft (Case A) saturation. Loadings are then on a transformed, not
  linear, scale — document this when illustrating.
- Penalty: elastic-net on `p_k` plus a within-component **equalisation** step driving
  nonzero member loadings to a shared level (`l1`, `l2`, `equalise`, `active_frac`).
- Initialise from the Route 2 solution (`init=(U, V)`) so the two routes share component
  identity and can be compared component-by-component.

**When to prefer which.** Route 2 = continuous coupling, linear units, fixed exponent 1,
best saturation composability — **default**. Route 1 = interpretable membership,
asinh error model, equal-loading abstraction — **run second, compare**.

## 4. Diagnostics (run before trusting either route) — `diagnostics.py`, `saturation.py`

1. **Saturation / mechanism check.** Per-ion intensity histogram. Pile-up at a ceiling ⇒
   hard clipping (mask). Smooth bending tail ⇒ soft compression (asinh).
2. **Compensation-artifact check** (`compensation_artifact_check`). For a suspected
   saturating ion, verify its loading in components 2/3 is spatially anti-correlated with
   its component-1 image (high where component 1 clips) — confirming the reappearance is
   a measurement nonlinearity, not biological co-localisation.
3. **Noise-model selection** (`variance_vs_mean`). Per-ion variance vs mean across
   pixels: variance ∝ mean² ⇒ IS; variance ∝ mean ⇒ KL.
4. **Component-recovery sanity** (`component_recovery`). Confirm that moving from
   Frobenius-NMF to IS/KL + mask collapses the spurious compensating components.

## 5. Public API

`fit` returns `U`, `V` (linear, Route 2) or `g`, `p` (asinh, Route 1), plus the fitted
mask and diagnostics, all feeding the pathway-activity-graph and pathway-activity-image
renderers via `illustrate_component`. See `demo_decomposition.py` for the wiring.

## 6. Related work and novelty (for write-ups)

Matrix factorisation on IMS (PCA/ICA/NNMF over pixel × m/z) is the established baseline
(Siy et al. 2007; Leuschner et al. 2019 — rows are spectra, columns are
pseudo-channels). Plain MF ignores spatial adjacency; Fernsel (2021) adds a
total-variation penalty (a **clustering**, orthogonal-NMF method — optional future
extension, not core). "Component → pathway" is imposed afterwards via annotate-then-
enrich: Jones et al. (2014), and **Wittmann et al. S2IsoMEr / METASPACE (2025)**, which
bootstraps over isomer/isobar ambiguity (MS1, MSI Level 2) and uses **RAMP-DB**
(SMPDB/Reactome/KEGG/WikiPathways) for pathway sets — use this as the mapping layer, not
a home-rolled KEGG lookup. MSI confounder: matrix-adduct formation co-localises with
parents (Janda et al. mass2adduct 2021) — note in limitations.

**Novelty (state plainly, do not over-claim):** the factorisation itself is not novel.
The defensible core is the **error-vs-coupling resolution** — modelling pathway
co-regulation as *multiplicative* and matching it to a *multiplicative* error model (IS/
KL-NMF in linear space; the equal-loading log-space reframing) — which "log-then-PCA"
gets wrong. The likely publishable increment is coupling **bootstrapped, isomer-aware
pathway enrichment** (S2IsoMEr-style) to a **per-component loading vector from a
multiplicatively-correct, saturation-aware factorisation** (ranking ions by loading on
spatial factor k). Saturation handling via a censored/masked NMF loss is a practical
novelty in this combination. Frame the contribution as the *pipeline and its statistical
coherence*, not any single algorithm.

---

# 7. Project state & cross-modality continuation (read this to resume in a new thread)

This section is the **handoff**: where the project stands and exactly what to do when the
matched MALDI-MSI metabolite data arrives. Everything below is committed on the dev branch
`claude/peaceful-mendel-efv8ps`; `main` is fast-forwarded from there by hand.

## 7.1 What is done

- **Engine + illustration**: Routes 1/2/2+ as in §§2–3; tests green (`pytest tests/ -q`).
- **Spatial-transcriptomics extension** (`gait/spatial.py`): Visium / 10x Space Ranger `.h5`
  loaders, gene-set select (case-insensitive), `library_normalize` (LINEAR depth correction —
  **never `log1p`**), `per_gene_scale`, `dominant_component`. Counts are treated exactly like
  metabolites: linear space + KL/IS loss.
- **Log-vs-linear benchmark** (`gait/benchmark.py`, `run_synthetic_benchmark`): formalises the
  R1/R2 claim — GAIT (KL/IS on linear X) recovers planted multiplicative coupling better than
  `log1p`+NMF/PCA. Table `tab:benchmark` in the manuscript.
- **Cross-section robustness** (`robustness_multisection.py`): a pathway run across all GEO
  `GSE232910` sections, scored by best **striatum-vs-rest AUC** (or **intact-vs-lesioned** with
  `--region lesion`) over `K=5` components. Striatal systems (dopaminergic 0.90, broad
  neurotransmission 0.95, GABAergic 0.72) localise reproducibly; non-striatal (serotonergic
  0.62, noradrenergic 0.53) sit at chance — *specificity* is the evidence.
- **Annotation-free pathway scan** (`scan_pathways.py`): ranks every KEGG gene set by
  `regional_structure_score` = **coherence x diversity** (dominant-share-weighted Moran's I of
  component activities, times entropy of the dominant-component map). `--match V11L12`
  aggregates by median score/rank over the eight striatal sections. The unsupervised scan
  reproduces the supervised ordering (synaptic / neuro-signalling pathways on top).
- **Magnitude-vs-direction**: `regional_structure_score` also returns `direction` =
  `1 - max_{i<j} cos(V_i, V_j)` over the components that own territory (see
  `loading_direction_distinctness`). It is scale-invariant in gene space and separates "the
  pathway is merely *more abundant* in a region" (a magnitude gradient NMF carved into
  near-collinear components; `direction ~ 0`) from "the pathway runs a *different program*
  there" (different gene composition; `direction -> 1`). On the striatal scan the top hits sit
  at an **intermediate** 0.3–0.5: real but moderate compositional structure on top of an
  abundance contrast. This is the honest framing in the manuscript.

## 7.2 The dataset and the open question

The data is **Vicari et al., Spatial Multimodal Analysis (SMA), *Nat. Biotechnol.* 2023**: the
same mouse-brain sections measured by **MALDI-MSI** (FMP-10 derivatisation, which charge-tags
monoamines/amino acids — dopamine, L-DOPA, 3-MT, DOPAC, HVA, norepinephrine, 5-HT, GABA,
glutamate…) **and** by **Visium** spatial transcriptomics, in a 6-OHDA unilateral Parkinson's
model. We have analysed only the **transcriptomic** side so far (GEO `GSE232910`).

**The headline prediction (the reason for the cross-modality step).** On transcripts the
dopaminergic programme localises cleanly to the striatum but is at **chance on the
intact-vs-lesioned axis** (`--region lesion`, AUC ≈ 0.55): 6-OHDA depletes the *metabolite*
dopamine in the lesioned hemisphere but leaves the receptor *transcripts* (`Drd1`/`Drd2`)
largely intact, so the transcriptomic programme cannot see the lesion. GAIT therefore predicts
that the **dopamine-metabolite component (MSI) should separate the lesioned hemisphere where
the transcript component does not** — a transcript-vs-metabolite dissociation only a multimodal
measurement resolves. Confirming this on the FMP-10 data is the manuscript's payoff and decides
the journal tier. Until then the relevant manuscript subsection is fenced as **PROVISIONAL**
(`%% ---- PROVISIONAL ... %% ---- end PROVISIONAL ----` around `sec:robustness` in
`manuscript/main.tex`).

## 7.3 When Fei's parquet arrives — the plan

We have asked Fei for the **dopaminergic part of the IMS data as a parquet** (see §7.4 for the
schema). To resume in a fresh thread:

1. **Write a loader** `load_msi_parquet(path) -> SpatialExport`-like object in `gait/spatial.py`
   (mirror `SpatialExport`: `X` = pixels × metabolites linear intensities, `genes` -> metabolite
   names, `x`, `y` = pixel coords, `sample` = section, `region` = intact/lesioned). Add a test in
   `tests/test_spatial.py` on a tiny synthetic frame. **Do not `log`** the intensities; pass them
   to `LinearNMF(loss=...)` directly. Pick the loss with the variance-vs-mean diagnostic
   (`diagnostics.variance_vs_mean`) — MSI gain noise is usually variance ∝ mean² ⇒ **IS**, but
   verify; counts-like ⇒ KL.
2. **Run saturation diagnostics first** (§4): FMP-10 monoamine ions can saturate; check the
   per-ion histogram and, if a pile-up is present, fit with `build_saturation_mask`.
3. **Decompose** the dopaminergic metabolite block with Route 2 (`LinearNMF`), `K` small.
   `illustrate_component` for the pathway-activity graph (use the existing catecholamine edge
   set in `gait/illustration`) and image. Compare the **spatial pattern** to the transcriptomic
   dopaminergic component from `demo_visium_dopamine.py` — they are *not* aligned pixel-to-pixel
   (different modalities, adjacent sections), so compare *region-level* structure, not per-pixel.
4. **Test the prediction**: score the dopamine-metabolite component on the intact-vs-lesioned
   axis (reuse `region_auc` from `robustness_multisection.py`). Expectation: the **metabolite**
   component separates the lesion (high AUC) while the **transcript** component did not (~0.55).
   If it holds, that is the multimodal dissociation — promote the PROVISIONAL subsection and add
   a side-by-side figure (transcript map vs metabolite map vs lesion annotation).
5. Also report the **`direction`** term for the metabolite component: with FMP-10 the
   dopaminergic ions are genuinely different molecules, so a real lesion effect should show as a
   magnitude collapse in the lesioned hemisphere, not merely a re-weighting.

## 7.4 Expected parquet schema (what we asked Fei for)

One **row per pixel**; columns:

- `section` (or `sample`) — section / slide id (string);
- `x`, `y` — pixel coordinates (µm or pixel index);
- `region` / `condition` — *optional but wanted*: intact vs 6-OHDA-lesioned hemisphere (or
  anatomical region);
- one **numeric column per metabolite**, **linear** intensities (raw or TIC-normalised; **not**
  log-transformed), named by annotation (`dopamine`, `DOPA`, `3-MT`, `DOPAC`, `HVA`,
  `norepinephrine`, …) and/or m/z. Ideally an accompanying m/z + annotation key so the pathway
  graph can be labelled.

If Fei sends all FMP-10-annotated peaks rather than just the dopaminergic subset, select the
dopaminergic metabolites in the loader.

## 7.5 Environment / workflow notes

- The repo was renamed `colorpath` -> `gait` (package dir `gait/`, `pyproject.toml` name `gait`,
  pip-installable; `[visium]` extra pulls `h5py`, `[figures]` extra pulls `pandas`/`pyarrow` —
  `pyarrow` is what reads the parquet). Continuing **locally** avoids the remote sandbox's
  no-egress limit and the post-rename push-to-`main` restriction.
- Manuscript: `manuscript/main.tex` (+ `ref.bib`, `Makefile`, `make_figures.py`). Title:
  *"Multiplicatively-coherent spatial pathway-activity analysis…"*. Build with
  `cd manuscript && make main.pdf`. Do not commit generated PDFs/figures.
- Key references already in `ref.bib`: GSEA (subramanian2005), pathway-analysis review
  (khatri2012), gene-set null methodology (goeman2007), MIPath (jeuken2024), survival pathway
  activity (jeuken2022), plus the IMS/NMF/HSIC stack.
