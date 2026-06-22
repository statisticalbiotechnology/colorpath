"""
gait — pathway-activity decomposition and illustration of imaging mass spectrometry.

Two subpackages:

* :mod:`gait.decomposition` — the factorisation engine that turns an IMS matrix
  into K pathway components ``(U[:, k], V[k, :])`` (Route 2 linear-space IS/KL NMF;
  Route 1 asinh equal-loading NMF).
* :mod:`gait.illustration` — the visualisation layer: pathway activity graph
  (loadings over the metabolite network) and pathway activity image (spatial scores
  over the tissue), plus a bridge tying a component to both.

The same engine applies unchanged to **spatial transcriptomics** (Visium / Visium HD),
where rows are spots and columns are genes; :mod:`gait.spatial` adds the I/O and
bookkeeping to load such an export and feed it to the decomposition + illustration layers
(see ``demo_visium_plasma.py``).
"""

__all__ = ["decomposition", "illustration", "spatial"]
__version__ = "0.1.0"
