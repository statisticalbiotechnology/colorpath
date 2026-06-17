"""
colorpath — pathway-activity decomposition and illustration of imaging mass spectrometry.

Two subpackages:

* :mod:`colorpath.decomposition` — the factorisation engine that turns an IMS matrix
  into K pathway components ``(U[:, k], V[k, :])`` (Route 2 linear-space IS/KL NMF;
  Route 1 asinh equal-loading NMF).
* :mod:`colorpath.illustration` — the visualisation layer: pathway activity graph
  (loadings over the metabolite network) and pathway activity image (spatial scores
  over the tissue), plus a bridge tying a component to both.
"""

__all__ = ["decomposition", "illustration"]
