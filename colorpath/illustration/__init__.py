"""
colorpath.illustration — the visualisation layer (pathway activity graph + image).

* :func:`draw_pathway` — the original colorpath renderer: a metabolite network coloured
  by a per-metabolite scalar (a component's spectral loading ``V[k, :]``).
* :func:`render_pathway_activity_image` — a component's per-pixel spatial score
  ``U[:, k]`` reshaped onto the tissue grid as a heatmap.
* :func:`illustrate_component` — bridge that renders both views of one decomposition
  component in a single call.
"""

from .pathway_graph import draw_pathway
from .pathway_image import render_pathway_activity_image
from .bridge import illustrate_component

__all__ = ["draw_pathway", "render_pathway_activity_image", "illustrate_component"]
