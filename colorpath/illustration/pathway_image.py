"""
pathway_image.py — the pathway *activity image* renderer.

Companion to :func:`pathway_graph.draw_pathway`. Where the graph renders a component's
spectral loading ``V[k, :]`` over the metabolite network, this renders a component's
spatial score ``U[:, k]`` back onto the tissue: the per-pixel activity vector is
reshaped to the acquisition grid ``(height, width)`` and drawn as a heatmap. Together
the two objects are colorpath's full description of one pathway component.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_pathway_activity_image(
    scores: np.ndarray,
    image_shape: tuple[int, int],
    output: str = "pathway_image.svg",
    title: str = "Pathway activity image",
    colormap: str = "viridis",
    colorbar_label: str = "Pathway activity",
    figsize: tuple[float, float] = (6, 5),
    pixel_index: np.ndarray | None = None,
    background: float = np.nan,
    ax=None,
    colorbar: bool = True,
    title_fontsize: int = 13,
) -> None:
    """Render a component's per-pixel spatial score ``U[:, k]`` as a tissue heatmap.

    Parameters
    ----------
    scores      : (P,) spatial score for one component (a column of ``U``).
    image_shape : (height, width) of the acquisition grid. ``height * width`` must equal
                  ``P`` unless ``pixel_index`` is given.
    output      : output path (.svg/.pdf for vector, .png for raster). Ignored when ``ax``
                  is supplied.
    title       : figure title.
    colormap    : matplotlib colormap name.
    colorbar_label : colour bar label.
    figsize     : figure size in inches (only when ``ax`` is None).
    pixel_index : optional (P,) integer flat indices placing each score on the grid when
                  pixels are a sparse/irregular subset of the full rectangle. Missing
                  grid cells are filled with ``background``.
    background  : value for grid cells with no measured pixel (NaN -> rendered blank).
    ax          : optional matplotlib Axes to draw into; when given, the renderer draws
                  onto it and does not create, save, or close a figure (use this to
                  compose several components into one figure).
    colorbar    : whether to attach a colour bar.
    title_fontsize : title font size.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    h, w = image_shape

    if pixel_index is not None:
        pixel_index = np.asarray(pixel_index, dtype=int).ravel()
        if pixel_index.shape != scores.shape:
            raise ValueError("pixel_index must have the same length as scores")
        img = np.full(h * w, background, dtype=float)
        img[pixel_index] = scores
        img = img.reshape(h, w)
    else:
        if scores.size != h * w:
            raise ValueError(
                f"scores has {scores.size} pixels but image_shape implies {h * w}; "
                f"pass pixel_index for an irregular pixel set"
            )
        img = scores.reshape(h, w)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(img, cmap=colormap, origin="upper", interpolation="nearest")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=10)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    if created:
        fig.tight_layout()
        fig.savefig(output, format=output.rsplit(".", 1)[-1], bbox_inches="tight")
        print(f"Saved: {output}")
        plt.close(fig)
