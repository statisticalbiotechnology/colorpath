"""
pathway_image.py — the pathway *activity image* renderer.

Companion to :func:`pathway_graph.draw_pathway`. Where the graph renders a component's
spectral loading ``V[k, :]`` over the metabolite network, this renders a component's
spatial score ``U[:, k]`` back onto the tissue: the per-pixel activity vector is
reshaped to the acquisition grid ``(height, width)`` and drawn as a heatmap. Together
the two objects are gait's full description of one pathway component.
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
    vmin: float | None = None,
    vmax: float | None = None,
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
    vmin, vmax  : fix the colour-scale limits. Leaving them None autoscales to the data,
                  which a few extreme pixels can blow out (everything else then renders
                  flat); pass e.g. ``vmax=np.percentile(scores, 99)`` for a robust scale.
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

    im = ax.imshow(img, cmap=colormap, origin="upper", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
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


def render_dominant_component(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output: str = "dominant_component.svg",
    *,
    component_names: list[str] | None = None,
    n_components: int | None = None,
    title: str = "Dominant pathway component",
    colormap: str = "tab10",
    figsize: tuple[float, float] = (6, 6),
    marker_size: float = 8.0,
    invert_yaxis: bool = True,
    legend: bool = True,
    ax=None,
) -> None:
    """Categorical map of which component dominates each spot.

    Companion to :func:`render_pathway_activity_image`: instead of one component's
    continuous score, this scatters the tissue spots coloured by a categorical label —
    typically :func:`gait.spatial.dominant_component` (the per-spot argmax of the
    fraction-of-variation-explained). When a pathway is decomposed into K sub-programmes,
    this segments the tissue by which sub-programme is locally strongest, so a single
    pathway can reveal anatomically distinct regions.

    Parameters
    ----------
    labels          : (P,) integer component index per spot.
    x, y            : (P,) spot coordinates (e.g. full-res pixels or grid indices).
    output          : output path (ignored when ``ax`` is supplied).
    component_names : optional length-K labels for the legend (defaults to ``comp k``).
    n_components    : K, inferred from ``labels`` when None.
    colormap        : a qualitative matplotlib colormap.
    invert_yaxis    : flip y so the image matches tissue orientation.
    legend          : draw a legend mapping colours to components.
    ax              : optional Axes to draw into (no figure is created/saved then).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    K = n_components if n_components is not None else int(labels.max()) + 1
    cmap = matplotlib.colormaps[colormap]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for k in range(K):
        sel = labels == k
        if not sel.any():
            continue
        name = component_names[k] if component_names is not None else f"comp {k}"
        ax.scatter(x[sel], y[sel], s=marker_size, color=cmap(k % cmap.N), label=name)

    ax.set_aspect("equal")
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold")
    if legend:
        ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)

    if created:
        fig.tight_layout()
        fig.savefig(output, format=output.rsplit(".", 1)[-1], bbox_inches="tight")
        print(f"Saved: {output}")
        plt.close(fig)
