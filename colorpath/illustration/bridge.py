"""
bridge.py — connect the decomposition engine to the illustration layer.

A fitted component is the pair ``(U[:, k], V[k, :])``:

    V[k, :]  spectral loading  -> :func:`pathway_graph.draw_pathway` (pathway activity graph)
    U[:, k]  spatial score     -> :func:`pathway_image.render_pathway_activity_image`
                                  (pathway activity image)

:func:`illustrate_component` maps one component to both renderers in one call, handling
the bookkeeping (column-name -> loading dict, score column -> grid). It accepts either a
:class:`~colorpath.decomposition.nmf_linear.LinearNMFResult` (Route 2) or a
:class:`~colorpath.decomposition.nmf_loglevel.LogLevelNMFResult` (Route 1), or raw
``U``/``V`` arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..decomposition.contributions import variation_explained
from .pathway_graph import draw_pathway
from .pathway_image import render_pathway_activity_image


def _components_from(result):
    """Extract ``(U, V, default_units)`` from a result object or pass arrays through."""
    # Route 2: U (P,K) / V (K,M) on linear-abundance units.
    if hasattr(result, "U") and hasattr(result, "V"):
        return result.U, result.V, "loading (linear)"
    # Route 1: g (P,K) activity images / p (K,M) membership loadings (asinh scale).
    if hasattr(result, "g") and hasattr(result, "p"):
        return result.g, result.p, "membership (asinh)"
    raise TypeError(
        "result must be a LinearNMFResult, LogLevelNMFResult, or have U/V attributes"
    )


def illustrate_component(
    result,
    component: int,
    metabolite_names: Sequence[str],
    pathway_edges: list[tuple[str, str]],
    image_shape: tuple[int, int],
    *,
    U: np.ndarray | None = None,
    V: np.ndarray | None = None,
    graph_output: str | None = None,
    image_output: str | None = None,
    positions: dict[str, tuple[float, float]] | None = None,
    pixel_index: np.ndarray | None = None,
    graph_colormap: str = "RdYlGn",
    image_colormap: str = "viridis",
    title_prefix: str = "Pathway component",
    graph_value: str = "loading",
    graph_kwargs: dict | None = None,
    image_kwargs: dict | None = None,
) -> dict[str, str]:
    """Render both views of one decomposition component.

    Parameters
    ----------
    result           : a fitted result object (Route 2 or Route 1). Pass ``None`` and
                       supply ``U``/``V`` directly to illustrate raw factor matrices.
    component        : component index ``k``.
    metabolite_names : length-M column labels for ``V``; these key the loading dict and
                       must match the node names used in ``pathway_edges``.
    pathway_edges    : (source, target) edges defining the metabolite network to draw.
    image_shape      : (height, width) acquisition grid for the spatial score.
    U, V             : raw factor matrices, used when ``result`` is None.
    graph_output     : path for the pathway activity graph (default per component).
    image_output     : path for the pathway activity image (default per component).
    positions        : optional fixed node positions for the graph.
    pixel_index      : optional flat pixel indices for an irregular grid.
    graph_colormap / image_colormap : colormaps for the two views.
    title_prefix     : prefix for both figure titles.
    graph_value      : what the graph nodes encode. ``"loading"`` (default) colours by the
                       raw loading ``V[k, :]`` (linear-abundance units); ``"explained"``
                       colours by the per-metabolite fraction of variation explained by the
                       component (:func:`variation_explained`), in [0, 1] — this removes the
                       concentration imbalance that makes the raw loading look near-binary.
    graph_kwargs     : extra keyword arguments forwarded to
                       :func:`pathway_graph.draw_pathway` (e.g. ``figsize``,
                       ``node_size``, ``font_size``, ``layout``).
    image_kwargs     : extra keyword arguments forwarded to
                       :func:`pathway_image.render_pathway_activity_image`
                       (e.g. ``figsize``, ``background``).

    Returns
    -------
    dict with the written ``"graph"`` and ``"image"`` file paths.
    """
    if result is not None:
        Umat, Vmat, units = _components_from(result)
    else:
        if U is None or V is None:
            raise ValueError("supply either `result` or both `U` and `V`")
        Umat, Vmat, units = np.asarray(U), np.asarray(V), "loading"

    K, M = Vmat.shape
    if not (0 <= component < K):
        raise IndexError(f"component {component} out of range for K={K}")
    if len(metabolite_names) != M:
        raise ValueError(
            f"metabolite_names has {len(metabolite_names)} entries but V has {M} columns"
        )

    scores = Umat[:, component]

    if graph_value == "explained":
        F = variation_explained(Umat, Vmat)          # (K, M) in [0, 1]
        node_values = F[component, :]
        graph_units = "fraction of variation explained"
        vlim = dict(vmin=0.0, vmax=1.0)
    elif graph_value == "loading":
        node_values = Vmat[component, :]
        graph_units = units
        vlim = {}
    else:
        raise ValueError("graph_value must be 'loading' or 'explained'")

    abundance = {name: float(node_values[j]) for j, name in enumerate(metabolite_names)}

    graph_output = graph_output or f"component_{component}_graph.svg"
    image_output = image_output or f"component_{component}_image.svg"

    draw_pathway(
        pathway=pathway_edges,
        abundance=abundance,
        colormap=graph_colormap,
        output=graph_output,
        title=f"{title_prefix} {component} — activity graph",
        positions=positions,
        colorbar_label=graph_units,
        **vlim,
        **(graph_kwargs or {}),
    )
    render_pathway_activity_image(
        scores=scores,
        image_shape=image_shape,
        output=image_output,
        title=f"{title_prefix} {component} — activity image",
        colormap=image_colormap,
        colorbar_label=units,
        pixel_index=pixel_index,
        **(image_kwargs or {}),
    )

    return {"graph": graph_output, "image": image_output}
