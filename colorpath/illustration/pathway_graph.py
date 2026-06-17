"""
pathway_graph.py — the pathway *activity graph* renderer.

This is the original colorpath illustration code (formerly ``pathway_viz.draw_pathway``),
relocated unchanged into the package so the decomposition engine can reuse it as the
visualisation layer. It draws a pathway as a directed graph with each metabolite node
coloured by a scalar value — for colorpath this scalar is a component's spectral loading
``V[k, :]`` (the pathway activity graph). The root-level ``pathway_viz.py`` re-exports
``draw_pathway`` from here for backward compatibility.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_pathway(
    pathway: list[tuple[str, str]],
    abundance: dict[str, float],
    colormap: str = "RdYlGn",
    output: str = "pathway.svg",
    title: str = "Pathway",
    node_size: int = 2000,
    font_size: int = 9,
    figsize: tuple[float, float] = (10, 7),
    layout: str = "dot",          # 'dot' (hierarchical) or 'spring' / 'kamada_kawai'
    positions: dict[str, tuple[float, float]] | None = None,
    colorbar_label: str = "Abundance",
    label_halo: bool = True,
) -> None:
    """
    Draw a pathway graph with nodes colored by abundance.

    Parameters
    ----------
    pathway   : list of (source, target) edge tuples
    abundance : {node_name: value} — nodes missing from this dict are drawn grey
    colormap  : matplotlib colormap name
    output    : output file path (.svg or .pdf recommended for vector output)
    title     : figure title
    node_size : size of each node circle
    font_size : label font size
    figsize   : (width, height) in inches
    layout    : graph layout algorithm
    positions : optional dict {node: (x, y)} — overrides automatic layout
    colorbar_label : label for the colour bar (e.g. a component's loading units)
    label_halo : draw a white outline around node labels so they stay legible on dark
                 (low-value) nodes; set False for the plain original style
    """
    G = nx.DiGraph()
    G.add_edges_from(pathway)

    # Add any abundance nodes not already in the graph as isolated nodes
    for node in abundance:
        if node not in G:
            G.add_node(node)

    # --- Layout ---
    if positions is not None:
        pos = positions
    elif layout == "dot":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    # --- Color mapping ---
    nodes = list(G.nodes())
    values = [abundance.get(n, np.nan) for n in nodes]

    known = [v for v in values if not np.isnan(v)]
    vmin = min(known) if known else 0
    vmax = max(known) if known else 1

    cmap = matplotlib.colormaps[colormap]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    node_colors = [
        cmap(norm(v)) if not np.isnan(v) else (0.7, 0.7, 0.7, 1.0)
        for v in values
    ]

    # --- Draw ---
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        arrowsize=20,
        edge_color="#555555",
        width=1.5,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        linewidths=1.2,
        edgecolors="#333333",
    )
    labels = nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=font_size,
        font_weight="bold",
    )
    if label_halo:
        # White outline keeps labels readable regardless of the node colour beneath.
        for text in labels.values():
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.5, foreground="white")]
            )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    fig.savefig(output, format=output.rsplit(".", 1)[-1], bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)
