"""
pathway_viz.py — Visualize a metabolic/signaling pathway with abundance-colored nodes.

Usage:
    python pathway_viz.py

Inputs (edit the EXAMPLE section at the bottom or import as a module):
    pathway : list of (source, target) tuples describing directed edges
    abundance: dict mapping node name -> numeric abundance value
    colormap : matplotlib colormap name (e.g. 'viridis', 'RdYlGn', 'plasma')

Output:
    A vector SVG (and/or PDF) figure saved to disk.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
    """
    G = nx.DiGraph()
    G.add_edges_from(pathway)

    # Add any abundance nodes not already in the graph as isolated nodes
    for node in abundance:
        if node not in G:
            G.add_node(node)

    # --- Layout ---
    if layout == "dot":
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

    cmap = cm.get_cmap(colormap)
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
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=font_size,
        font_weight="bold",
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Abundance", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    fig.savefig(output, format=output.rsplit(".", 1)[-1], bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example — edit this section to plug in your own data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Directed edges: (upstream, downstream)
    pathway = [
        ("Glucose", "G6P"),
        ("G6P", "F6P"),
        ("F6P", "F1,6BP"),
        ("F1,6BP", "DHAP"),
        ("F1,6BP", "G3P"),
        ("DHAP", "G3P"),
        ("G3P", "1,3BPG"),
        ("1,3BPG", "3PG"),
        ("3PG", "2PG"),
        ("2PG", "PEP"),
        ("PEP", "Pyruvate"),
    ]

    # Measured abundance values (e.g. log2 fold-change or raw intensities)
    abundance = {
        "Glucose":  1.2,
        "G6P":      2.5,
        "F6P":      1.8,
        "F1,6BP":   3.1,
        "DHAP":     0.5,
        "G3P":      2.0,
        "1,3BPG":  -0.3,
        "3PG":     -1.1,
        "2PG":     -0.8,
        "PEP":      0.2,
        "Pyruvate": 1.5,
    }

    # Any matplotlib colormap: 'RdYlGn', 'viridis', 'plasma', 'coolwarm', …
    colormap = "RdYlGn"

    draw_pathway(
        pathway=pathway,
        abundance=abundance,
        colormap=colormap,
        output="pathway.svg",
        title="Glycolysis — Substrate Abundance",
        layout="dot",          # requires pygraphviz; falls back to kamada_kawai
    )
