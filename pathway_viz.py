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
    positions: dict[str, tuple[float, float]] | None = None,
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
    # Directed edges: dopaminergic pathway (catecholamine synthesis & metabolism)
    pathway = [
        # Synthesis branch
        ("DOPA", "DA"),         # AADC
        ("DA", "NE"),           # DBH
        ("NE", "EP"),           # PNMT
        # COMT O-methylation of precursor
        ("DOPA", "3-OMD"),      # COMT
        ("3-OMD", "VP"),
        ("VP", "VLA"),
        # DA catabolism
        ("DA", "3-MT"),         # COMT
        ("DA", "DOPAL"),        # MAO
        ("3-MT", "MOPAL"),      # MAO
        ("DOPAL", "DOPAC"),     # AD
        ("DOPAL", "DOPET"),     # AR
        ("MOPAL", "HVA"),       # AD
        ("DOPAC", "HVA"),       # COMT
        ("DOPET", "MOPET"),     # COMT
        # NE / EP catabolism
        ("NE", "DOPEGAL"),      # MAO
        ("NE", "NMN"),          # COMT
        ("EP", "MN"),           # COMT
        ("DOPEGAL", "DOPEG"),   # AR
        ("DOPEG", "MOPEG"),     # COMT
        ("NMN", "DOMA"),        # MAO
        ("MN", "MOPEGAL"),      # MAO
        # Convergence to VMA
        ("DOMA", "VMA"),        # COMT
        ("MOPEGAL", "VMA"),     # AD
        ("DOPEG", "VMA"),       # AD  (liver, adrenal gland)
        ("MOPEG", "VMA"),       # AD
        ("MOPET", "VMA"),       # ADH (liver, adrenal gland)
    ]

    # Measured abundance values (e.g. log2 fold-change or raw intensities)
    abundance = {
        "DOPA":     2.0,
        "3-OMD":    0.8,
        "VP":       0.3,
        "VLA":      0.1,
        "DA":       3.5,
        "NE":       2.8,
        "EP":       1.5,
        "3-MT":     1.2,
        "DOPAL":   -0.5,
        "MOPAL":   -0.2,
        "DOPAC":    0.9,
        "DOPET":    0.4,
        "HVA":      2.2,
        "MOPET":    0.3,
        "DOPEGAL": -0.8,
        "NMN":      1.0,
        "MN":       0.6,
        "DOPEG":   -0.4,
        "MOPEG":    0.5,
        "DOMA":     0.7,
        "MOPEGAL": -0.1,
        "VMA":      1.8,
    }

    # Any matplotlib colormap: 'RdYlGn', 'viridis', 'plasma', 'coolwarm', …
    colormap = "RdYlGn"

    # Hand-tuned positions mirroring the reference figure layout
    # x = left→right, y = top→bottom (y is inverted so higher = top)
    positions = {
        # Row 0 — precursor + COMT branch
        "DOPA":     (0,  6),
        "3-OMD":    (3,  6),
        "VP":       (5,  6),
        "VLA":      (7,  6),
        # Row 1 — core synthesis
        "DA":       (0,  4.5),
        "NE":       (4,  4.5),
        "EP":       (7,  4.5),
        # Row 2 — first catabolism step
        "3-MT":     (-2, 3),
        "DOPAL":    (1,  3),
        "DOPEGAL":  (4,  3),
        "NMN":      (6,  3),
        "MN":       (8,  3),
        # Row 3 — second catabolism step
        "MOPAL":    (-2, 1.5),
        "DOPAC":    (0,  1.5),
        "DOPET":    (2,  1.5),
        "DOPEG":    (4,  1.5),
        "DOMA":     (6,  1.5),
        "MOPEGAL":  (8,  1.5),
        # Row 4 — terminal metabolites
        "HVA":      (-1, 0),
        "MOPET":    (2,  0),
        "MOPEG":    (4,  0),
        "VMA":      (6,  0),
    }

    draw_pathway(
        pathway=pathway,
        abundance=abundance,
        colormap=colormap,
        output="pathway.svg",
        title="Dopaminergic Pathway — Metabolite Abundance",
        positions=positions,
        figsize=(14, 10),
    )
