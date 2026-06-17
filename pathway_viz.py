"""
pathway_viz.py — backward-compatible entry point for the pathway activity graph.

The renderer now lives in :mod:`colorpath.illustration.pathway_graph` (reused by the
decomposition engine). This module re-exports ``draw_pathway`` so existing imports and
``python pathway_viz.py`` keep working, and retains the dopaminergic-pathway example.

Usage:
    python pathway_viz.py

See ``demo_decomposition.py`` for the full decomposition -> illustration pipeline.
"""

from colorpath.illustration import draw_pathway

__all__ = ["draw_pathway"]


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
