"""
pathways.py — reusable pathway topologies and hand-tuned node layouts.

The decomposition produces a loading over metabolite *names*; to draw it as a pathway
activity graph we need (a) the reaction edges and (b) a readable layout. Automatic
layouts (spring / kamada-kawai) overlap long metabolite labels, and the hierarchical
``dot`` layout needs Graphviz installed. We therefore ship hand-tuned positions that
spread the nodes by their role in the pathway, mirroring the style of the original
dopamine example in ``pathway_viz.py``.

Currently provided: the **catecholamine + serotonin** pathway whose metabolites match
the columns of the example MALDI-MSI datasets (``Dopamine``, ``L-DOPA``, ``HVA``,
``Serotonin`` …). Positions are laid out as synthesis rows (top) flowing down into
catabolism, with the serotonin branch as a separate column on the right.
"""

from __future__ import annotations

# Reaction edges (source -> product) over the dataset's metabolite names.
CATECHOLAMINE_SEROTONIN_EDGES: list[tuple[str, str]] = [
    # --- catecholamine synthesis ---
    ("L-DOPA", "Dopamine"),
    ("Dopamine", "Norepinephrine"),
    ("Norepinephrine", "Epinephrine"),
    # --- COMT O-methylation of the precursor ---
    ("L-DOPA", "3-OMD"),
    ("3-OMD", "Vanilpyruvic_acid"),
    ("Vanilpyruvic_acid", "Vanillactic_acid"),
    # --- dopamine catabolism ---
    ("Dopamine", "3-MT"),
    ("Dopamine", "DOPAL"),
    ("3-MT", "MOPAL__Homovanillin"),
    ("DOPAL", "DOPAC"),
    ("DOPAL", "DOPET"),
    ("MOPAL__Homovanillin", "HVA"),
    ("DOPAC", "HVA"),
    ("DOPET", "MOPET"),
    # --- norepinephrine / epinephrine catabolism ---
    ("Norepinephrine", "DOPEGAL"),
    ("Norepinephrine", "Normetanephrine"),
    ("Epinephrine", "Metanephrine"),
    ("DOPEGAL", "DOPEG"),
    ("DOPEG", "MOPEG"),
    ("Normetanephrine", "DOMA"),
    ("Metanephrine", "MOPEGAL"),
    # --- convergence to VMA ---
    ("DOMA", "VMA"),
    ("MOPEGAL", "VMA"),
    ("DOPEG", "VMA"),
    ("MOPEG", "VMA"),
    ("MOPET", "VMA"),
    # --- serotonin branch ---
    ("5-HTP", "Serotonin"),
    ("Serotonin", "5-HIAL"),
    ("5-HIAL", "5-HIAA"),
    ("5-HIAL", "5-HTOL"),
]

# Hand-tuned (x, y) positions: higher y = upstream. Columns are spread generously so the
# long metabolite labels do not overlap. The serotonin branch sits at the far right.
CATECHOLAMINE_SEROTONIN_POSITIONS: dict[str, tuple[float, float]] = {
    # row y=10 — precursor + COMT methylation branch
    "L-DOPA": (0, 10),
    "3-OMD": (4, 10),
    "Vanilpyruvic_acid": (8, 10),
    "Vanillactic_acid": (12, 10),
    # row y=8 — core synthesis
    "Dopamine": (0, 8),
    "Norepinephrine": (5, 8),
    "Epinephrine": (10, 8),
    # row y=6 — first catabolism step
    "3-MT": (-4, 6),
    "DOPAL": (1, 6),
    "DOPEGAL": (5, 6),
    "Normetanephrine": (8.5, 6),
    "Metanephrine": (13, 6),
    # row y=4 — second catabolism step
    "MOPAL__Homovanillin": (-4, 4),
    "DOPAC": (0.5, 4),
    "DOPET": (3, 4),
    "DOPEG": (5.5, 4),
    "DOMA": (9, 4),
    "MOPEGAL": (13, 4),
    # row y=2 — terminal metabolites
    "HVA": (-2.5, 2),
    "MOPET": (3, 2),
    "MOPEG": (6, 2),
    "VMA": (9.5, 2),
    # serotonin branch — far-right column
    "5-HTP": (18, 10),
    "Serotonin": (18, 8),
    "5-HIAL": (18, 6),
    "5-HIAA": (16, 4),
    "5-HTOL": (20, 4),
}
