"""
scan_pathways.py — rank gene-set pathways by GAIT's regional-structure score.

Annotation-free, hypothesis-generating screen: for every pathway in a ``.gmt`` (e.g. a KEGG
collection), fit GAIT (KL-NMF, K components) on one Visium section and score how *regionally*
structured the pathway's activity is with
:func:`gait.benchmark.regional_structure_score` (= coherence x diversity, Moran's-I based).
Sort descending to see which pathways are most likely differentially regulated across the
tissue. **No permutation null** — the score ranks directly.

Symbol matching is case-insensitive, so a human KEGG ``.gmt`` (UPPER-case symbols) works on
the mouse sections (``Th`` etc.) for orthologues that differ only in case.

Usage:
    python scan_pathways.py GSE232910_RAW.tar c2.cp.kegg.v2023.2.symbols.gmt \
        [--section V11L12-038_A1] [--min-genes 10] [--K 5] [--top 16]

Writes  pathway_scan_<section>.csv  (all scored pathways, sorted) and
        pathway_scan_<section>_top.png  (dominant-component maps of the top hits).
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tarfile
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gait.benchmark import regional_structure_score
from gait.decomposition import LinearNMF
from gait.spatial import (
    dominant_component,
    library_normalize,
    load_visium_10x_h5,
    per_gene_scale,
    select_genes,
)
# reuse the GEO-tar helpers from the robustness script (same directory)
from robustness_multisection import _extract, _member, sample_prefixes


def read_gmt(path: str) -> dict[str, list[str]]:
    """Parse a GMT gene-set file (name <tab> description <tab> gene1 <tab> ...).

    Tolerates weighted tokens (``GENE,1.0`` -> ``GENE``), as some Enrichr/MSigDB exports use.
    """
    opener = gzip.open if path.endswith(".gz") else open
    sets: dict[str, list[str]] = {}
    with opener(path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sets[parts[0]] = [g.split(",")[0].strip() for g in parts[2:] if g.strip()]
    return sets


def load_section(tar, names, section):
    """Load one Visium section (counts + coords; no region annotation needed)."""
    prefixes = sample_prefixes(names)
    if section:
        hits = [p for p in prefixes if section in p]
        if not hits:
            sys.exit(f"no section matching {section!r}; available: "
                     f"{[p.strip('_') for p in prefixes]}")
        prefix = hits[0]
    else:
        prefix = prefixes[0]
    pos = _member(set(names), prefix, "tissue_positions_list.csv")
    with tempfile.TemporaryDirectory() as tmp:
        h5 = _extract(tar, prefix + "filtered_feature_bc_matrix.h5", tmp)
        posp = _extract(tar, pos, tmp)
        exp = load_visium_10x_h5(h5, posp)
    return prefix.strip("_").replace("_RNA", ""), exp


def main(tar_path, gmt_path, section, min_genes, K, top, max_iter):
    gmt = read_gmt(gmt_path)
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        label, exp = load_section(tar, names, section)
    print(f"[scan] section {label}: {exp.n_spots} spots; {len(gmt)} pathways in "
          f"{os.path.basename(gmt_path)}")

    coords = np.column_stack([exp.x, exp.y])
    Xn = library_normalize(exp.X)
    rows = []
    for i, (name, genes) in enumerate(gmt.items()):
        gidx, present = select_genes(exp.genes, symbols=genes, case_insensitive=True)
        if len(present) < min_genes:
            continue
        Xs = per_gene_scale(Xn[:, gidx])
        res = LinearNMF(K, loss="kl", max_iter=max_iter, random_state=0).fit(Xs)
        sc = regional_structure_score(res.U, res.V, coords)
        rows.append({"pathway": name, "n_genes": len(present), **sc,
                     "dom": dominant_component(res.U, res.V)})
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(gmt)} pathways")
    rows.sort(key=lambda r: -r["score"])
    print(f"[scan] scored {len(rows)} pathways (>= {min_genes} genes present)")

    # ranked CSV
    out_csv = f"pathway_scan_{label}.csv"
    with open(out_csv, "w") as fh:
        fh.write("rank,pathway,n_genes,score,coherence,diversity\n")
        for r, row in enumerate(rows, 1):
            fh.write(f"{r},{row['pathway']},{row['n_genes']},{row['score']:.4f},"
                     f"{row['coherence']:.4f},{row['diversity']:.4f}\n")

    print("\ntop pathways by regional-structure score:")
    for row in rows[:min(top, len(rows))]:
        print(f"  {row['score']:.3f}  (coh {row['coherence']:.2f} div {row['diversity']:.2f}"
              f" n={row['n_genes']:>3})  {row['pathway']}")

    # maps of the top hits
    n = min(top, len(rows))
    if n:
        ncol = min(4, n); nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for row, ax in zip(rows[:n], axes.ravel()):
            ax.axis("on")
            ax.scatter(exp.x, exp.y, c=row["dom"], s=3,
                       cmap=ListedColormap(plt.cm.tab10.colors[:K]), vmin=-0.5, vmax=K - 0.5)
            ax.set_title(f"{row['pathway'][:34]}\nscore={row['score']:.2f}", fontsize=7)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Top regionally-structured KEGG pathways — section {label} "
                     f"(GAIT KL-NMF, K={K})", fontweight="bold")
        out_png = f"pathway_scan_{label}_top.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\nwrote {out_csv} and {out_png}")


if __name__ == "__main__":
    default_tar = os.path.expanduser("~/Downloads/dopamine_mouse/GSE232910_RAW.tar")
    ap = argparse.ArgumentParser(description="Rank pathways by GAIT regional-structure score.")
    ap.add_argument("tar", nargs="?", default=default_tar, help="GSE..._RAW.tar bundle")
    ap.add_argument("gmt", help="gene-set .gmt (e.g. a KEGG collection; symbols)")
    ap.add_argument("--section", default=None,
                    help="substring of the section to scan (default: first; use a striatal "
                         "one e.g. V11L12-038_A1)")
    ap.add_argument("--min-genes", type=int, default=10, help="min pathway genes present")
    ap.add_argument("--K", type=int, default=5, help="components")
    ap.add_argument("--top", type=int, default=16, help="top hits to print/plot")
    ap.add_argument("--max-iter", type=int, default=300, help="NMF iterations per pathway")
    args = ap.parse_args()
    if not os.path.exists(args.tar):
        sys.exit(f"tar not found: {args.tar}")
    if not os.path.exists(args.gmt):
        sys.exit(f"gmt not found: {args.gmt}")
    main(args.tar, args.gmt, args.section, args.min_genes, args.K, args.top, args.max_iter)
