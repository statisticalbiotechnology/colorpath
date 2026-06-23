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


def _load_prefix(tar, names, prefix):
    """Load the Visium section at an exact tar prefix (counts + coords; no annotation)."""
    pos = _member(set(names), prefix, "tissue_positions_list.csv")
    with tempfile.TemporaryDirectory() as tmp:
        h5 = _extract(tar, prefix + "filtered_feature_bc_matrix.h5", tmp)
        posp = _extract(tar, pos, tmp)
        exp = load_visium_10x_h5(h5, posp)
    return prefix.strip("_").replace("_RNA", ""), exp


def load_section(tar, names, section):
    """Load one section by name substring (default: the first section in the tar)."""
    prefixes = sample_prefixes(names)
    if section:
        hits = [p for p in prefixes if section in p]
        if not hits:
            sys.exit(f"no section matching {section!r}; available: "
                     f"{[p.strip('_') for p in prefixes]}")
        prefix = hits[0]
    else:
        prefix = prefixes[0]
    return _load_prefix(tar, names, prefix)


def score_one_section(exp, gmt, min_genes, K, max_iter, keep_dom=False):
    """Return the pathways scored on one section, sorted by regional-structure score."""
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
        row = {"pathway": name, "n_genes": len(present), "score": sc["score"],
               "coherence": sc["coherence"], "diversity": sc["diversity"],
               "direction": sc["direction"]}
        if keep_dom:
            row["dom"] = dominant_component(res.U, res.V)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{len(gmt)} pathways")
    rows.sort(key=lambda r: -r["score"])
    return rows


def scan_single(tar, names, section, gmt, min_genes, K, top, max_iter):
    label, exp = load_section(tar, names, section)
    print(f"[scan] section {label}: {exp.n_spots} spots; {len(gmt)} pathways")
    rows = score_one_section(exp, gmt, min_genes, K, max_iter, keep_dom=True)
    print(f"[scan] scored {len(rows)} pathways (>= {min_genes} genes present)")

    out_csv = f"pathway_scan_{label}.csv"
    with open(out_csv, "w") as fh:
        fh.write("rank,pathway,n_genes,score,coherence,diversity,direction\n")
        for r, row in enumerate(rows, 1):
            fh.write(f"{r},{row['pathway']},{row['n_genes']},{row['score']:.4f},"
                     f"{row['coherence']:.4f},{row['diversity']:.4f},{row['direction']:.4f}\n")
    print("\ntop pathways by regional-structure score "
          "(dir = magnitude-invariant loading-direction distinctness):")
    for row in rows[:min(top, len(rows))]:
        print(f"  {row['score']:.3f}  (coh {row['coherence']:.2f} div {row['diversity']:.2f}"
              f" dir {row['direction']:.2f} n={row['n_genes']:>3})  {row['pathway']}")

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
        fig.suptitle(f"Top regionally-structured pathways — section {label} "
                     f"(GAIT KL-NMF, K={K})", fontweight="bold")
        out_png = f"pathway_scan_{label}_top.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\nwrote {out_csv} and {out_png}")


def scan_multi(tar, names, match, gmt, min_genes, K, top, max_iter):
    """Aggregate the scan over every section whose prefix contains ``match``."""
    sel = [p for p in sample_prefixes(names) if match in p]
    if not sel:
        sys.exit(f"no sections matching {match!r}")
    print(f"[scan] {len(sel)} sections matching {match!r}; {len(gmt)} pathways")
    scores: dict[str, list[float]] = {}
    ranks: dict[str, list[int]] = {}
    directions: dict[str, list[float]] = {}
    n_genes: dict[str, int] = {}
    for pre in sel:
        label, exp = _load_prefix(tar, names, pre)
        rows = score_one_section(exp, gmt, min_genes, K, max_iter)
        print(f"  {label}: scored {len(rows)} pathways")
        for rank, row in enumerate(rows, 1):
            scores.setdefault(row["pathway"], []).append(row["score"])
            ranks.setdefault(row["pathway"], []).append(rank)
            directions.setdefault(row["pathway"], []).append(row["direction"])
            n_genes[row["pathway"]] = row["n_genes"]
    agg = [{"pathway": p, "n_sections": len(s), "median_score": float(np.median(s)),
            "median_rank": float(np.median(ranks[p])),
            "median_direction": float(np.median(directions[p])), "n_genes": n_genes[p]}
           for p, s in scores.items()]
    agg.sort(key=lambda r: -r["median_score"])

    out_csv = f"pathway_scan_{match}_aggregate.csv"
    with open(out_csv, "w") as fh:
        fh.write("rank,pathway,n_sections,median_score,median_rank,median_direction,n_genes\n")
        for r, row in enumerate(agg, 1):
            fh.write(f"{r},{row['pathway']},{row['n_sections']},{row['median_score']:.4f},"
                     f"{row['median_rank']:.1f},{row['median_direction']:.4f},{row['n_genes']}\n")
    print(f"\ntop pathways by median regional-structure score across {len(sel)} sections "
          f"(med-dir = magnitude-invariant loading-direction distinctness):")
    for row in agg[:min(top, len(agg))]:
        print(f"  med {row['median_score']:.3f}  med-rank {row['median_rank']:>4.0f}  "
              f"med-dir {row['median_direction']:.2f}  "
              f"(n_sec {row['n_sections']}/{len(sel)}, genes {row['n_genes']:>3})  {row['pathway']}")
    print(f"\nwrote {out_csv}")


def main(tar_path, gmt_path, section, match, min_genes, K, top, max_iter):
    gmt = read_gmt(gmt_path)
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        if match:
            scan_multi(tar, names, match, gmt, min_genes, K, top, max_iter)
        else:
            scan_single(tar, names, section, gmt, min_genes, K, top, max_iter)


if __name__ == "__main__":
    default_tar = os.path.expanduser("~/Downloads/dopamine_mouse/GSE232910_RAW.tar")
    ap = argparse.ArgumentParser(description="Rank pathways by GAIT regional-structure score.")
    ap.add_argument("tar", nargs="?", default=default_tar, help="GSE..._RAW.tar bundle")
    ap.add_argument("gmt", help="gene-set .gmt (e.g. a KEGG collection; symbols)")
    ap.add_argument("--section", default=None,
                    help="substring of a single section to scan (default: first; e.g. "
                         "V11L12-038_A1)")
    ap.add_argument("--match", default=None,
                    help="scan ALL sections whose name contains this substring and aggregate "
                         "by median score/rank (e.g. V11L12 for the striatal slides)")
    ap.add_argument("--min-genes", type=int, default=10, help="min pathway genes present")
    ap.add_argument("--K", type=int, default=5, help="components")
    ap.add_argument("--top", type=int, default=16, help="top hits to print/plot")
    ap.add_argument("--max-iter", type=int, default=300, help="NMF iterations per pathway")
    args = ap.parse_args()
    if not os.path.exists(args.tar):
        sys.exit(f"tar not found: {args.tar}")
    if not os.path.exists(args.gmt):
        sys.exit(f"gmt not found: {args.gmt}")
    main(args.tar, args.gmt, args.section, args.match, args.min_genes, args.K, args.top,
         args.max_iter)
