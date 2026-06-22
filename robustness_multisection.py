"""
robustness_multisection.py — does GAIT recover region-specific pathway components
*reproducibly across many sections*?

Runs the dominant-component analysis of ``demo_visium_dopamine.py`` over every Visium
section in a GEO ``_RAW.tar`` bundle (e.g. GSE232910, 19 mouse-brain sections), for a chosen
brain pathway, and quantifies how well the resulting components align with each section's
region annotation (mutual information between the per-spot dominant component and the region
label). A consistent, well-above-zero MI across sections is the robustness evidence.

Everything stays in linear space (a linear ``library_normalize``, no log) with KL-NMF, as in
the rest of GAIT.

Usage:
    python robustness_multisection.py [GSE232910_RAW.tar] [pathway] [K]

    pathway is one of:  dopaminergic serotonergic noradrenergic gabaergic
                        glutamatergic cholinergic myelination
    (default tar: ~/Downloads/dopamine_mouse/GSE232910_RAW.tar; pathway: dopaminergic; K=5)

Writes  robustness_<pathway>_maps.png  (a grid of per-section dominant-component maps) and
        robustness_<pathway>.csv       (per-section summary), and prints a table.
"""

from __future__ import annotations

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

from gait.decomposition import LinearNMF
from gait.benchmark import region_mutual_information
from gait.spatial import (
    dominant_component,
    library_normalize,
    load_visium_10x_h5,
    per_gene_scale,
    select_genes,
)

# Curated mouse-brain pathway gene sets (case-insensitive, so human caps match too).
PATHWAYS: dict[str, list[str]] = {
    "dopaminergic":  ["Drd1", "Drd2", "Adora2a", "Ppp1r1b", "Pde10a", "Gpr88", "Penk",
                      "Pdyn", "Rgs9", "Gnal", "Adcy5", "Tac1"],
    "serotonergic":  ["Tph2", "Slc6a4", "Htr1a", "Htr1b", "Htr2a", "Htr2c", "Fev",
                      "Gch1", "Ddc", "Maoa", "Slc18a2"],
    "noradrenergic": ["Dbh", "Slc6a2", "Pnmt", "Th", "Ddc", "Slc18a2", "Adra1a",
                      "Adra2a", "Adrb1"],
    "gabaergic":     ["Gad1", "Gad2", "Slc32a1", "Gabra1", "Gabrb2", "Gabbr1", "Pvalb",
                      "Sst", "Vip"],
    "glutamatergic": ["Slc17a7", "Slc17a6", "Grin1", "Grin2a", "Grin2b", "Gria1", "Gria2",
                      "Grm5", "Camk2a", "Satb2"],
    "cholinergic":   ["Chat", "Slc5a7", "Slc18a3", "Ache", "Chrm1", "Chrm2", "Chrm3",
                      "Chrna4", "Chrnb2"],
    "myelination":   ["Mbp", "Plp1", "Mobp", "Mog", "Mag", "Cnp", "Sox10", "Cldn11"],
}


def _extract(tar: tarfile.TarFile, name: str, dest: str) -> str:
    """Extract a tar member to ``dest``, gunzipping if needed; return the local path."""
    data = tar.extractfile(name).read()
    base = os.path.basename(name)
    if name.endswith(".gz"):
        data = gzip.decompress(data)
        base = base[:-3]
    out = os.path.join(dest, base)
    with open(out, "wb") as fh:
        fh.write(data)
    return out


def _member(names: set[str], prefix: str, suffix: str) -> str | None:
    """Find ``prefix+suffix`` in the tar, with or without a trailing ``.gz``."""
    for cand in (prefix + suffix, prefix + suffix + ".gz"):
        if cand in names:
            return cand
    return None


def sample_prefixes(names: list[str]) -> list[str]:
    """Every distinct sample prefix (the part before filtered_feature_bc_matrix.h5)."""
    tag = "filtered_feature_bc_matrix.h5"
    return sorted(n[: -len(tag)] for n in names if n.endswith(tag))


def process_sample(tar, names, prefix, wanted, K):
    """Fit GAIT on one section; return a summary dict (or None if unusable)."""
    pos = _member(names, prefix, "tissue_positions_list.csv")
    if pos is None:
        return None
    h5_name = prefix + "filtered_feature_bc_matrix.h5"
    reg_name = _member(names, prefix, "region.csv")
    label = prefix.strip("_").split("_", 1)[-1].replace("_RNA", "")

    with tempfile.TemporaryDirectory() as tmp:
        h5 = _extract(tar, h5_name, tmp)
        posp = _extract(tar, pos, tmp)
        regp = _extract(tar, reg_name, tmp) if reg_name else None
        exp = load_visium_10x_h5(h5, posp, regp)

    gidx, present = select_genes(exp.genes, symbols=wanted, case_insensitive=True)
    if len(present) < 3:
        return {"sample": label, "n_spots": exp.n_spots, "n_genes": len(present),
                "mi": np.nan, "x": None}
    Xs = per_gene_scale(library_normalize(exp.X)[:, gidx])
    res = LinearNMF(K, loss="kl", max_iter=500, random_state=0).fit(Xs)
    dom = dominant_component(res.U, res.V)
    mi = (region_mutual_information(dom, exp.region)
          if exp.region is not None else np.nan)
    regions = sorted(set(exp.region)) if exp.region is not None else []
    return {"sample": label, "n_spots": exp.n_spots, "n_genes": len(present),
            "mi": mi, "regions": regions, "x": exp.x, "y": exp.y, "dom": dom, "K": res.K}


def main(tar_path: str, pathway: str, K: int) -> None:
    wanted = PATHWAYS[pathway]
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        nameset = set(names)
        prefixes = sample_prefixes(names)
        print(f"[{pathway}] {len(prefixes)} sections in {os.path.basename(tar_path)}")
        rows = []
        for pre in prefixes:
            try:
                r = process_sample(tar, nameset, pre, wanted, K)
            except Exception as e:                       # keep going on a bad section
                print(f"  ! {pre}: {type(e).__name__}: {e}")
                continue
            if r:
                rows.append(r)

    usable = [r for r in rows if r.get("x") is not None]
    print(f"\n{'section':<22}{'spots':>7}{'genes':>6}{'region MI (bits)':>18}")
    for r in rows:
        mi = f"{r['mi']:.3f}" if r["mi"] == r["mi"] else "  n/a"
        print(f"{r['sample']:<22}{r['n_spots']:>7}{r['n_genes']:>6}{mi:>18}")
    mis = [r["mi"] for r in usable if r["mi"] == r["mi"]]
    if mis:
        print(f"\nregion MI across sections: median={np.median(mis):.3f} "
              f"min={np.min(mis):.3f} max={np.max(mis):.3f}  (n={len(mis)})")

    # CSV
    with open(f"robustness_{pathway}.csv", "w") as fh:
        fh.write("sample,n_spots,n_genes,region_mi\n")
        for r in rows:
            fh.write(f"{r['sample']},{r['n_spots']},{r['n_genes']},{r['mi']}\n")

    # Figure: grid of per-section dominant-component maps.
    if usable:
        n = len(usable)
        ncol = min(5, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow),
                                 squeeze=False)
        cmap = ListedColormap(plt.cm.tab10.colors[:max(r["K"] for r in usable)])
        for ax in axes.ravel():
            ax.axis("off")
        for r, ax in zip(usable, axes.ravel()):
            ax.axis("on")
            ax.scatter(r["x"], r["y"], c=r["dom"], s=3, cmap=cmap,
                       vmin=-0.5, vmax=r["K"] - 0.5)
            mi = f"{r['mi']:.2f}" if r["mi"] == r["mi"] else "n/a"
            ax.set_title(f"{r['sample']}\nMI={mi}", fontsize=8)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"GAIT dominant-component map — {pathway} pathway, KL-NMF (K={K}) "
                     f"across {n} sections", fontweight="bold")
        out = f"robustness_{pathway}_maps.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out} and robustness_{pathway}.csv")


if __name__ == "__main__":
    default_tar = os.path.expanduser("~/Downloads/dopamine_mouse/GSE232910_RAW.tar")
    tar_path = sys.argv[1] if len(sys.argv) > 1 else default_tar
    pathway = sys.argv[2] if len(sys.argv) > 2 else "dopaminergic"
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    if pathway not in PATHWAYS:
        sys.exit(f"unknown pathway {pathway!r}; choose from {list(PATHWAYS)}")
    if not os.path.exists(tar_path):
        print(__doc__)
        sys.exit(f"error: tar not found at {tar_path} (pass it as the first argument)")
    main(tar_path, pathway, K)
