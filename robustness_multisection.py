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

    pathway is one of:  neurotransmission dopaminergic da_synthesis msn neuropeptide
                        serotonergic noradrenergic gabaergic glutamatergic cholinergic
                        myelination
    (neurotransmission = the broad manuscript set that splits into striatal/cortical/...
     components; it and dopaminergic are the ones whose axis matches the striatum
     annotation, so report their striatum AUC as the robustness metric)
    (msn = striatal medium-spiny-neuron direct/D1 vs indirect/D2 programmes; the
     dopaminergic/msn/neuropeptide/serotonergic/noradrenergic/gabaergic sets each have a
     metabolite measured by FMP-10, enabling a metabolite<->transcript comparison)
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

from scipy.stats import mannwhitneyu

from gait.decomposition import LinearNMF
from gait.benchmark import region_mutual_information
from gait.spatial import (
    NEUROTRANSMISSION_GENES,
    dominant_component,
    library_normalize,
    load_visium_10x_h5,
    per_gene_scale,
    select_genes,
)

# Curated mouse-brain pathway gene sets (case-insensitive, so human caps match too).
# Pathways marked (MSI) below have a metabolite measured by FMP-10 in the SMA dataset
# (dopamine, norepinephrine, serotonin, GABA, and the Penk/dynorphin neuropeptides), so they
# support the same metabolite<->transcript concordance test as dopamine.
PATHWAYS: dict[str, list[str]] = {
    # broad multi-system set used for the manuscript figure (splits into striatal /
    # cortical / GABAergic dominating components -- the right kind of set for region MI):
    "neurotransmission": NEUROTRANSMISSION_GENES,
    "dopaminergic":  ["Drd1", "Drd2", "Adora2a", "Ppp1r1b", "Pde10a", "Gpr88", "Penk",
                      "Pdyn", "Rgs9", "Gnal", "Adcy5", "Tac1"],                  # (MSI: dopamine)
    "da_synthesis":  ["Th", "Slc6a3", "Slc18a2", "Ddc", "Nr4a2", "Pitx3"],      # SNc cell bodies
    "msn":           ["Drd1", "Tac1", "Pdyn", "Pcp4",                           # direct/D1
                      "Drd2", "Penk", "Adora2a", "Cartpt",                      # indirect/D2
                      "Gpr88", "Ppp1r1b", "Rgs9", "Pde10a"],   # striatal MSNs (D1 vs D2 split)
    "neuropeptide":  ["Penk", "Pdyn", "Tac1", "Cartpt", "Pcp4", "Scg2"],        # (MSI: Penk, dynorphin)
    "serotonergic":  ["Tph2", "Slc6a4", "Htr1a", "Htr1b", "Htr2a", "Htr2c", "Fev",
                      "Gch1", "Ddc", "Maoa", "Slc18a2"],                        # (MSI: serotonin)
    "noradrenergic": ["Dbh", "Slc6a2", "Pnmt", "Th", "Ddc", "Slc18a2", "Adra1a",
                      "Adra2a", "Adrb1"],                                       # (MSI: norepinephrine)
    "gabaergic":     ["Gad1", "Gad2", "Slc32a1", "Gabra1", "Gabrb2", "Gabbr1", "Pvalb",
                      "Sst", "Vip"],                                            # (MSI: GABA)
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


def region_auc(U: np.ndarray, region) -> tuple[float, float, object]:
    """Best AUC over the K component activities for separating one region label from rest.

    A far more sensitive region-recovery test than the dominant-component MI: it asks
    whether *any* component's activity separates a region, and at what AUC. Works for any
    annotation — binary (striatum/not, intact/lesioned) uses the natural split; >2 labels
    take the most separable one-vs-rest. Returns (auc, MWU p, label); (nan, nan, None) when
    no usable annotation is present.
    """
    if region is None:
        return np.nan, np.nan, None
    region = np.asarray(region)
    labels = [v for v in np.unique(region) if v != ""]
    if len(labels) < 2:
        return np.nan, np.nan, None
    candidates = labels if len(labels) > 2 else labels[:1]   # binary: one split suffices
    best = (0.0, np.nan, None)
    for lab in candidates:
        y = region == lab
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        denom = y.sum() * (~y).sum()
        for k in range(U.shape[1]):
            r = mannwhitneyu(U[y, k], U[~y, k], alternative="two-sided")
            auc = max(r.statistic / denom, 1 - r.statistic / denom)   # direction-agnostic
            if auc > best[0]:
                best = (auc, float(r.pvalue), lab)
    return float(best[0]), best[1], best[2]


def process_sample(tar, names, prefix, wanted, K, region_file="region.csv"):
    """Fit GAIT on one section; return a summary dict (or None if unusable).

    ``region_file`` is the annotation scored against ('region.csv' = striatum/not_striatum,
    'lesion.csv' = intact/lesioned)."""
    pos = _member(names, prefix, "tissue_positions_list.csv")
    if pos is None:
        return None
    h5_name = prefix + "filtered_feature_bc_matrix.h5"
    reg_name = _member(names, prefix, region_file)
    label = prefix.strip("_").split("_", 1)[-1].replace("_RNA", "")

    with tempfile.TemporaryDirectory() as tmp:
        h5 = _extract(tar, h5_name, tmp)
        posp = _extract(tar, pos, tmp)
        regp = _extract(tar, reg_name, tmp) if reg_name else None
        exp = load_visium_10x_h5(h5, posp, regp)

    gidx, present = select_genes(exp.genes, symbols=wanted, case_insensitive=True)
    if len(present) < 3:
        return {"sample": label, "n_spots": exp.n_spots, "n_genes": len(present),
                "mi": np.nan, "auc": np.nan, "auc_p": np.nan, "auc_label": None, "x": None}
    Xs = per_gene_scale(library_normalize(exp.X)[:, gidx])
    res = LinearNMF(K, loss="kl", max_iter=500, random_state=0).fit(Xs)
    dom = dominant_component(res.U, res.V)
    mi = (region_mutual_information(dom, exp.region)
          if exp.region is not None else np.nan)
    auc, auc_p, auc_label = region_auc(res.U, exp.region)
    regions = sorted(set(exp.region)) if exp.region is not None else []
    return {"sample": label, "n_spots": exp.n_spots, "n_genes": len(present),
            "mi": mi, "auc": auc, "auc_p": auc_p, "auc_label": auc_label, "regions": regions,
            "x": exp.x, "y": exp.y, "dom": dom, "K": res.K}


def main(tar_path: str, pathway: str, K: int, region: str = "region") -> None:
    wanted = PATHWAYS[pathway]
    region_file = f"{region}.csv"        # 'region' -> striatum/not; 'lesion' -> intact/lesioned
    tag = f"{pathway}_{region}"
    auc_hdr = "intact/les AUC" if region == "lesion" else "striatum AUC"
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        nameset = set(names)
        prefixes = sample_prefixes(names)
        print(f"[{pathway} | annotation: {region_file}] {len(prefixes)} sections "
              f"in {os.path.basename(tar_path)}")
        rows = []
        for pre in prefixes:
            try:
                r = process_sample(tar, nameset, pre, wanted, K, region_file)
            except Exception as e:                       # keep going on a bad section
                print(f"  ! {pre}: {type(e).__name__}: {e}")
                continue
            if r:
                rows.append(r)

    usable = [r for r in rows if r.get("x") is not None]
    print(f"\n{'section':<22}{'spots':>7}{'genes':>6}{'MI':>9}{auc_hdr:>16}")
    for r in rows:
        mi = f"{r['mi']:.3f}" if r["mi"] == r["mi"] else "n/a"
        auc = f"{r['auc']:.3f}" if r.get("auc") == r.get("auc") else "n/a"
        print(f"{r['sample']:<22}{r['n_spots']:>7}{r['n_genes']:>6}{mi:>9}{auc:>16}")
    mis = [r["mi"] for r in usable if r["mi"] == r["mi"]]
    aucs = [r["auc"] for r in usable if r.get("auc") == r.get("auc")]
    if mis:
        print(f"\nMI  : median={np.median(mis):.3f} "
              f"[{np.min(mis):.3f}, {np.max(mis):.3f}]  (n={len(mis)})")
    if aucs:
        print(f"AUC : median={np.median(aucs):.3f} "
              f"[{np.min(aucs):.3f}, {np.max(aucs):.3f}]  (n={len(aucs)})")

    # CSV
    with open(f"robustness_{tag}.csv", "w") as fh:
        fh.write("sample,n_spots,n_genes,mi,auc,auc_p,auc_label\n")
        for r in rows:
            fh.write(f"{r['sample']},{r['n_spots']},{r['n_genes']},{r['mi']},"
                     f"{r.get('auc', float('nan'))},{r.get('auc_p', float('nan'))},"
                     f"{r.get('auc_label')}\n")

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
            auc = f"{r['auc']:.2f}" if r.get("auc") == r.get("auc") else "n/a"
            ax.set_title(f"{r['sample']}\nAUC={auc}", fontsize=8)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"GAIT dominant-component map — {pathway} pathway vs {region}, "
                     f"KL-NMF (K={K}) across {n} sections", fontweight="bold")
        out = f"robustness_{tag}_maps.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out} and robustness_{tag}.csv")


if __name__ == "__main__":
    import argparse

    default_tar = os.path.expanduser("~/Downloads/dopamine_mouse/GSE232910_RAW.tar")
    ap = argparse.ArgumentParser(description="GAIT multi-section robustness over a GEO tar.")
    ap.add_argument("tar", nargs="?", default=default_tar, help="GSE..._RAW.tar bundle")
    ap.add_argument("pathway", nargs="?", default="dopaminergic",
                    help=f"one of: {' '.join(PATHWAYS)}")
    ap.add_argument("--region", choices=["region", "lesion"], default="region",
                    help="annotation to score against: region=striatum/not, lesion=intact/lesioned")
    ap.add_argument("--K", type=int, default=5, help="number of components")
    args = ap.parse_args()
    if args.pathway not in PATHWAYS:
        sys.exit(f"unknown pathway {args.pathway!r}; choose from {list(PATHWAYS)}")
    tar_path = args.tar
    if not os.path.exists(tar_path):
        print(__doc__)
        sys.exit(f"error: tar not found at {tar_path} (pass it as the first argument)")
    main(tar_path, args.pathway, args.K, args.region)
