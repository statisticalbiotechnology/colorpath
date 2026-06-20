"""
spatial.py — apply the colorpath decomposition engine to spatial transcriptomics.

colorpath was written for imaging mass spectrometry (pixels x ions), but the maths is
identical for **spatial transcriptomics** (Visium / Visium HD): a non-negative matrix

    X  in  R>=0^{P x M}        P spots (rows), M genes (columns)
    X ~= U V                   U spatial scores (one map per component)
                               V gene loadings  (one gene vector per component)

so a rank-1 component is a *pathway/program activity* exactly as for IMS, and feeds the
same illustration layer (:func:`colorpath.illustration.illustrate_component`):

    V[k, :]  -> pathway activity graph  (which genes load, coloured by loading share)
    U[:, k]  -> pathway activity image  (where the program is active over the tissue)

This module only adds **I/O and bookkeeping** around that engine — it does not
reimplement the factorisation or the renderers. It loads a directory exported as

    gene_names_*.txt          one HGNC symbol per line  (M genes)
    gene_counts_*.csr.npz      scipy sparse matrix (P spots x M genes); .mtx also accepted
    spot_barcodes.txt          one barcode per line, row order of the matrix (P spots)
    spatial_coordinates.csv    header with columns barcode, x, y, ... , sampleID

and provides helpers to: build the ``(image_shape, pixel_index)`` a single Visium section
needs for the activity-image renderer; select a gene set (by symbol and/or Ensembl-style
prefix); equalise per-gene influence before factorising (so a few hyper-abundant genes do
not each capture their own component); and derive a data-driven co-expression network for
the activity graph.

Notes
-----
* Many exports are already log1p-normalised (non-integer values, max ~ 8-10). Such data is
  non-negative and works directly with the KL loss as a *normalised-abundance* fit; if you
  hold raw counts, those are the cleaner linear-space (Route 2) input. Either way, prefer
  :func:`per_gene_scale` for membership-style analyses so abundance does not dominate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# A compact, curated plasma-cell / B-lineage program (human HGNC symbols). The secretory
# machinery (MZB1, DERL3, XBP1, SSR4, SEL1L, HERPUD1) and plasma transcription / receptor
# markers (PRDM1, POU2AF1, TNFRSF17, SLAMF7, SDC1, CD38) sit alongside the B-cell markers
# (MS4A1, CD79A/B, CD27); combine with the immunoglobulin loci via ``ig_prefixes``.
PLASMA_CELL_GENES = [
    "MZB1", "JCHAIN", "XBP1", "DERL3", "PRDM1", "SDC1", "TNFRSF17", "SLAMF7",
    "POU2AF1", "FKBP11", "SEC11C", "SSR4", "HERPUD1", "SEL1L", "PIM2", "ELL2",
    "CD38", "CD27", "CD79A", "CD79B", "MS4A1",
]

# Immunoglobulin heavy/kappa/lambda loci, by symbol prefix.
IG_PREFIXES = ("IGH", "IGK", "IGL")


@dataclass
class SpatialExport:
    """A loaded spatial-transcriptomics export, ready for the colorpath engine.

    Attributes
    ----------
    X        : (P, M) ``scipy.sparse`` matrix of spots x genes (non-negative).
    genes    : list of M gene symbols (columns of ``X``).
    barcodes : list of P spot barcodes (rows of ``X``).
    x, y     : (P,) integer grid coordinates of each spot.
    sample   : (P,) sample id of each spot (Visium runs often tile several sections).
    """

    X: sp.spmatrix
    genes: list[str]
    barcodes: list[str]
    x: np.ndarray
    y: np.ndarray
    sample: np.ndarray

    @property
    def n_spots(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def samples(self) -> list:
        """Unique sample ids, sorted."""
        return sorted(set(self.sample.tolist()))

    def sample_mask(self, sample) -> np.ndarray:
        """Boolean (P,) mask selecting the spots of one ``sample``."""
        return self.sample == sample


def load_spatial_export(
    directory: str | Path,
    *,
    genes_file: str | None = None,
    counts_file: str | None = None,
    barcodes_file: str = "spot_barcodes.txt",
    coords_file: str = "spatial_coordinates.csv",
) -> SpatialExport:
    """Load a directory exported as gene names + sparse counts + barcodes + coordinates.

    Parameters
    ----------
    directory     : folder containing the export.
    genes_file    : gene-symbol file (one per line). Defaults to the first
                    ``gene_names*.txt`` found.
    counts_file   : counts matrix. Defaults to the first ``*.csr.npz`` (preferred) or
                    ``*.mtx`` found; ``.npz`` is loaded with :func:`scipy.sparse.load_npz`,
                    ``.mtx`` with :func:`scipy.io.mmread`.
    barcodes_file : one barcode per line, in the row order of the matrix.
    coords_file   : CSV with (at least) ``barcode``, ``x``, ``y`` and ``sampleID`` columns;
                    rows are reordered to match ``barcodes_file``.

    Returns
    -------
    :class:`SpatialExport`.
    """
    directory = Path(directory)

    def _first(pattern):
        hits = sorted(directory.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"no {pattern!r} in {directory}")
        return hits[0]

    gpath = directory / genes_file if genes_file else _first("gene_names*.txt")
    genes = [line.strip() for line in gpath.read_text().splitlines() if line.strip()]

    if counts_file:
        cpath = directory / counts_file
    else:
        npz = sorted(directory.glob("*.csr.npz")) or sorted(directory.glob("*.npz"))
        cpath = npz[0] if npz else _first("*.mtx")
    if cpath.suffix == ".npz":
        X = sp.load_npz(cpath).tocsr()
    else:
        from scipy.io import mmread
        X = sp.csr_matrix(mmread(cpath))

    barcodes = [b.strip() for b in (directory / barcodes_file).read_text().splitlines()
                if b.strip()]

    # Read coordinates keyed by barcode, then reorder to the matrix's row order.
    coords: dict[str, tuple[float, float, object]] = {}
    with open(directory / coords_file, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            coords[row["barcode"]] = (
                float(row["x"]), float(row["y"]),
                row.get("sampleID", row.get("sample", 0)),
            )
    x = np.array([coords[b][0] for b in barcodes])
    y = np.array([coords[b][1] for b in barcodes])
    sample = np.array([coords[b][2] for b in barcodes])

    if not (X.shape[0] == len(barcodes) == len(x)):
        raise ValueError(
            f"row mismatch: matrix has {X.shape[0]} rows, {len(barcodes)} barcodes, "
            f"{len(x)} coordinates"
        )
    if X.shape[1] != len(genes):
        raise ValueError(f"column mismatch: matrix has {X.shape[1]} cols, {len(genes)} genes")
    return SpatialExport(X=X, genes=genes, barcodes=barcodes, x=x, y=y, sample=sample)


def select_genes(
    genes: list[str],
    symbols: list[str] | None = None,
    prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    """Indices (and names) of a gene set, by exact symbol and/or symbol prefix.

    Parameters
    ----------
    genes            : the full list of gene symbols (columns of ``X``).
    symbols          : exact symbols to include (missing ones are ignored).
    prefixes         : include any gene whose symbol starts with one of these (e.g.
                       ``("IGH", "IGK", "IGL")`` for the immunoglobulin loci).
    exclude_prefixes : drop genes matching these even if a ``prefixes`` rule caught them
                       (e.g. ``("IGHMBP",)`` to keep the IGHMBP helicase out of the Ig set).

    Returns
    -------
    (idx, names) : sorted integer column indices and the corresponding symbols.
    """
    pos = {g: i for i, g in enumerate(genes)}
    idx: set[int] = set()
    for s in symbols or []:
        if s in pos:
            idx.add(pos[s])
    if prefixes:
        for i, g in enumerate(genes):
            if g.startswith(tuple(prefixes)) and not g.startswith(tuple(exclude_prefixes)):
                idx.add(i)
    sel = np.array(sorted(idx), dtype=int)
    return sel, [genes[i] for i in sel]


def plasma_cell_gene_set(genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Convenience: the plasma-cell/B-lineage markers plus all immunoglobulin loci."""
    return select_genes(
        genes, symbols=PLASMA_CELL_GENES, prefixes=IG_PREFIXES,
        exclude_prefixes=("IGHMBP",),
    )


def to_dense(X) -> np.ndarray:
    """Return a dense float ndarray (accepts sparse or dense input)."""
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)


def per_gene_scale(X) -> np.ndarray:
    """Scale every gene (column) to its own maximum, so each lies in ``[0, 1]``.

    Equalises gene influence before factorising while staying non-negative (required by
    NMF). Without it, a handful of hyper-abundant genes each capture a component and the
    lower-abundance pathway members fragment into separate ones; with it, co-varying genes
    consolidate into a single, biologically coherent program.
    """
    Xd = to_dense(X)
    return Xd / np.maximum(Xd.max(axis=0, keepdims=True), 1e-9)


def sample_grid(x: np.ndarray, y: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Build ``(image_shape, pixel_index)`` for the activity-image renderer.

    Maps integer spot coordinates onto the smallest enclosing rectangular grid. The
    returned ``pixel_index`` places each spot's score at the right grid cell (cells with no
    spot render blank), and is passed straight to
    :func:`colorpath.illustration.render_pathway_activity_image`.

    Parameters
    ----------
    x, y : (P,) integer grid coordinates of the spots of a *single* section.

    Returns
    -------
    (image_shape, pixel_index) : ``((H, W), (P,) int array)``.
    """
    x = np.asarray(x); y = np.asarray(y)
    x0, y0 = int(x.min()), int(y.min())
    W = int(x.max()) - x0 + 1
    H = int(y.max()) - y0 + 1
    pixel_index = (y.astype(int) - y0) * W + (x.astype(int) - x0)
    return (H, W), pixel_index.astype(int)


def coexpression_edges(
    X,
    names: list[str],
    active: np.ndarray | None = None,
    threshold: float = 0.30,
) -> list[tuple[str, str]]:
    """Co-expression edges (Pearson > ``threshold``) among ``names``, for the graph view.

    Parameters
    ----------
    X         : (P, len(names)) matrix of the *selected* genes only.
    names     : gene symbols labelling the columns of ``X``.
    active    : optional boolean (P,) mask restricting the correlation to a subset of
                spots — pass the program-active spots (e.g. the top decile of a component's
                score) so the program's genes actually co-vary; across all spots the
                correlation is diluted by the many spots where every gene is zero.
    threshold : minimum Pearson correlation for an edge.

    Returns
    -------
    list of ``(gene_i, gene_j)`` edges (undirected; each pair once).
    """
    Xd = to_dense(X)
    if active is not None:
        Xd = Xd[np.asarray(active, dtype=bool)]
    if Xd.shape[0] < 3:
        return []
    C = np.corrcoef(Xd.T)
    edges: list[tuple[str, str]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(C[i, j]) and C[i, j] > threshold:
                edges.append((names[i], names[j]))
    return edges
