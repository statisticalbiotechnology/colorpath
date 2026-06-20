"""Tests for the spatial-transcriptomics (Visium) loader and helpers.

A tiny synthetic export is written to a temp directory and pushed through the real
decomposition + illustration layers, so the Visium path is exercised end-to-end without
the large breast-cancer data.
"""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorpath.decomposition import LinearNMF, loading_share, variation_explained
from colorpath.illustration import illustrate_component
from colorpath.spatial import (
    coexpression_edges,
    load_spatial_export,
    per_gene_scale,
    plasma_cell_gene_set,
    sample_grid,
    select_genes,
)


# ----------------------------- synthetic export -----------------------------

def write_export(directory, P=120, seed=0):
    """Write a minimal Visium-style export: two co-expressed plasma genes + filler."""
    rng = np.random.default_rng(seed)
    genes = ["IGKC", "MZB1", "DERL3", "ACTB", "GAPDH", "IGHMBP2"]
    # A spatial program active in a corner of an 11x11 grid drives IGKC/MZB1/DERL3 together.
    side = 11
    yy, xx = np.mgrid[0:side, 0:side]
    activity = np.exp(-(((xx - 2) ** 2 + (yy - 2) ** 2) / 6.0)).ravel()[:P]
    M = len(genes)
    X = np.zeros((P, M))
    X[:, 0] = activity * 8 + 0.01 * rng.random(P)     # IGKC (abundant)
    X[:, 1] = activity * 3 + 0.01 * rng.random(P)     # MZB1
    X[:, 2] = activity * 2 + 0.01 * rng.random(P)     # DERL3
    X[:, 3] = 0.5 + 0.5 * rng.random(P)               # ACTB (housekeeping)
    X[:, 4] = 0.5 + 0.5 * rng.random(P)               # GAPDH
    X[:, 5] = 0.01 * rng.random(P)                     # IGHMBP2 (should be excluded)

    barcodes = [f"spot_{i}" for i in range(P)]
    directory = str(directory)
    (open(os.path.join(directory, "gene_names_hvg.txt"), "w")
     .write("\n".join(genes) + "\n"))
    (open(os.path.join(directory, "spot_barcodes.txt"), "w")
     .write("\n".join(barcodes) + "\n"))
    sp.save_npz(os.path.join(directory, "gene_counts_hvg.csr.npz"), sp.csr_matrix(X))
    with open(os.path.join(directory, "spatial_coordinates.csv"), "w") as fh:
        fh.write('"","barcode","x","y","sampleID"\n')
        for i, b in enumerate(barcodes):
            fh.write(f'"{b}","{b}",{xx.ravel()[i]},{yy.ravel()[i]},1\n')
    return genes, barcodes, X


# ----------------------------- loader -----------------------------

def test_load_spatial_export_shapes(tmp_path):
    genes, barcodes, X = write_export(tmp_path)
    exp = load_spatial_export(tmp_path)
    assert exp.n_spots == len(barcodes)
    assert exp.genes == genes
    assert exp.barcodes == barcodes
    assert exp.X.shape == (len(barcodes), len(genes))
    assert np.allclose(exp.X.toarray(), X)
    assert exp.samples() == [1] or exp.samples() == ["1"]


def test_load_reorders_coords_to_matrix_rows(tmp_path):
    genes, barcodes, X = write_export(tmp_path)
    # Shuffle the coordinate-file order; loader must realign to the barcode order.
    path = os.path.join(str(tmp_path), "spatial_coordinates.csv")
    lines = open(path).read().splitlines()
    header, body = lines[0], lines[1:]
    rng = np.random.default_rng(1)
    body = list(np.array(body)[rng.permutation(len(body))])
    open(path, "w").write("\n".join([header] + body) + "\n")
    exp = load_spatial_export(tmp_path)
    # x,y of the first spot must still correspond to barcode spot_0 (grid origin 0,0).
    assert (exp.x[0], exp.y[0]) == (0, 0)


# ----------------------------- gene selection -----------------------------

def test_select_genes_symbol_and_prefix_with_exclude():
    genes = ["IGKC", "IGHG1", "IGHMBP2", "MZB1", "ACTB"]
    idx, names = select_genes(
        genes, symbols=["MZB1"], prefixes=("IGH", "IGK"),
        exclude_prefixes=("IGHMBP",),
    )
    assert "IGHMBP2" not in names           # excluded despite IGH prefix
    assert set(names) == {"IGKC", "IGHG1", "MZB1"}
    assert list(idx) == sorted(idx)


def test_plasma_cell_gene_set(tmp_path):
    genes, _, _ = write_export(tmp_path)
    idx, names = plasma_cell_gene_set(genes)
    assert {"IGKC", "MZB1", "DERL3"}.issubset(names)
    assert "IGHMBP2" not in names           # IGHMBP excluded
    assert "ACTB" not in names              # housekeeping not in the module


# ----------------------------- grid / scaling / edges -----------------------------

def test_sample_grid_places_scores():
    x = np.array([0, 1, 0, 1]); y = np.array([0, 0, 1, 1])
    shape, pix = sample_grid(x, y)
    assert shape == (2, 2)
    assert sorted(pix.tolist()) == [0, 1, 2, 3]


def test_per_gene_scale_unit_max_nonnegative():
    X = np.array([[0.0, 2.0], [4.0, 1.0], [2.0, 0.5]])
    Xs = per_gene_scale(X)
    assert np.allclose(Xs.max(axis=0), 1.0)
    assert np.all(Xs >= 0)


def test_coexpression_edges_finds_coexpressed_core(tmp_path):
    genes, _, X = write_export(tmp_path)
    Xs = per_gene_scale(X)
    edges = coexpression_edges(Xs, genes, threshold=0.5)
    flat = {g for e in edges for g in e}
    # The three co-driven plasma genes should connect; housekeeping should not join them.
    assert {"IGKC", "MZB1", "DERL3"}.issubset(flat)
    assert ("ACTB", "GAPDH") not in edges and ("GAPDH", "ACTB") not in edges


def test_coexpression_edges_active_subset_and_small_n():
    assert coexpression_edges(np.zeros((2, 3)), ["a", "b", "c"]) == []  # <3 spots


# ----------------------------- loading_share -----------------------------

def test_loading_share_sums_to_one_per_gene():
    rng = np.random.default_rng(0)
    U = rng.random((100, 3)) + 0.1
    V = rng.random((3, 8)) + 0.1
    L = loading_share(U, V, normalize="sum")
    assert L.shape == (3, 8)
    assert np.all((L >= 0) & (L <= 1))
    assert np.allclose(L.sum(axis=0), 1.0)


def test_loading_share_scale_invariant_to_component_rescale():
    rng = np.random.default_rng(1)
    U = rng.random((80, 3)) + 0.1
    V = rng.random((3, 6)) + 0.1
    c = np.array([5.0, 0.2, 3.0])
    assert np.allclose(loading_share(U, V), loading_share(U * c, V / c[:, None]))


def test_loading_share_gentler_than_variation_explained():
    # One gene is owned by comp 0 but at low abundance; another is hugely abundant in comp 1.
    U = np.abs(np.random.default_rng(2).standard_normal((200, 2))) + 0.05
    V = np.array([[1.0, 0.0], [0.0, 50.0]])
    Lshare = loading_share(U, V)
    F = variation_explained(U, V)
    # Both assign the low-abundance gene to comp 0...
    assert Lshare[0, 0] > 0.9 and F[0, 0] > 0.9
    # ...and neither lets comp 0 claim the abundant comp-1 gene.
    assert Lshare[0, 1] < 0.1 and F[0, 1] < 0.1


# ----------------------------- end-to-end through the engine -----------------------------

def test_end_to_end_visium_pipeline(tmp_path):
    genes, _, _ = write_export(tmp_path)
    exp = load_spatial_export(tmp_path)
    gidx, names = plasma_cell_gene_set(exp.genes)
    Xs = per_gene_scale(exp.X[:, gidx])

    res = LinearNMF(3, loss="kl", max_iter=300, random_state=0).fit(Xs)
    assert np.all(res.U >= 0) and np.all(res.V >= 0)

    ref = Xs[:, names.index("IGKC")]
    k = int(np.nanargmax([np.corrcoef(res.U[:, j], ref)[0, 1] for j in range(res.K)]))

    active = res.U[:, k] > np.percentile(res.U[:, k], 90)
    edges = coexpression_edges(Xs, names, active=active, threshold=0.3)
    image_shape, pixel_index = sample_grid(exp.x, exp.y)

    out_graph = os.path.join(str(tmp_path), "g.svg")
    out_image = os.path.join(str(tmp_path), "i.svg")
    paths = illustrate_component(
        res, component=k, metabolite_names=names, pathway_edges=edges,
        image_shape=image_shape, pixel_index=pixel_index,
        graph_value="loading_share",
        image_kwargs=dict(vmin=0.0, vmax=float(np.percentile(res.U[:, k], 99))),
        graph_kwargs=dict(layout="spring"),
        graph_output=out_graph, image_output=out_image,
    )
    assert os.path.exists(paths["graph"]) and os.path.exists(paths["image"])
