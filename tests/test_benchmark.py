"""Tests for the log-vs-linear recovery benchmark."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait.benchmark import (
    region_mutual_information,
    run_synthetic_benchmark,
    synthetic_coupling_dataset,
)


def test_synthetic_dataset_shapes():
    X, U, V = synthetic_coupling_dataset(side=12, n_components=3, per_group=5, seed=0)
    assert X.shape == (144, 15)
    assert U.shape == (144, 3) and V.shape == (3, 15)
    assert np.all(X >= 0)


def test_benchmark_reports_all_methods_and_metrics():
    res = run_synthetic_benchmark(side=14, n_init=1, max_iter=200, seed=0)
    assert {"GAIT (KL)", "GAIT (IS)", "linear NMF (Frob)", "log1p + NMF",
            "log1p + PCA"}.issubset(res)
    for sc in res.values():
        assert set(sc) == {"U_recovery", "V_recovery", "region_acc"}
        assert all(0.0 <= v <= 1.0 for v in sc.values())


def test_linear_recovers_coupling_better_than_log():
    # The R2 claim: linear-space NMF recovers the multiplicative loadings; the log pipelines
    # do not. Use a wide margin so the assertion is robust to the random seed / few restarts.
    res = run_synthetic_benchmark(side=16, n_init=1, max_iter=250, noise="mult", seed=0)
    gait_v = max(res["GAIT (KL)"]["V_recovery"], res["GAIT (IS)"]["V_recovery"])
    assert gait_v > 0.9
    assert gait_v > res["log1p + NMF"]["V_recovery"] + 0.1
    assert gait_v > res["log1p + PCA"]["V_recovery"] + 0.2


def test_region_mutual_information_detects_alignment():
    regions = np.array([0, 0, 0, 1, 1, 1])
    aligned = np.array([2, 2, 2, 5, 5, 5])        # perfectly predicts region
    independent = np.array([0, 1, 0, 1, 0, 1])    # tells you nothing
    assert region_mutual_information(aligned, regions) > \
           region_mutual_information(independent, regions)
    assert region_mutual_information(independent, regions) < 0.2
