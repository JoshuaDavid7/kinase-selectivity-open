"""Regression tests for problems found during the adversarial review."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from kinase_open import conformal, evaluate
from kinase_open.data import curate
from kinase_open.featurize import FP_DIM, MORGAN_BITS
from kinase_open.model import FingerprintEncoder, masked_mse
from kinase_open.splits import scaffold_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import four_way, json_safe, selectivity_report, require_finite_predictions

def test_prediction_uses_only_measured_competitors():
    y = np.array([[8., 7., np.nan, np.nan]])
    pred = np.array([[8., 7., 12., 15.]])
    truth, estimate = evaluate.matched_selectivity_margins(y, pred, 0)
    assert truth[0] == estimate[0] == 1
    full_truth, full_estimate = evaluate.matched_selectivity_margins(y, pred, 0, complete_panel=True)
    assert np.isnan(full_truth[0]) and np.isnan(full_estimate[0])

def test_missing_prediction_does_not_shrink_observed_panel():
    truth, pred = evaluate.matched_selectivity_margins(
        np.array([[8., 7., 9., np.nan]]), np.array([[8., 7., np.nan, 5.]]), 0)
    assert np.isnan(truth[0]) and np.isnan(pred[0])

def test_conformal_quantile_is_exact_rank_not_numpy_interpolation():
    assert conformal._finite_quantile(np.arange(10.), alpha=.2) == 8

def test_small_calibration_requires_infinite_interval():
    fit = conformal.fit(np.array([0., 1.]), np.zeros(2), np.ones(2), alpha=.1)
    assert np.isinf(fit.q)
    lo, hi = fit.interval(np.zeros(2), np.ones(2))
    assert conformal.coverage(np.array([-1e6, 1e6]), lo, hi) == 1
    assert np.isinf(conformal.mean_width(lo, hi))

@pytest.mark.parametrize("alpha", [0, 1, -.1, 1.1])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        conformal._finite_quantile(np.array([1.]), alpha)

def test_floor_is_supplied_independently_of_calibration():
    fit = conformal.fit(np.arange(20.), np.zeros(20), np.arange(20.) + 1, sigma_floor=.25)
    assert fit.sigma_floor == .25
    with pytest.raises(ValueError):
        conformal.fit(np.ones(20), np.zeros(20), -np.ones(20))

def test_masked_nan_does_not_poison_loss_or_gradient():
    pred = torch.tensor([[2., 7.]], requires_grad=True)
    loss = masked_mse(pred, torch.tensor([[1., float("nan")]]), torch.tensor([[1., 0.]]))
    loss.backward()
    assert loss.item() == 1
    assert torch.equal(pred.grad, torch.tensor([[2., 0.]]))

def test_descriptor_scaling_ignores_test_and_preserves_morgan_bits():
    x = torch.zeros(3, FP_DIM)
    x[:, 0] = torch.tensor([0., 1., 1.])
    x[:, MORGAN_BITS:] = torch.tensor([10., 20., 10000.])[:, None]
    encoder = FingerprintEncoder()
    encoder.fit_descriptors(x[:2])
    scaled = (x - encoder.feature_mean) / encoder.feature_scale
    assert torch.equal(scaled[:, :MORGAN_BITS], x[:, :MORGAN_BITS])
    assert scaled[0, -1].item() == -1 and scaled[1, -1].item() == 1
    assert encoder.feature_mean[-1].item() == 15

def test_constant_descriptor_uses_unit_scale_for_new_values():
    x = torch.zeros(2, FP_DIM)
    x[:, MORGAN_BITS:] = 3
    encoder = FingerprintEncoder()
    encoder.fit_descriptors(x)
    new_value = torch.full((FP_DIM,), 5.)
    assert torch.all(encoder.feature_scale[MORGAN_BITS:] == 1)
    assert torch.all(((new_value - encoder.feature_mean) / encoder.feature_scale)[MORGAN_BITS:] == 2)

def test_benchmark_rejects_incomplete_predictions_instead_of_shrinking_test_set():
    with pytest.raises(ValueError, match='refusing partial scoring'):
        require_finite_predictions({'broken model': np.array([[1., np.nan]])})

def test_paired_interval_resamples_differences():
    y = np.zeros(12)
    a = np.arange(12.) + 1
    point, lo, hi = evaluate.paired_mae_difference_ci(y, a, a + .5, np.repeat(np.arange(4), 3))
    assert point == lo == hi == -.5

def test_baseline_choice_uses_validation_not_order():
    y = np.zeros((4, 4))
    assert evaluate.select_baseline(y, {"bad": y + 2, "good": y}) == "good"

def test_knn_and_other_baselines_do_not_treat_descriptors_as_bits():
    rng = np.random.default_rng(5)
    train = np.zeros((8, FP_DIM), dtype=np.float32)
    train[:, :10] = rng.integers(0, 2, (8, 10))
    y = rng.normal(size=(8, 1))
    a = evaluate.classical_baselines(train, y, train[:2], seed=1)
    train[:, MORGAN_BITS:] = rng.normal(size=(8, FP_DIM - MORGAN_BITS)) * 100
    b = evaluate.classical_baselines(train, y, train[:2], seed=1)
    assert all(np.array_equal(a[k], b[k]) for k in a if not k.endswith("_descriptors"))
    assert {"ridge_descriptors", "random_forest_descriptors", "extra_trees_descriptors"} <= set(b)
    assert all(np.isfinite(v).all() for v in b.values())

def test_seed_replication_does_not_claim_new_scaffold_partitions():
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "C", "CC"]
    a, b = four_way(smiles, "scaffold", 1), four_way(smiles, "scaffold", 99)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

def test_selectivity_reports_separate_partial_complete_and_pairwise():
    y = np.array([[8., 7., np.nan, np.nan], [8., 7., 8., 8.]])
    pred = np.nan_to_num(y, nan=15.)
    r = selectivity_report(y, pred, y)
    assert r["measured_subset"]["JAK1"]["n"] == 2
    assert r["complete_panel"]["JAK1"]["n"] == 1
    assert r["pairwise"]["JAK1/JAK3"]["exact_ties"] == 1
    assert r["pairwise"]["JAK1/JAK3"]["direction_n_excluding_exact_ties"] == 0

def test_curation_drops_non_numeric_and_infinite_pchembl():
    df = pd.DataFrame({"canonical_smiles": ["CC"] * 4, "target": ["JAK1"] * 4,
        "standard_relation": ["="] * 4, "data_validity_comment": [None] * 4,
        "pchembl_value": [7., "bad", np.inf, None]})
    table, report = curate(df)
    assert len(table) == 1 and table.JAK1.iloc[0] == 7
    assert report.dropped_no_pchembl == 3

def test_strict_json_has_no_nan_or_infinity_tokens():
    assert json_safe({"a": np.inf, "b": np.nan, "path": Path("test")}) == {"a": None, "b": None, "path": "test"}
