"""Tests for the parts that fail silently rather than loudly.

Each test here corresponds to a mistake that produces a plausible-looking
number instead of a crash: a graph that leaks messages back along a bond, a
loss that treats an unmeasured compound as inactive, a split that lets the same
scaffold sit on both sides, an interval that is narrower than its guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kinase_open import conformal, evaluate
from kinase_open.featurize import ATOM_DIM, BOND_DIM, FP_DIM, collate, fingerprint, mol_to_graph, parse
from kinase_open.model import MolecularNetwork, masked_mse
from kinase_open.splits import murcko_scaffold, scaffold_split
from kinase_open.train import Standardizer

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "C",  # single atom, no bonds - the degenerate graph
    "CCO",
]


def graphs():
    return [mol_to_graph(parse(s)) for s in SMILES]


class TestGraphConstruction:
    def test_reverse_index_is_an_involution(self):
        """rev[rev[e]] == e, or messages leak back along the bond they came from."""
        batch = collate(graphs())
        assert torch.equal(batch.rev[batch.rev], torch.arange(batch.rev.numel()))

    def test_reverse_edge_swaps_endpoints(self):
        batch = collate(graphs())
        assert torch.equal(batch.src[batch.rev], batch.dst)
        assert torch.equal(batch.dst[batch.rev], batch.src)

    def test_every_bond_is_directed_both_ways(self):
        for g in graphs():
            assert g.n_bonds % 2 == 0

    def test_feature_dimensions_are_fixed(self):
        batch = collate(graphs())
        assert batch.atom_f.shape[1] == ATOM_DIM
        if batch.bond_f.numel():
            assert batch.bond_f.shape[1] == BOND_DIM
        assert fingerprint(parse(SMILES[0])).shape == (FP_DIM,)

    def test_batching_offsets_do_not_collide(self):
        """Atom indices must point inside the molecule they came from."""
        batch = collate(graphs())
        for e in range(batch.src.numel()):
            assert batch.atom_to_mol[batch.src[e]] == batch.atom_to_mol[batch.dst[e]]

    def test_unknown_element_does_not_crash(self):
        g = mol_to_graph(parse("[Zr]"))
        assert g.atom_f.shape[1] == ATOM_DIM


class TestMaskedLoss:
    def test_missing_labels_contribute_nothing(self):
        pred = torch.tensor([[1.0, 5.0]])
        target = torch.tensor([[1.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])
        assert masked_mse(pred, target, mask).item() == pytest.approx(0.0)

    def test_missing_is_not_treated_as_zero(self):
        """The whole point: an unmeasured target must not read as inactive."""
        pred = torch.tensor([[2.0, 9.0]])
        target = torch.tensor([[2.0, 0.0]])
        masked = masked_mse(pred, target, torch.tensor([[1.0, 0.0]]))
        unmasked = masked_mse(pred, target, torch.ones(1, 2))
        assert masked.item() < unmasked.item()

    def test_all_missing_gives_zero_not_nan(self):
        loss = masked_mse(torch.randn(3, 4), torch.zeros(3, 4), torch.zeros(3, 4))
        assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0)


class TestModel:
    @pytest.mark.parametrize("kind", ["hybrid", "fp", "graph"])
    def test_forward_shape_and_gradients(self, kind):
        batch = collate(graphs())
        fps = torch.tensor(np.stack([fingerprint(parse(s)) for s in SMILES]))
        net = MolecularNetwork(kind=kind)
        out = net(batch if kind != "fp" else None, fps if kind != "graph" else None)
        assert out.shape == (len(SMILES), 4)
        out.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

    def test_depth_changes_the_representation(self):
        """If depth did nothing, the message passing is not wired up."""
        batch = collate(graphs())
        torch.manual_seed(0)
        shallow = MolecularNetwork(kind="graph", depth=1, dropout=0.0).embed(batch, None)
        torch.manual_seed(0)
        deep = MolecularNetwork(kind="graph", depth=4, dropout=0.0).embed(batch, None)
        assert not torch.allclose(shallow, deep)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            MolecularNetwork(kind="nonsense")


class TestSplits:
    def test_no_scaffold_spans_train_and_test(self):
        smiles = SMILES * 12
        train, _, test = scaffold_split(smiles)
        s_train = {murcko_scaffold(smiles[i]) for i in train}
        s_test = {murcko_scaffold(smiles[i]) for i in test}
        assert not (s_train & s_test)

    def test_split_is_a_partition(self):
        smiles = SMILES * 12
        train, val, test = scaffold_split(smiles)
        combined = np.concatenate([train, val, test])
        assert sorted(combined) == list(range(len(smiles)))

    def test_split_is_deterministic(self):
        smiles = SMILES * 12
        a = scaffold_split(smiles)
        b = scaffold_split(smiles)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


class TestConformal:
    def test_coverage_holds_on_exchangeable_data(self):
        rng = np.random.default_rng(0)
        n = 4000
        sigma = rng.uniform(0.2, 1.0, n)
        mu = rng.normal(0, 1, n)
        y = mu + rng.normal(0, 1, n) * sigma

        half = n // 2
        calib = conformal.fit(y[:half], mu[:half], sigma[:half], alpha=0.1)
        lo, hi = calib.interval(mu[half:], sigma[half:])
        assert conformal.coverage(y[half:], lo, hi) >= 0.86  # 0.90 target, sampling slack

    def test_tighter_alpha_gives_wider_intervals(self):
        rng = np.random.default_rng(1)
        n = 2000
        sigma = rng.uniform(0.2, 1.0, n)
        mu = rng.normal(0, 1, n)
        y = mu + rng.normal(0, 1, n) * sigma
        wide = conformal.fit(y, mu, sigma, alpha=0.01)
        narrow = conformal.fit(y, mu, sigma, alpha=0.2)
        assert wide.q > narrow.q

    def test_floor_bounds_a_collapsed_sigma(self):
        """A near-zero sigma must not blow the quantile up."""
        rng = np.random.default_rng(2)
        n = 1000
        sigma = np.concatenate([np.full(50, 1e-12), rng.uniform(0.3, 1.0, n - 50)])
        mu = np.zeros(n)
        y = rng.normal(0, 0.5, n)
        calib = conformal.fit(y, mu, sigma, alpha=0.1)
        assert calib.sigma_floor > 0
        assert np.isfinite(calib.q) and calib.q < 1e6

    def test_ignores_unmeasured_entries(self):
        y = np.array([1.0, np.nan, 2.0, np.nan])
        mu = np.array([1.1, 0.0, 2.2, 0.0])
        sigma = np.array([0.3, 0.3, 0.3, 0.3])
        assert conformal.fit(y, mu, sigma, alpha=0.1).n_calibration == 2


class TestEvaluate:
    def test_margin_is_nan_without_a_comparison(self):
        y = np.array([[8.0, np.nan, np.nan, np.nan]])
        assert np.isnan(evaluate.selectivity_margin(y, 0)[0])

    def test_margin_uses_the_best_competitor(self):
        y = np.array([[9.0, 7.0, 8.5, 6.0]])
        assert evaluate.selectivity_margin(y, 0)[0] == pytest.approx(0.5)

    def test_majority_rate_is_reported_alongside_accuracy(self):
        truth = np.array([1.0] * 9 + [-1.0])
        always_positive = np.ones(10)
        agree, majority = evaluate.margin_preference_accuracy(truth, always_positive)
        assert agree == pytest.approx(majority)

    def test_r2_of_a_constant_predictor_is_about_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert evaluate.regression_metrics(y, np.full(4, y.mean())).r2 == pytest.approx(0.0)

    def test_bootstrap_resamples_groups_not_rows(self):
        """One group means every resample is identical, so the CI collapses."""
        rng = np.random.default_rng(0)
        y = rng.normal(size=60)
        pred = y + rng.normal(scale=0.2, size=60)
        _, lo, hi = evaluate.scaffold_bootstrap_ci(y, pred, np.zeros(60, dtype=int), n_boot=200)
        assert hi - lo == pytest.approx(0.0, abs=1e-9)

    def test_variance_decomposition_detects_pure_potency(self):
        """Four identical values per compound means zero selectivity."""
        base = np.array([5.0, 7.0, 9.0])
        y = np.repeat(base[:, None], 4, axis=1)
        assert evaluate.variance_decomposition(y)["selectivity_fraction"] == pytest.approx(0.0)


class TestStandardizer:
    def test_roundtrip(self):
        y = np.array([[5.0, 8.0], [6.0, 9.0], [7.0, 10.0]], dtype=np.float32)
        std = Standardizer.fit(y, np.ones_like(y))
        assert np.allclose(std.decode(std.encode(y)), y, atol=1e-5)

    def test_ignores_unmeasured_entries(self):
        y = np.array([[5.0, np.nan], [7.0, np.nan]], dtype=np.float32)
        mask = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        assert Standardizer.fit(y, mask).mean[0] == pytest.approx(6.0)

    def test_constant_target_does_not_divide_by_zero(self):
        y = np.full((4, 1), 3.0, dtype=np.float32)
        std = Standardizer.fit(y, np.ones_like(y))
        assert np.isfinite(std.encode(y)).all()
