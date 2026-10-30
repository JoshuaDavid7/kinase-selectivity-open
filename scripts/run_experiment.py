"""Run the full pipeline on one split design and write results to JSON.

Usage:
    python scripts/run_experiment.py --split scaffold --members 5 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinase_open import conformal, evaluate  # noqa: E402
from kinase_open.evaluate import TARGETS  # noqa: E402
from kinase_open.splits import murcko_scaffold, random_split, scaffold_split  # noqa: E402
from kinase_open.train import featurize_frame, train_ensemble  # noqa: E402


def four_way(smiles: list[str], design: str, seed: int):
    """train / val / calibration / test.

    Conformal needs a calibration set that the model never saw *and* that was
    not used for early stopping, or the coverage guarantee leaks. The middle
    block is therefore split again, by scaffold group, into val and cal.
    """
    if design == "scaffold":
        train, middle, test = scaffold_split(smiles, fractions=(0.70, 0.15, 0.15), seed=seed)
        mid_smiles = [smiles[i] for i in middle]
        groups: dict[str, list[int]] = {}
        for local, global_i in enumerate(middle):
            groups.setdefault(murcko_scaffold(mid_smiles[local]), []).append(global_i)
        ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
        val: list[int] = []
        cal: list[int] = []
        for group in ordered:
            (val if len(val) <= len(cal) else cal).extend(group)
        return train, np.array(sorted(val)), np.array(sorted(cal)), test

    train, middle, test = random_split(len(smiles), fractions=(0.70, 0.15, 0.15), seed=seed)
    half = len(middle) // 2
    return train, middle[:half], middle[half:], test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kind", default="hybrid", choices=["hybrid", "fp", "graph"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "jak_pki.csv")

    started = time.time()
    data = featurize_frame(df)
    train, val, cal, test = four_way(data.smiles, args.split, args.seed)

    pred_cal, _, _ = train_ensemble(
        data, train, val, cal, n_members=args.members, base_seed=args.seed * 100, epochs=args.epochs, kind=args.kind
    )
    pred_test, _, _ = train_ensemble(
        data, train, val, test, n_members=args.members, base_seed=args.seed * 100, epochs=args.epochs, kind=args.kind
    )

    y_cal, y_test = data.y[cal], data.y[test]
    results: dict = {
        "config": vars(args) | {"n_compounds": len(data)},
        "split_sizes": {"train": len(train), "val": len(val), "calibration": len(cal), "test": len(test)},
        "variance_decomposition": evaluate.variance_decomposition(data.y),
        "per_target": {},
        "uncertainty": {},
        "selectivity": {},
    }

    test_groups = np.array([murcko_scaffold(data.smiles[i]) for i in test])

    for t, name in enumerate(TARGETS):
        metrics = evaluate.regression_metrics(y_test[:, t], pred_test.mu[:, t])
        point, lo, hi = evaluate.scaffold_bootstrap_ci(y_test[:, t], pred_test.mu[:, t], test_groups)
        results["per_target"][name] = metrics.to_dict() | {"mae_ci95": [lo, hi]}

        try:
            calib = conformal.fit(y_cal[:, t], pred_cal.mu[:, t], pred_cal.sigma[:, t], alpha=0.1)
        except ValueError:
            continue
        c_lo, c_hi = calib.interval(pred_test.mu[:, t], pred_test.sigma[:, t])
        r_lo, r_hi = conformal.raw_gaussian_interval(pred_test.mu[:, t], pred_test.sigma[:, t], alpha=0.1)
        results["uncertainty"][name] = {
            "nominal": 0.90,
            "raw_ensemble_coverage": conformal.coverage(y_test[:, t], r_lo, r_hi),
            "raw_ensemble_width": conformal.mean_width(r_lo, r_hi),
            "conformal_coverage": conformal.coverage(y_test[:, t], c_lo, c_hi),
            "conformal_width": conformal.mean_width(c_lo, c_hi),
            "calibrator": calib.to_dict(),
        }

        true_margin = evaluate.selectivity_margin(y_test, t)
        pred_margin = evaluate.selectivity_margin(pred_test.mu, t)
        agree, majority = evaluate.margin_preference_accuracy(true_margin, pred_margin)
        margin_metrics = evaluate.regression_metrics(true_margin, pred_margin)
        results["selectivity"][name] = {
            "margin_mae": margin_metrics.mae,
            "margin_n": margin_metrics.n,
            "preference_accuracy": agree,
            "majority_rate": majority,
        }

    baselines = evaluate.classical_baselines(data.fps[train], data.y[train], data.fps[test], seed=args.seed)
    results["baselines"] = {
        name: {t: evaluate.regression_metrics(y_test[:, i], pred[:, i]).to_dict() for i, t in enumerate(TARGETS)}
        for name, pred in baselines.items()
    }

    results["runtime_seconds"] = round(time.time() - started, 1)

    out = Path(args.out) if args.out else root / "results" / f"{args.split}_{args.kind}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out}  ({results['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
