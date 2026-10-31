"""Versioned benchmark; save predictions before scoring test labels."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kinase_open import conformal, evaluate
from kinase_open.evaluate import TARGETS
from kinase_open.splits import murcko_scaffold, random_split, scaffold_split
from kinase_open.train import featurize_frame, train_ensemble_multi

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def four_way(smiles, design, seed):
    if design == "scaffold":
        train, middle, test = scaffold_split(smiles, fractions=(0.70, 0.15, 0.15))
        groups = {}
        for i in middle:
            groups.setdefault(murcko_scaffold(smiles[i]), []).append(int(i))
        val, cal = [], []
        for group in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
            (val if len(val) <= len(cal) else cal).extend(group)
        return train, np.array(sorted(val), dtype=int), np.array(sorted(cal), dtype=int), test
    train, middle, test = random_split(len(smiles), fractions=(0.70, 0.15, 0.15), seed=seed)
    middle = np.random.default_rng(seed + 1701).permutation(middle)
    half = len(middle) // 2
    return train, np.sort(middle[:half]), np.sort(middle[half:]), test

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")

def selectivity_report(y, pred, y_train):
    report = {"measured_subset": {}, "complete_panel": {}, "pairwise": {}}
    for complete, section in [(False, "measured_subset"), (True, "complete_panel")]:
        for t, name in enumerate(TARGETS):
            truth, estimate = evaluate.matched_selectivity_margins(y, pred, t, complete_panel=complete)
            train_truth, _ = evaluate.matched_selectivity_margins(y_train, y_train, t, complete_panel=complete)
            train_ok, test_ok = np.isfinite(train_truth), np.isfinite(truth) & np.isfinite(estimate)
            agree, test_majority = evaluate.margin_preference_accuracy(truth, estimate)
            positive = bool(np.mean(train_truth[train_ok] > 0) > 0.5) if train_ok.any() else None
            baseline = float(np.mean((truth[test_ok] > 0) == positive)) if positive is not None and test_ok.any() else None
            report[section][name] = evaluate.regression_metrics(truth, estimate).to_dict() | {
                "preference_accuracy": agree, "training_majority_baseline_accuracy": baseline,
                "test_majority_rate_descriptive_only": test_majority,
                "exact_zero_truth_margins": int(np.sum(truth[test_ok] == 0)),
            }
    for a, b in itertools.combinations(range(4), 2):
        truth, estimate = y[:, a] - y[:, b], pred[:, a] - pred[:, b]
        measured = np.isfinite(truth)
        strict = measured & (truth != 0) & np.isfinite(estimate)
        train_margin = y_train[:, a] - y_train[:, b]
        tr = np.isfinite(train_margin) & (train_margin != 0)
        majority = bool(np.mean(train_margin[tr] > 0) > 0.5) if tr.any() else None
        report["pairwise"][f"{TARGETS[a]}/{TARGETS[b]}"] = evaluate.regression_metrics(truth, estimate).to_dict() | {
            "exact_ties": int(np.sum(measured & (truth == 0))), "direction_n_excluding_exact_ties": int(strict.sum()),
            "direction_accuracy_excluding_exact_ties": float(np.mean((truth[strict] > 0) == (estimate[strict] > 0))) if strict.any() else None,
            "training_majority_baseline_accuracy": float(np.mean((truth[strict] > 0) == majority)) if majority is not None and strict.any() else None,
        }
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kind", choices=["hybrid", "fp", "graph"], default="hybrid")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if min(args.members, args.epochs, args.threads) < 1:
        parser.error("members, epochs, and threads must be positive")
    torch.set_num_threads(args.threads)
    started = time.time()
    out = args.out or ROOT / "results" / "audit_v2" / f"{args.split}_{args.kind}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "jak_pki.csv"
    frame = pd.read_csv(data_path)
    data = featurize_frame(frame)
    train, val, cal, test = four_way(data.smiles, args.split, args.seed)
    groups = np.array([murcko_scaffold(s) for s in data.smiles])
    parts = {"train": train, "validation": val, "calibration": cal, "test": test}
    if len(np.unique(np.concatenate(list(parts.values())))) != len(data):
        raise RuntimeError("split is not a partition")
    if args.split == "scaffold":
        for a, b in itertools.combinations(parts, 2):
            if set(groups[parts[a]]) & set(groups[parts[b]]):
                raise RuntimeError(f"scaffold leakage between {a} and {b}")
    write_json(out / "split.json", {"design": args.split,
        "seed_changes_membership": args.split == "random", "indices": parts, "smiles": data.smiles})
    source_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    write_json(out / "provenance.json", {
        "protocol": "audit_v2", "config": vars(args) | {"out": str(out)},
        "python": sys.version, "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "torch", "rdkit", "scikit-learn", "scipy"]},
        "data_sha256": digest(data_path), "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT, text=True).strip()),
        "rows_in_csv": len(frame), "rows_featurized": len(data),
        "limitations": ["retrospective public labels", "fixed chemical partition for scaffold runs", "heterogeneous assays", "no prospective validation"],
    })
    pred_val, pred_cal, pred_test = train_ensemble_multi(data, train, val, [val, cal, test],
        n_members=args.members, base_seed=args.seed * 100, epochs=args.epochs, kind=args.kind,
        checkpoint_dir=out / "models" if args.save_models else None)
    combined = np.concatenate([val, test])
    baselines_all = evaluate.classical_baselines(data.fps[train], data.y[train], data.fps[combined], seed=args.seed)
    baselines_val = {k: v[:len(val)] for k, v in baselines_all.items()}
    baselines_test = {k: v[len(val):] for k, v in baselines_all.items()}
    selected = evaluate.select_baseline(data.y[val], baselines_val)
    calibrators = {}
    for t, name in enumerate(TARGETS):
        spread = pred_val.sigma[np.isfinite(data.y[val, t]), t]
        floor = max(float(np.percentile(spread, 10)), 1e-3) if len(spread) else 1e-3
        try:
            calibrators[name] = conformal.fit(data.y[cal, t], pred_cal.mu[:, t], pred_cal.sigma[:, t], sigma_floor=floor)
        except ValueError:
            continue
    predictions_path = out / "predictions.npz"
    np.savez_compressed(predictions_path, val_mu=pred_val.mu, val_sigma=pred_val.sigma,
        cal_mu=pred_cal.mu, cal_sigma=pred_cal.sigma, test_mu=pred_test.mu, test_sigma=pred_test.sigma,
        **{f"baseline_val_{k}": v for k, v in baselines_val.items()},
        **{f"baseline_test_{k}": v for k, v in baselines_test.items()})
    write_json(out / "frozen.json", {"predictions_sha256": digest(predictions_path),
        "split_sha256": digest(out / "split.json"), "selected_baseline_by_validation": selected,
        "calibrators": {k: v.to_dict() for k, v in calibrators.items()}, "frozen_before_test_scoring": True})
    # Test labels enter scoring only after the artifacts above are frozen. This
    # is a software boundary on public data, not independent external blinding.
    y_test = data.y[test]
    result = {"protocol": "audit_v2", "config": vars(args) | {"out": str(out)},
        "split_sizes": {k: len(v) for k, v in parts.items()}, "scaffold_groups": {k: len(set(groups[v])) for k, v in parts.items()},
        "selected_baseline_by_validation": selected, "per_target": {}, "uncertainty": {}, "baselines": {}, "paired_comparison": {},
        "selectivity": selectivity_report(y_test, pred_test.mu, data.y[train])}
    for t, name in enumerate(TARGETS):
        metrics = evaluate.regression_metrics(y_test[:, t], pred_test.mu[:, t]).to_dict()
        _, lo, hi = evaluate.scaffold_bootstrap_ci(y_test[:, t], pred_test.mu[:, t], groups[test])
        result["per_target"][name] = metrics | {"mae_ci95": [lo, hi]}
        diff, lo, hi = evaluate.paired_mae_difference_ci(y_test[:, t], pred_test.mu[:, t], baselines_test[selected][:, t], groups[test])
        result["paired_comparison"][name] = {"neural_minus_selected_baseline_mae": diff, "exploratory_ci95": [lo, hi], "multiplicity_adjusted": False}
        if name in calibrators:
            calib = calibrators[name]
            lo, hi = calib.interval(pred_test.mu[:, t], pred_test.sigma[:, t])
            raw_lo, raw_hi = conformal.raw_gaussian_interval(pred_test.mu[:, t], pred_test.sigma[:, t])
            measured = np.isfinite(y_test[:, t])
            result["uncertainty"][name] = {"nominal": 0.9,
                "conformal_coverage": conformal.coverage(y_test[:, t], lo, hi),
                "conformal_width_on_measured": conformal.mean_width(lo[measured], hi[measured]),
                "raw_ensemble_coverage": conformal.coverage(y_test[:, t], raw_lo, raw_hi),
                "raw_ensemble_width_on_measured": conformal.mean_width(raw_lo[measured], raw_hi[measured]),
                "calibrator": calib.to_dict(), "unbounded": not np.isfinite(calib.q),
                "exchangeability_guarantee_claimed_for_scaffold_split": False}
    for name, pred in baselines_test.items():
        result["baselines"][name] = {t: evaluate.regression_metrics(y_test[:, i], pred[:, i]).to_dict() for i, t in enumerate(TARGETS)}
    result["runtime_seconds"] = round(time.time() - started, 3)
    write_json(out / "results.json", result)
    print(f"wrote {out}; {result['runtime_seconds']} seconds; validation-selected comparator: {selected}", flush=True)

if __name__ == "__main__":
    main()
