"""Recompute saved results from frozen predictions without retraining."""
from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import ROOT, TARGETS, conformal, digest, evaluate, json_safe, murcko_scaffold, selectivity_report

def same(actual, expected, label):
    actual, expected = json_safe(actual), json_safe(expected)
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: keys differ")
        for k in expected:
            same(actual[k], expected[k], f"{label}.{k}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: lengths differ")
        for i, value in enumerate(expected):
            same(actual[i], value, f"{label}[{i}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if actual is None or not np.isclose(actual, expected, rtol=1e-7, atol=1e-7):
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{label}: {actual} != {expected}")

def verify_run(path):
    path = Path(path)
    result = json.loads((path / "results.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    split = json.loads((path / "split.json").read_text())
    frozen = json.loads((path / "frozen.json").read_text())
    data_path = ROOT / "data" / "jak_pki.csv"
    if provenance["data_sha256"] != digest(data_path):
        raise AssertionError("data checksum mismatch")
    for key, file in [("predictions_sha256", "predictions.npz"), ("split_sha256", "split.json")]:
        if frozen[key] != digest(path / file):
            raise AssertionError(f"{path}: {file} checksum mismatch")
    # Auxiliary report tools can evolve. These files determine the fitted model,
    # splits and evaluation; their exact training-time bytes must still agree.
    for name, sha in provenance["source_sha256"].items():
        if name.startswith("src/") or name in {"scripts/benchmark.py", "scripts/run_experiment.py", "scripts/replicate.py"}:
            historical = subprocess.run(["git", "show", f"{provenance['git_base_commit']}:{name}"],
                cwd=ROOT, capture_output=True, check=False)
            archived_matches = historical.returncode == 0 and hashlib.sha256(historical.stdout).hexdigest() == sha
            if not archived_matches and digest(ROOT / name) != sha:
                raise AssertionError(f"no exact recorded source available: {name}")
    frame = pd.read_csv(data_path).set_index("canonical_smiles")
    y = frame.loc[split["smiles"], TARGETS].to_numpy(dtype=np.float32)
    parts = {k: np.asarray(v, dtype=int) for k, v in split["indices"].items()}
    combined = np.concatenate(list(parts.values()))
    if len(combined) != len(y) or sorted(combined.tolist()) != list(range(len(y))):
        raise AssertionError("split indices do not partition the dataset")
    if split["design"] == "scaffold":
        seen = set()
        for idx in parts.values():
            here = {murcko_scaffold(split["smiles"][i]) for i in idx}
            if seen & here:
                raise AssertionError("scaffold leakage")
            seen |= here
    train, val, cal, test = [parts[k] for k in ["train", "validation", "calibration", "test"]]
    preds = np.load(path / "predictions.npz", allow_pickle=False)
    baseline_val = {k.removeprefix("baseline_val_"): preds[k] for k in preds.files if k.startswith("baseline_val_")}
    selected = evaluate.select_baseline(y[val], baseline_val)
    same(selected, frozen["selected_baseline_by_validation"], "frozen selection")
    same(selected, result["selected_baseline_by_validation"], "reported selection")
    groups = np.array([murcko_scaffold(split["smiles"][i]) for i in test])
    test_mu, test_sigma = preds["test_mu"], preds["test_sigma"]
    for t, name in enumerate(TARGETS):
        metrics = evaluate.regression_metrics(y[test, t], test_mu[:, t]).to_dict()
        _, lo, hi = evaluate.scaffold_bootstrap_ci(y[test, t], test_mu[:, t], groups)
        same(result["per_target"][name], metrics | {"mae_ci95": [lo, hi]}, f"{name} metrics")
        diff, lo, hi = evaluate.paired_mae_difference_ci(y[test, t], test_mu[:, t], preds[f"baseline_test_{selected}"][:, t], groups)
        same(result["paired_comparison"][name], {"neural_minus_selected_baseline_mae": diff,
            "exploratory_ci95": [lo, hi], "multiplicity_adjusted": False}, f"{name} paired difference")
        if name in frozen["calibrators"]:
            spread = preds["val_sigma"][np.isfinite(y[val, t]), t]
            floor = max(float(np.percentile(spread, 10)), 1e-3) if len(spread) else 1e-3
            fit = conformal.fit(y[cal, t], preds["cal_mu"][:, t], preds["cal_sigma"][:, t], sigma_floor=floor)
            same(frozen["calibrators"][name], fit.to_dict(), f"{name} calibrator")
            lo, hi = fit.interval(test_mu[:, t], test_sigma[:, t])
            rlo, rhi = conformal.raw_gaussian_interval(test_mu[:, t], test_sigma[:, t])
            observed = np.isfinite(y[test, t])
            expected = {"nominal": .9, "conformal_coverage": conformal.coverage(y[test, t], lo, hi),
                "conformal_width_on_measured": conformal.mean_width(lo[observed], hi[observed]),
                "raw_ensemble_coverage": conformal.coverage(y[test, t], rlo, rhi),
                "raw_ensemble_width_on_measured": conformal.mean_width(rlo[observed], rhi[observed]),
                "calibrator": fit.to_dict(), "unbounded": not np.isfinite(fit.q),
                "exchangeability_guarantee_claimed_for_scaffold_split": False}
            same(result["uncertainty"][name], expected, f"{name} uncertainty")
    same(result["selectivity"], selectivity_report(y[test], test_mu, y[train]), "selectivity")
    for name in result["baselines"]:
        same(result["baselines"][name], {t: evaluate.regression_metrics(y[test, i], preds[f"baseline_test_{name}"][:, i]).to_dict()
            for i, t in enumerate(TARGETS)}, f"baseline {name}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "results" / "audit_v2")
    args = parser.parse_args()
    plan = json.loads((args.study / "plan.json").read_text())
    count = 0
    for seed in plan["config"]["seeds"]:
        for kind in plan["config"]["kinds"]:
            run = args.study / f"scaffold_{kind}_seed{seed}"
            verify_run(run)
            print(f"verified {run.name}", flush=True)
            count += 1
    if not count:
        raise AssertionError("no declared runs verified")
    print(f"{count} declared runs verified", flush=True)

if __name__ == "__main__":
    main()
