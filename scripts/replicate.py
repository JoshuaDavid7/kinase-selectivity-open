"""Does the winner survive replication?

Picking a model on one seed is the standard way to publish a result that does
not reproduce. This trains every architecture on several seeds and asks two
separate questions:

  1. How much does a single target's MAE move between seeds? That is the noise
     floor - any difference smaller than it is not a finding.
  2. Does the *ranking* change? A pipeline that selects a different winner on
     seed 2 than on seed 1 has not selected anything.

Usage:
    python scripts/replicate.py --seeds 1 2 3 --members 3 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinase_open.evaluate import TARGETS, regression_metrics  # noqa: E402
from kinase_open.train import featurize_frame, train_ensemble  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import four_way  # noqa: E402

KINDS = ["hybrid", "fp", "graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--kinds", nargs="+", default=KINDS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = featurize_frame(pd.read_csv(root / "data" / "jak_pki.csv"))

    # {kind: {seed: {target: mae}}}
    table: dict[str, dict[int, dict[str, float]]] = {k: {} for k in args.kinds}

    for seed in args.seeds:
        train, val, _cal, test = four_way(data.smiles, args.split, seed)
        for kind in args.kinds:
            pred, _, _ = train_ensemble(
                data,
                train,
                val,
                test,
                n_members=args.members,
                base_seed=seed * 100,
                epochs=args.epochs,
                kind=kind,
            )
            table[kind][seed] = {
                t: regression_metrics(data.y[test][:, i], pred.mu[:, i]).mae for i, t in enumerate(TARGETS)
            }
            mean_mae = float(np.nanmean(list(table[kind][seed].values())))
            print(f"seed {seed}  {kind:7s}  mean MAE {mean_mae:.4f}", flush=True)

    summary: dict = {"config": vars(args), "per_kind": {}, "winner_by_seed": {}, "stable": None}

    for kind in args.kinds:
        per_seed_mean = {s: float(np.nanmean(list(v.values()))) for s, v in table[kind].items()}
        values = list(per_seed_mean.values())
        summary["per_kind"][kind] = {
            "mean_mae_by_seed": per_seed_mean,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_target_mean": {
                t: float(np.mean([table[kind][s][t] for s in args.seeds])) for t in TARGETS
            },
        }

    for seed in args.seeds:
        ranked = sorted(args.kinds, key=lambda k: np.nanmean(list(table[k][seed].values())))
        summary["winner_by_seed"][str(seed)] = ranked[0]

    winners = set(summary["winner_by_seed"].values())
    summary["stable"] = len(winners) == 1

    # The comparison that matters: is the gap between the top two bigger than
    # the seed-to-seed spread of either? If not, the ranking is noise.
    ordered = sorted(args.kinds, key=lambda k: summary["per_kind"][k]["mean"])
    if len(ordered) >= 2:
        best, second = ordered[0], ordered[1]
        gap = summary["per_kind"][second]["mean"] - summary["per_kind"][best]["mean"]
        spread = max(summary["per_kind"][best]["sd"], summary["per_kind"][second]["sd"])
        summary["separation"] = {
            "best": best,
            "second": second,
            "gap": gap,
            "largest_seed_sd": spread,
            "gap_exceeds_seed_noise": bool(gap > spread),
        }

    out = root / "results" / f"replication_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\nwinner by seed: {summary['winner_by_seed']}")
    print(f"ranking stable across seeds: {summary['stable']}")
    if "separation" in summary:
        sep = summary["separation"]
        print(
            f"{sep['best']} beats {sep['second']} by {sep['gap']:.4f} MAE; "
            f"largest seed-to-seed SD is {sep['largest_seed_sd']:.4f} -> "
            f"{'a real gap' if sep['gap_exceeds_seed_noise'] else 'INSIDE seed noise, not a finding'}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
