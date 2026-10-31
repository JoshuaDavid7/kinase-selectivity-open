"""Describe repeated labels without inferring their experimental provenance.

Equal recorded values can reflect rounding, genuine similarity, copied values,
or other causes. Equality alone cannot distinguish these explanations.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinase_open.evaluate import TARGETS, variance_decomposition  # noqa: E402
from kinase_open.splits import murcko_scaffold  # noqa: E402


def audit(df: pd.DataFrame) -> dict:
    y = df[TARGETS].to_numpy(dtype=float)
    n_labels = np.isfinite(y).sum(axis=1)

    report: dict = {
        "n_compounds": int(len(df)),
        "labels_per_target": {t: int(df[t].notna().sum()) for t in TARGETS},
        "compounds_by_label_count": {str(k): int((n_labels == k).sum()) for k in range(1, len(TARGETS) + 1)},
        "complete_panel": int((n_labels == len(TARGETS)).sum()),
        "duplicate_pairs": {},
        "variance_decomposition": variance_decomposition(y),
    }

    for a, b in itertools.combinations(TARGETS, 2):
        both = df[df[a].notna() & df[b].notna()]
        if not len(both):
            continue
        identical = int((both[a] == both[b]).sum())
        report["duplicate_pairs"][f"{a}/{b}"] = {
            "paired": int(len(both)),
            "identical": identical,
            "fraction": round(identical / len(both), 4),
            "median_abs_difference": round(float((both[a] - both[b]).abs().median()), 4),
        }

    scaffolds = [murcko_scaffold(s) for s in df["canonical_smiles"]]
    sizes = pd.Series(scaffolds).value_counts()
    report["scaffolds"] = {
        "unique": int(sizes.size),
        "singletons": int((sizes == 1).sum()),
        "largest_groups": [int(v) for v in sizes.head(8)],
    }
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "jak_pki.csv")
    report = audit(df)

    out = root / "results" / "data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{report['n_compounds']} compounds, {report['complete_panel']} with all four targets\n")
    print("labels per target:")
    for t, n in report["labels_per_target"].items():
        print(f"  {t:5s} {n:5d}")
    print("\nbyte-identical pKi across target pairs:")
    for pair, stats in sorted(report["duplicate_pairs"].items(), key=lambda kv: -kv[1]["fraction"]):
        print(
            f"  {pair:11s} {stats['identical']:4d}/{stats['paired']:4d} "
            f"= {100 * stats['fraction']:5.1f}%   median |diff| {stats['median_abs_difference']:.2f}"
        )
    vd = report["variance_decomposition"]
    if "selectivity_fraction" in vd:
        print(
            f"\nof the spread among complete-panel compounds, "
            f"{100 * vd['selectivity_fraction']:.1f}% is selectivity and the rest is shared potency"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
