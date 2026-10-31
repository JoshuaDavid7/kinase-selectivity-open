"""Check every number in the README against the results it claims to report.

A README drifts the moment a rerun changes a third decimal place, and a stale
number in the one file people actually read is worse than no number at all.
This parses the tables back out and compares them to the JSON.

It fails loudly on a mismatch *and* on finding nothing to check, because a
checker that silently parses zero rows is indistinguishable from a passing one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["JAK1", "JAK2", "JAK3", "TYK2"]
TOL = 0.0006  # the README prints three decimals; allow half a unit in the last


def cell(text: str) -> float | None:
    """First number in a markdown cell, ignoring bold markers and unicode minus."""
    match = re.search(r"-?\d+\.?\d*", text.replace("−", "-").replace("**", ""))
    return float(match.group()) if match else None


def rows(readme: str, header_fragment: str) -> list[list[str]]:
    """Table rows whose first cell is a target name, after a given heading."""
    start = readme.find(header_fragment)
    if start < 0:
        return []
    out = []
    for line in readme[start:].splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0].replace("**", "") in TARGETS:
            out.append(cells)
    return out


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    results_path = ROOT / "results" / "scaffold_hybrid_seed1.json"
    audit_path = ROOT / "results" / "data_audit.json"
    if not results_path.exists():
        print(f"missing {results_path}; run scripts/run_experiment.py first")
        return 1

    results = json.loads(results_path.read_text())
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    failures: list[str] = []
    checked = 0

    def compare(label: str, claimed: float | None, actual: float, scale: float = 1.0, text: str = "") -> None:
        """Compare at the precision the README actually prints.

        A README showing 39.3 is not wrong because the JSON holds 39.2739 - it
        is correctly rounded. The tolerance is therefore half a unit in the
        last displayed decimal, inferred from the printed string, so the check
        catches a stale number without flagging an honest round.
        """
        nonlocal checked
        if claimed is None:
            failures.append(f"{label}: could not parse a number from the README")
            return
        checked += 1
        printed = re.search(r"-?\d+\.(\d+)", text.replace("−", "-").replace("**", ""))
        decimals = len(printed.group(1)) if printed else 0
        tolerance = 0.5 * 10**-decimals + 1e-9
        if abs(claimed - actual * scale) > tolerance:
            failures.append(
                f"{label}: README says {claimed}, results say {actual * scale:.4f} "
                f"(tolerance {tolerance:g} at {decimals} dp)"
            )

    for cells in rows(readme, "### Accuracy, and the baseline that matters"):
        name = cells[0].replace("**", "")
        actual = results["per_target"][name]
        compare(f"{name} test n", cell(cells[1]), actual["n"], text=cells[1])
        compare(f"{name} MAE", cell(cells[2]), actual["mae"], text=cells[2])
        lo, hi = re.findall(r"\d+\.\d+", cells[2])[1:3]
        compare(f"{name} CI low", float(lo), actual["mae_ci95"][0], text=lo)
        compare(f"{name} CI high", float(hi), actual["mae_ci95"][1], text=hi)
        compare(f"{name} R2", cell(cells[3]), actual["r2"], text=cells[3])
        compare(f"{name} rho", cell(cells[4]), actual["spearman"], text=cells[4])

    for cells in rows(readme, "### Uncertainty"):
        name = cells[0].replace("**", "")
        actual = results["uncertainty"][name]
        compare(f"{name} raw coverage", cell(cells[1]), actual["raw_ensemble_coverage"], 100, text=cells[1])
        compare(f"{name} raw width", cell(cells[2]), actual["raw_ensemble_width"], text=cells[2])
        compare(f"{name} conformal coverage", cell(cells[3]), actual["conformal_coverage"], 100, text=cells[3])
        compare(f"{name} conformal width", cell(cells[4]), actual["conformal_width"], text=cells[4])

    for cells in rows(readme, "### Selectivity"):
        name = cells[0].replace("**", "")
        actual = results["selectivity"][name]
        compare(f"{name} margin n", cell(cells[1]), actual["margin_n"], text=cells[1])
        compare(f"{name} margin MAE", cell(cells[2]), actual["margin_mae"], text=cells[2])
        compare(f"{name} preference", cell(cells[3]), actual["preference_accuracy"], 100, text=cells[3])
        compare(f"{name} majority", cell(cells[4]), actual["majority_rate"], 100, text=cells[4])
        compare(f"{name} gain", cell(cells[5]), actual["preference_accuracy"] - actual["majority_rate"], 100, text=cells[5])

    if audit:
        for pair, stats in audit["duplicate_pairs"].items():
            a, b = pair.split("/")
            pattern = rf"\|\s*{a}\s*/\s*{b}\s*\|\s*([\d,]+)\s*/\s*([\d,]+)\s*\|"
            found = re.search(pattern, readme)
            if found:
                compare(f"{pair} identical", float(found.group(1).replace(",", "")), stats["identical"], text="0")
                compare(f"{pair} paired", float(found.group(2).replace(",", "")), stats["paired"], text="0")

        for claim, actual in [
            (rf"{audit['complete_panel']}\s+carry all four labels", audit["complete_panel"]),
            (rf"{audit['n_compounds']:,}\s+compounds", audit["n_compounds"]),
        ]:
            checked += 1
            if not re.search(claim, readme):
                failures.append(f"README does not state the audited value {actual}")

    if checked == 0:
        print("FAIL: parsed zero values - the README tables did not match the expected shape")
        return 1

    for line in failures:
        print(f"  {line}")
    print(f"\n{checked} values checked, {len(failures)} mismatched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
