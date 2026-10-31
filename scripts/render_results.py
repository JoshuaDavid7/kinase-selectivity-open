"""Generate the report from saved run JSON, with no hand-entered result values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark import ROOT, TARGETS

def report(study):
    plan = json.loads((study / "plan.json").read_text())
    seeds, kinds = plan["config"]["seeds"], plan["config"]["kinds"]
    runs = {(k, s): json.loads((study / f"scaffold_{k}_seed{s}" / "results.json").read_text()) for s in seeds for k in kinds}
    def macro(entry):
        return np.mean([entry[t]["mae"] for t in TARGETS])
    def mean_sd(values):
        return f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"
    lines = ["# Corrected retrospective benchmark", "",
        "Protocol: audit_v2. One fixed scaffold partition; seeds vary initialization and batch order only.",
        "The historical test partition was already known. These are corrected retrospective results, not new blinded validation.", "",
        "## Affinity prediction", "",
        "MAE is in pKi units. Macro-MAE weights the four targets equally. Values are mean +/- sample SD over initialization seeds, not confidence intervals.", "",
        "| Model | Macro-MAE |" , "|---|---:|"]
    for kind in kinds:
        lines.append(f"| {kind} | {mean_sd([macro(runs[kind,s]['per_target']) for s in seeds])} |")
    reference = kinds[0]
    baseline_names = runs[reference,seeds[0]]["baselines"].keys()
    for name in baseline_names:
        lines.append(f"| {name} | {mean_sd([macro(runs[reference,s]['baselines'][name]) for s in seeds])} |")
    lines += ["", "All classical configurations are reported; choosing the smallest test error from this table is not an unbiased model-selection procedure.", "",
        "## Declared runs", "", "| Seed | Architecture | Macro-MAE | Validation-selected comparator | Comparator macro-MAE | Seconds |", "|---|---|---:|---|---:|---:|"]
    for seed in seeds:
        for kind in kinds:
            r = runs[kind,seed]
            selected = r["selected_baseline_by_validation"]
            lines.append(f"| {seed} | {kind} | {macro(r['per_target']):.4f} | {selected} | {macro(r['baselines'][selected]):.4f} | {r['runtime_seconds']:.1f} |")
    first = runs[reference,seeds[0]]
    lines += ["", f"## First declared {reference} run: paired differences", "",
        "Neural MAE minus the validation-selected baseline MAE. Negative favors the neural model. Intervals resample paired errors by scaffold; they are exploratory and not corrected for multiple comparisons. An interval containing zero does not establish equivalence.", "",
        "| Target | Difference | Exploratory 95% interval |", "|---|---:|---|" ]
    for t in TARGETS:
        r = first["paired_comparison"][t]
        lo, hi = r["exploratory_ci95"]
        lines.append(f"| {t} | {r['neural_minus_selected_baseline_mae']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ["", f"## Uncertainty: {reference} across initialization seeds", "",
        "Nominal coverage is 90%. Coverage under scaffold shift is empirical; no automatic exchangeability guarantee is claimed. Widths are evaluated on the same measured target rows as coverage.", "",
        "| Target | Raw coverage range | Calibrated coverage range | Calibrated mean width |", "|---|---|---|---:|"]
    for t in TARGETS:
        entries = [runs[reference,s]["uncertainty"][t] for s in seeds]
        raw = [e["raw_ensemble_coverage"] for e in entries]
        coverage = [e["conformal_coverage"] for e in entries]
        lines.append(f"| {t} | {100*min(raw):.1f}-{100*max(raw):.1f}% | {100*min(coverage):.1f}-{100*max(coverage):.1f}% | {np.mean([e['conformal_width_on_measured'] for e in entries]):.3f} |")
    lines += ["", "## Selectivity: first declared run", "",
        "Partial panels compare the same observed competitors on both sides. Complete panels require all four measured targets. The constant comparator chooses its direction using training labels. Exact-zero margins count as non-positive here; pairwise direction metrics separately exclude exact ties.", "",
        "| Panel | Target | n | Margin MAE | Direction accuracy | Training-majority baseline |", "|---|---|---:|---:|---:|---:|"]
    for panel in ["measured_subset", "complete_panel"]:
        for t in TARGETS:
            e = first["selectivity"][panel][t]
            lines.append(f"| {panel} | {t} | {e['n']} | {e['mae']:.3f} | {100*e['preference_accuracy']:.1f}% | {100*e['training_majority_baseline_accuracy']:.1f}% |")
    lines += ["", "Pairwise metrics, exact-tie counts, every target's affinity results and all seeds are available in each run's `results.json`.", "",
        "## Interpretation limits", "",
        "This correction does not establish a neural advantage on new chemistry. Assays are heterogeneous and not source-matched; the chemical split is fixed and previously observed. Architecture comparisons also change parameter counts and are not compute-matched. No prospective biological validation was performed.", "",
        "## Verify", "", "```bash", f"python scripts/verify_results.py --study {study.relative_to(ROOT)}", "python scripts/render_results.py --check", "```", ""]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "results" / "audit_v2")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = report(args.study.resolve())
    path = args.study / "REPORT.md"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("report is missing or differs from saved results")
        print("report matches saved results")
    else:
        path.write_text(content)
        print(f"wrote {path}")

if __name__ == "__main__":
    main()
