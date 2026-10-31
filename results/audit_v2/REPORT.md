# Corrected retrospective benchmark

Protocol: audit_v2. One fixed scaffold partition; seeds vary initialization and batch order only.
The historical test partition was already known. These are corrected retrospective results, not new blinded validation.

## Affinity prediction

MAE is in pKi units. Macro-MAE weights the four targets equally. Values are mean +/- sample SD over initialization seeds, not confidence intervals.

| Model | Macro-MAE |
|---|---:|
| hybrid | 0.5644 +/- 0.0150 |
| fp | 0.5598 +/- 0.0050 |
| graph | 0.6812 +/- 0.0048 |
| train_mean | 1.1406 +/- 0.0000 |
| ridge | 0.6625 +/- 0.0000 |
| random_forest | 0.6120 +/- 0.0011 |
| extra_trees | 0.6922 +/- 0.0008 |
| knn_1 | 0.6933 +/- 0.0000 |
| knn_5 | 0.6371 +/- 0.0000 |
| knn_5_distance | 0.6329 +/- 0.0000 |
| ridge_descriptors | 0.9375 +/- 0.0000 |
| random_forest_descriptors | 0.5733 +/- 0.0013 |
| extra_trees_descriptors | 0.6181 +/- 0.0009 |

Trainable parameters per ensemble member: hybrid: 460,036, fp: 423,940, graph: 36,580. Three members are averaged per run.

All classical configurations are reported; choosing the smallest test error from this table is not an unbiased model-selection procedure.
Classical predictions are reused across architectures within a seed after input/hash checks. Run seconds include different amounts of reused work and are not a compute-efficiency comparison.

## Declared runs

| Seed | Architecture | Macro-MAE | Validation-selected comparator | Comparator macro-MAE | Seconds |
|---|---|---:|---|---:|---:|
| 1 | hybrid | 0.5706 | random_forest | 0.6115 | 631.7 |
| 1 | fp | 0.5576 | random_forest | 0.6115 | 61.9 |
| 1 | graph | 0.6863 | random_forest | 0.6115 | 252.6 |
| 2 | hybrid | 0.5473 | random_forest | 0.6133 | 687.9 |
| 2 | fp | 0.5655 | random_forest | 0.6133 | 256.0 |
| 2 | graph | 0.6769 | random_forest | 0.6133 | 347.7 |
| 3 | hybrid | 0.5752 | random_forest | 0.6113 | 813.0 |
| 3 | fp | 0.5564 | random_forest | 0.6113 | 49.7 |
| 3 | graph | 0.6803 | random_forest | 0.6113 | 334.5 |

## First declared hybrid run: paired differences

Neural MAE minus the validation-selected baseline MAE. Negative favors the neural model. Intervals resample paired errors by scaffold; they are exploratory and not corrected for multiple comparisons. An interval containing zero does not establish equivalence.

| Target | Difference | Exploratory 95% interval |
|---|---:|---|
| JAK1 | -0.0059 | [-0.0333, 0.0203] |
| JAK2 | -0.0477 | [-0.0811, -0.0132] |
| JAK3 | -0.0700 | [-0.1769, 0.0191] |
| TYK2 | -0.0402 | [-0.1115, 0.0295] |

## Uncertainty: hybrid across initialization seeds

Nominal coverage is 90%. Coverage under scaffold shift is empirical; no automatic exchangeability guarantee is claimed. Widths are evaluated on the same measured target rows as coverage.

| Target | Raw coverage range | Calibrated coverage range | Calibrated mean width |
|---|---|---|---:|
| JAK1 | 30.0-37.0% | 95.0-95.7% | 1.695 |
| JAK2 | 25.3-33.1% | 78.8-92.2% | 2.453 |
| JAK3 | 15.7-21.7% | 81.7-93.0% | 3.639 |
| TYK2 | 20.0-26.2% | 86.9-89.7% | 3.507 |

## Selectivity: first declared run

Partial panels compare the same observed competitors on both sides. Complete panels require all four measured targets. The constant comparator chooses its direction using training labels. Exact-zero margins count as non-positive here; pairwise direction metrics separately exclude exact ties.

| Panel | Target | n | Margin MAE | Direction accuracy | Training-majority baseline |
|---|---|---:|---:|---:|---:|
| measured_subset | JAK1 | 209 | 0.286 | 61.7% | 69.9% |
| measured_subset | JAK2 | 287 | 0.334 | 74.6% | 53.0% |
| measured_subset | JAK3 | 91 | 0.590 | 76.9% | 79.1% |
| measured_subset | TYK2 | 103 | 0.662 | 79.6% | 60.2% |
| complete_panel | JAK1 | 34 | 0.457 | 23.5% | 91.2% |
| complete_panel | JAK2 | 34 | 0.439 | 94.1% | 82.4% |
| complete_panel | JAK3 | 34 | 0.679 | 97.1% | 97.1% |
| complete_panel | TYK2 | 34 | 0.752 | 97.1% | 100.0% |

Pairwise metrics, exact-tie counts, every target's affinity results and all seeds are available in each run's `results.json`.

## Interpretation limits

This correction does not establish a neural advantage on new chemistry. Assays are heterogeneous and not source-matched; the chemical split is fixed and previously observed. Architecture comparisons also change parameter counts and are not compute-matched. No prospective biological validation was performed.

## Verify

```bash
python scripts/verify_results.py --study results/audit_v2
python scripts/render_results.py --check
```
