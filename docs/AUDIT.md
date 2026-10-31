# Adversarial review, 2026-10-31

Reviewed baseline: `9759244a4ad4a41fd82adfb04f5dd1752bab5805`.
The implementation was developed with AI coding assistance. This review examines
the actual code and saved results rather than treating the original narrative as evidence.

## Findings and corrections

| Priority | Finding | Correction |
|---|---|---|
| High | Observed one-vs-rest margins used only measured competitors, but predicted margins used all three competitors. The two sides sometimes represented different endpoints. | Mask predicted competitors to the same observed panel. Report partial-panel, complete-panel and pairwise results separately. Do not reuse the old selectivity claims. |
| High | The best classical comparator was chosen separately for each target using test MAE. A baseline point inside a neural model's confidence interval was called a statistical tie, and a point outside was treated as superiority. | Choose the comparator by validation macro-MAE. Bootstrap paired error differences by scaffold group. Intervals are exploratory, with no multiplicity adjustment; overlap with zero does not establish equivalence. |
| High | The normalized conformal score used a floor estimated from calibration examples. The ordinary fixed-score guarantee does not directly justify that procedure. | Choose the floor on validation predictions before calibration. Explicitly distinguish a valid fixed-score construction under exchangeability from empirical coverage under scaffold shift. |
| High | The conformal quantile used NumPy's `higher` interpolation at a corrected level. This can select one order statistic too high. For tiny calibration sets, capping at the maximum score replaces the required infinite endpoint. | Use the exact one-based rank `ceil((n+1)(1-alpha))`; return infinity if the rank exceeds n. Include infinite intervals in coverage and width reporting. |
| Medium | Jaccard neighbors received Morgan bits concatenated with continuous descriptors; the distance treated nonzero descriptor values as binary features. | Use Morgan bits only for Jaccard. Keep explicit descriptor-augmented ridge/tree baselines as well, so fixing neighbors does not weaken the comparator suite. |
| Medium | Unscaled descriptors, including molecular weight, entered the neural fingerprint branch alongside binary bits. The 7.5% historical hybrid-vs-FP improvement therefore did not isolate the benefit of adding a graph to a well-conditioned FP branch. | Fit descriptor scaling using training molecules only. Store it in model buffers. Rerun all three architectures under the corrected protocol. |
| Medium | Scaffold splitting permuted then sorted each partition, leaving membership unchanged across seeds. The repetition was described too broadly as reproducibility of the architecture finding. | State that the chemical partition is fixed. Seeds replicate initialization and batch order only. Three seeds cannot establish performance on new chemical partitions. |
| Medium | Equality of recorded pKi values was used to assert copied/panel-derived measurements without source-level evidence. | Retain the equality count as a descriptive audit. Remove the proposed provenance and causal explanations. |
| Medium | The majority rate was calculated from test labels. It is useful descriptively, but it is an oracle comparison if described as a fitted constant classifier. | Also report the test accuracy of a constant direction selected from training labels. Separate exact ties in pairwise direction evaluation. |
| Medium | Aggregate JSON lacked predictions, split manifests, environment versions and model checkpoints. The original README checker verified transcription, not scientific validity. | Save predictions, split indices, source/data hashes, versions, a validation-selected comparator and calibration settings before test scoring; add a metric verifier and a first-seed hybrid checkpoint. |
| Low | The masked loss could propagate NaNs if given actual NaN targets, although the existing loader replaced them. Numeric curation admitted infinite pChEMBL values. | Index observed entries before computing squared errors; reject non-finite numeric labels. Add regression cases. |

## What survives the review

The directed message-passing implementation excludes reverse edges as intended.
The multitask training loader masks unmeasured labels. Target scaling uses training
labels only. Calibration and test predictions come from the same fitted ensemble.
The scaffold protocol keeps train, validation, calibration and test scaffolds disjoint.
These are useful implementation choices; they do not establish a new scientific method.

## Historical results

The old JSON files are preserved under `results/legacy/`. Affinity MAEs were not
shown to be arithmetically false by this review. However, their comparative
interpretation, the selectivity evaluation and the uncertainty construction have
the limitations above. Old selectivity and coverage figures must not be presented
as results from the corrected code. The historical 7.5% gain was against an
unscaled fingerprint neural model on one fixed chemical partition, not against
classical baselines or across independent chemical splits.

## Corrected study boundary

The corrected study fixes three architectures, three initialization seeds, three
members per ensemble and a maximum of 40 epochs before running them. It reuses the
historical chemical split. Test labels and old results were already public and
known; this is a retrospective correction, not a new blinded validation.
No winner-dependent architecture search or test-driven hyperparameter sweep is
part of this rerun. All declared runs must be reported, including unfavorable ones.

The frozen-artifact step is an auditable software boundary. Because the dataset
is public and loaded in one process, it is not independent external blinding.

## Remaining scientific limitations

- Assay confidence, assay conditions, protein constructs and document identifiers
  were not all retained in the original raw cache. The median pKi table pools
  heterogeneous experiments. Equal values alone do not establish label copying.
- The original pull did not record a ChEMBL release or retrieval timestamp. The
  committed curated CSV and its SHA256 define this study; a future live API pull
  may change the data. Full raw caches are not redistributed by this repository.
- No consistent salt/tautomer normalization or stereochemical representation was
  added in this correction. The graph features and default Morgan fingerprints
  do not explicitly encode stereochemistry. This limits enantiomer-specific claims.
- One deterministic scaffold partition is not a temporal or prospective assay
  test. Disjoint scaffolds can still contain close structural analogues.
- Partial-panel selectivity is conditional on which targets were measured. It
  cannot establish full JAK-family or wider-kinome selectivity.
- A larger within-compound variance fraction does not by itself establish useful
  selectivity learning: target offsets and assay differences can contribute.
- Conformal intervals have no automatic marginal or conditional coverage promise
  on this deterministic scaffold split. Wider intervals alone do not prove utility.
- The baseline configurations are declared and fixed, not exhaustively tuned.
  Uncertainty comparisons with classical models have not been established.
- There is no prospective synthesis, wet-lab validation, ADME, docking or clinical
  claim. These results support retrospective ligand-based modeling only.

## References

- [Angelopoulos and Bates, conformal prediction tutorial](https://arxiv.org/abs/2107.07511)
- [ChEMBL assay and activity documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)
- [scikit-learn pairwise distance documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

## Before advertising the project

Use the corrected report and its explicit scope. Do not claim a new architecture,
superiority over classical models, guaranteed scaffold-shift calibration, a
validated drug candidate, or independent wet-lab work. Be ready to reproduce a run
and explain the masked loss, directed graph update, matched-panel evaluation,
validation selection, conformal rank and paired comparison without assistance.
