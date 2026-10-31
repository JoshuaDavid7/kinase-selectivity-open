# Explaining this project

Implementation used substantial AI coding assistance. The useful ownership test
is whether you can reproduce, explain and change the work yourself. Read the
audit before using historical numbers. Do not describe a standard D-MPNN as an
invented architecture or describe the old 7.5% ablation as a classical-model win.

## A concise description

"I built an open benchmark for JAK-family affinity prediction using public
ChEMBL data. It compares a graph/fingerprint multitask model with classical
baselines and examines uncertainty and selectivity under a scaffold split.
An adversarial review exposed several evaluation problems, which are documented
and covered by regression tests. The corrected results are reproducible from
saved predictions. It is retrospective modeling, not prospective drug discovery."

## Walk through the implementation

1. In `featurize.py`, find the 2,048 binary Morgan bits, twelve descriptors,
   atom/bond vocabularies and reverse-edge index. Explain why the current
   features cannot establish enantiomer-specific binding predictions.
2. Trace a directed bond update in `model.py`: sum incoming messages at its
   source atom, subtract the reverse-edge message, update with the initial
   representation, and pool atom states into a molecule embedding. Explain
   the difference between graph, FP and hybrid branches.
3. Explain missing target labels. They are absent from the loss, not inactive
   compounds. Demonstrate the masked-NaN gradient regression test.
4. Explain the descriptor and target scaling. Both are fitted using training
   data, and descriptor scaling is stored in the checkpoint for inference.
5. Inspect `split.json`. Identify which scaffold groups appear in each part.
   Show why repeating with three initialization seeds does not create three
   chemical generalization tests.
6. Show baseline selection using validation macro-MAE. Explain why taking the
   smallest test MAE across ten models is optimistic model selection.
7. Derive the normalized conformal score and exact order-statistic rank.
   Explain why its floor must be fixed independently of calibration, why tiny
   calibration sets can require infinite intervals, and why this scaffold
   protocol has no automatic exchangeability guarantee.
8. Work through the partial-panel regression case: true values `[8,7,missing,missing]`
   and predicted values `[8,7,12,15]`. The measured JAK1-vs-JAK2 margin is +1 on
   both sides. Predicting a margin against all four kinases answers a different
   question and cannot be scored against that label.
9. Explain paired scaffold-bootstrap differences. Zero inside an interval is
   not proof of equivalence. These intervals are exploratory and are not
   corrected for testing many targets/comparisons.
10. Run `verify_results.py`; then change one byte in a temporary copy of a
    prediction artifact and show that the checksum check rejects it.

## What to say about the results

Use `results/audit_v2/REPORT.md` and its complete run table. Distinguish point
estimates, initialization variability and chemical-partition uncertainty.
State the strongest baseline's result directly, whether or not the neural model
wins. Do not attribute exact-label equality to copied assays without provenance.
The current study does not establish that uncertainty improves a chemist's
decisions or that a prediction corresponds to a useful drug candidate.

## Practice a substantive change

Without AI assistance, add an absolute-residual conformal comparator, choosing
all settings on training/validation data. Test the exact quantile, compute
coverage and width from frozen predictions, and explain what the comparison
would and would not show under scaffold shift. Treat this as an interview
exercise, not a promised improvement on the existing test set.
