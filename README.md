# Kinase selectivity from public data

Predicting binding affinity across the four JAK-family kinases — **JAK1, JAK2, JAK3, TYK2** —
and asking a harder question than "is this compound potent": *can a model tell which member of a
closely related family a compound prefers?*

Everything here runs from public ChEMBL data and rebuilds from a clean checkout. The model is a
directed message-passing network written from the algorithm — no chemprop, no torch-geometric,
no DGL — fused with a fingerprint branch, trained multitask, and wrapped in conformal prediction
so a number handed to a chemist arrives with an error bar that means something.

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
python scripts/build_dataset.py     # pulls ChEMBL, writes data/jak_pki.csv
python scripts/audit_data.py        # check the data before trusting a model on it
python scripts/run_experiment.py --split scaffold --members 5 --seed 1
pytest
```

## The dataset, and what is wrong with it

13,680 raw Ki activities across the four targets reduce to **3,621 compounds**, of which **338
carry all four labels**. Curation rules are stated in `data.py` and each one reports what it
discarded: a measurement needs a structure and a ChEMBL pChEMBL value, ChEMBL's own validity flag
is trusted, censored values (`>`, `<`) are dropped rather than imputed at the bound, and replicate
measurements of the same compound–target pair collapse to their median.

| Target | Labelled compounds |
|---|---|
| JAK1 | 2,303 |
| JAK2 | 2,356 |
| JAK3 | 1,088 |
| TYK2 | 951 |

**The defect worth knowing about before reading any score.** For each pair of targets, some
fraction of compounds carry *byte-identical* pKi values:

| Pair | Identical / paired | | Median \|difference\| |
|---|---|---|---|
| JAK3 / TYK2 | 261 / 523 | **49.9%** | 0.04 |
| JAK1 / TYK2 | 138 / 429 | 32.2% | 1.01 |
| JAK2 / TYK2 | 164 / 648 | 25.3% | 1.04 |
| JAK1 / JAK3 | 77 / 442 | 17.4% | 1.96 |
| JAK1 / JAK2 | 250 / 1,686 | 14.8% | 0.27 |
| JAK2 / JAK3 | 76 / 517 | 14.7% | 0.96 |

Paralogs do agree often, but not to the last decimal place. When they do, the two numbers
generally came from one reported figure covering a panel rather than two independent assays.
**JAK3/TYK2 is the case to be careful with**: half the paired values are identical and the median
absolute difference across all of them is 0.04, so a model that appears to distinguish that pair
may only be copying. Any JAK3/TYK2 result here should be read with that in mind.

**How much of this is even selectivity?** Among the 338 complete-panel compounds, decomposing the
spread into variation *within* a compound's four values (its selectivity profile) and *between*
compound means (overall potency) puts **52.4% in selectivity**. Roughly half the signal is about
which kinase, and half is about how potent — so the selectivity question is a real one on this
data rather than potency wearing a disguise.

## Design decisions, and why

**The message passing is directed, and the reverse bond is excluded.** A message on bond `u→v`
aggregates every bond entering `u` *except* `v→u`. Without that exclusion, information bounces
back and forth along a single bond and network depth stops meaning anything. It is also the
easiest part of a D-MPNN to get silently wrong, so `tests/test_core.py` checks that the reverse
index is a true involution and that it swaps endpoints.

**Missing labels are absent, not zero.** Most compounds are measured against one or two targets.
The loss is masked, so an unmeasured target contributes nothing — treating it as a zero would
teach the model that an unmeasured compound is inactive, which is the opposite of the truth.

**The random split is kept on purpose.** Not as a baseline to beat, but as an *interpolation
comparator*. Analogues of one scaffold land on both sides of a random split, so the model is
mostly asked to interpolate within series it already knows. Reporting both says how much of a
headline number came from the split rather than the model. On this data the scaffold split moves
the test set genuinely further out: median nearest-training Tanimoto 0.772, against 0.833 random.

**Uncertainty is calibrated, not assumed.** A deep ensemble's standard deviation is a useful
*relative* signal — it says which predictions are shakier — but its absolute width carries no
guarantee and in practice is far too narrow. Split conformal fixes the width on a calibration set
that was used neither for training nor for early stopping, and makes one distribution-free
promise: on exchangeable data the interval covers the truth at least `1 - α` of the time. The
quantile uses the finite-sample correction `⌈(n+1)(1-α)⌉/n`, which is what makes that hold at the
sample sizes here rather than only asymptotically.

**The conformal score has a floor on σ.** With a purely normalised score `|y - μ| / σ`, any
calibration point whose ensemble happens to agree produces an enormous score, that score lands in
the upper quantile, and *every* interval inflates. The floor is set from a low percentile of the
calibration spread itself, so it adapts to the ensemble rather than being a magic constant.

**Error bars resample scaffold groups, not rows.** Fifty analogues of one series are not fifty
independent observations. Bootstrapping rows produces a confidence interval far too narrow to be
honest; resampling whole groups keeps the correlation intact.

**Selectivity is reported against the majority rate.** A one-vs-rest margin is
`pKi(target) − max(others)`, and it is only defined where the target *and* at least one other are
measured. Sign accuracy is meaningless without saying how often a model that always guesses the
majority class would be right, so both are printed together.

## Results

See [`results/`](results/). `run_experiment.py` writes one JSON per split design and seed,
covering per-target regression metrics with scaffold-group bootstrap intervals, raw-versus-
conformal coverage at a nominal 90%, selectivity margins against the majority rate, and seven
classical baselines fitted per target.

## What this does not do

No docking, no structure-based scoring, no 3D conformers — this is a ligand-based model over 2D
structure. No generative design. Affinity is modelled, not ADME, not selectivity against the
wider kinome, and not anything that would make a compound a drug. The dataset is an aggregation
of heterogeneous assays from many papers, and the duplicate-label finding above is a direct
consequence of that; a single-source panel would be better and is not publicly available at this
scale.

## Layout

```
src/kinase_open/
  data.py        ChEMBL pull, curation rules, and what each rule discarded
  featurize.py   Morgan fingerprints + descriptors; molecules as directed bond graphs
  model.py       D-MPNN encoder, fingerprint branch, masked multitask loss
  splits.py      scaffold split, random comparator, Butina clusters, similarity
  conformal.py   split conformal with a variance floor
  train.py       deep ensemble, per-target standardisation
  evaluate.py    metrics, selectivity margins, baselines, group bootstrap
scripts/         build_dataset · audit_data · run_experiment
tests/           30 tests, aimed at the failures that return a number instead of an error
```

## Licence

MIT. ChEMBL data is used under its own terms (CC BY-SA 3.0).
