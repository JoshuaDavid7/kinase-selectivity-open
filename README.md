# JAK affinity and selectivity benchmark

A ligand-based molecular ML project on public ChEMBL Ki measurements for
JAK1, JAK2, JAK3 and TYK2. A custom PyTorch directed message-passing network
is evaluated beside fingerprint neural models and classical baselines.
The implementation was developed with AI coding assistance.

The current version reports an audited, corrected benchmark. Changes to the
original evaluation and reporting are documented in [the audit](docs/AUDIT.md).
Historical JSON is retained in `results/legacy/`; it must not be interpreted as
output from the corrected code. There is no claim that the neural models beat
classical methods, or that this is a new molecular architecture.

## Reproduce

Tested on Python 3.11, CPU PyTorch 2.2.2 and NumPy 1.26.4.

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/replicate.py --seeds 1 2 3 --members 3 --epochs 40
python scripts/audit_sources.py
python scripts/verify_results.py
python scripts/render_results.py
```

Existing runs are never overwritten. Use `--out results/my-study` for a new
replication. Exact tested package versions are recorded in every run's provenance.
The committed curated CSV defines this benchmark; a live ChEMBL rebuild is not
expected to be identical. Data attribution and hashes are in [data/](data/README.md).

## Corrected study

- 3,621 compounds, 6,698 observed compound-target labels; 338 complete four-target panels.
- Byte-for-byte archives of the four raw input CSVs, an offline raw-to-label check,
  and recovered metadata for all 292 source assays across 101 documents.
- Fixed train / validation / calibration / test scaffold partition. Seeds change
  initialization and batch order, not chemical membership.
- Three architectures: directed graph, Morgan bits plus twelve descriptors, and
  their hybrid. Descriptors and target standardization are fitted on training data.
- Ten fixed classical comparators, including Morgan-only and descriptor-augmented
  trees/ridge, Jaccard neighbors on bits only, and a training-mean predictor.
- The comparator is selected by validation macro-MAE. Paired scaffold-bootstrap
  intervals describe test error differences; no equivalence or multiplicity-adjusted
  significance claim is made.
- Partial-panel selectivity compares identical measured competitor sets. Full-panel
  and pairwise results are separate. Training-majority baselines and exact ties are explicit.
- Normalized conformal scores use a floor fixed on validation data and the exact
  finite-sample order statistic. Scaffold-shift coverage is measured, not guaranteed.
- Predictions, split membership, calibration settings and comparator selection are
  frozen before test scoring. Data/source hashes and package versions are saved.

The corrected study reuses a test partition already seen in the original work.
It is retrospective validation of corrections, not a new independently blinded test.
See [the generated report](results/audit_v2/REPORT.md) for every declared run.

## Reload a trained ensemble

The corrected release includes the first declared hybrid run's three checkpoints.
Extract them into that run's `models/` directory, then:

```bash
python scripts/validate_checkpoint.py
python scripts/predict.py --run results/audit_v2/scaffold_hybrid_seed1 --smiles 'CCO' --out example.json
```

The example is a CLI smoke test, not a recommended ligand. Inference reports pKi,
ensemble spread, empirical conformal intervals and nearest-training fingerprint
similarity. Similarity is descriptive and does not guarantee prediction quality.
Use only trusted model artifacts and verify their release checksums.

## What this demonstrates

Directed graph implementation, sparse-label multitask training, chemical split
handling, controlled ablations, uncertainty evaluation and independently
recomputable metrics. The audit and regression tests are part of the project.

## Scope

Pooled historical assays are heterogeneous. The source audit found confidence-8
homologous target assignments in 1,805 of 9,707 retained raw rows and one annotated
JAK2 V617F assay. The benchmark preserves the historical table and is not a pure
wild-type human panel. Shared-document support does not establish identical
conditions or explain identical cross-target values. No consistent salt/tautomer processing
or explicit stereochemical features were added. Partial-panel results cannot
establish whole-family selectivity. There is no prospective assay, docking,
ADME, generative molecule design, or wet-lab validation.

A stronger next study needs assay/document-aware provenance and chemical or
source partitions reserved before model selection. See [the research plan](docs/RESEARCH_PLAN.md).

## License

Code: MIT. Derived ChEMBL data: CC BY-SA 3.0, attributed separately in `data/README.md`.
