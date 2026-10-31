# Next scientific study

The corrected benchmark establishes an inspectable implementation, not a
validated biological discovery or a general neural advantage. The following
stages have explicit stopping conditions; they are not claims of completed work.

1. **Recover measurement provenance.** Retrieve versioned ChEMBL activity,
   document and assay metadata. Record target-assignment confidence, organism,
   construct/mutation information when available, measurement type, relation,
   units and replicate dispersion. Keep an immutable source snapshot. Audit
   exact cross-target values against those sources without assuming copying.
   Stop treating selectivity as an assay-independent label if conditions cannot
   be reconciled. Define matched-assay/paper panels separately from pooled medians.
2. **Reserve a genuinely new evaluation.** Create multiple chemical/source
   partitions and, where metadata supports it, a document or temporal holdout.
   Keep final labels inaccessible to model selection. Document near-neighbor
   similarity, label missingness and target coverage per partition. A new seed
   on the historical fixed partition does not fulfill this requirement.
3. **Test one useful hypothesis.** Ask whether shared graph representations
   improve sparse-target or low-data prediction over a comparably conditioned
   fingerprint network and tuned classical models. Predeclare data budgets,
   configurations, primary metric, paired comparisons and all planned runs.
   Include parameter counts and training time. Count a result as useful only
   if it survives chemical/source partitions and has a practically relevant
   effect, rather than just a favorable single split.
4. **Evaluate uncertainty as a decision tool.** Compare normalized and absolute
   residual conformal scores using validation-selected settings, plus uncertainty
   from a strong classical model. Report coverage and width together, stratified
   by target, structural distance and source. Test abstention/measurement
   prioritization at a fixed budget. Do not interpret empirical calibration as a
   guarantee under arbitrary distribution shift.
5. **Release evidence.** Save every declared run, trained artifacts, raw/derived
   provenance, frozen predictions, a results verifier and limitations. Keep an
   unfavorable result rather than changing the test until a neural method wins.

This work strengthens molecular prediction and research engineering evidence.
For generative-biology/post-training roles, a separate sequence-model study must
show controlled feedback learning and useful generalization; ligand prediction
does not substitute for that experience. Large-cluster training and founding-team
leadership likewise require their own evidence.
