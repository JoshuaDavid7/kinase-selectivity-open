# Data attribution and snapshot

`jak_pki.csv` is a derived table of JAK-family Ki activities from
[ChEMBL, EMBL-EBI](https://www.ebi.ac.uk/chembl/).
ChEMBL data are distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
This derived data table is distributed under those terms; the repository's MIT
license applies to code, not to the underlying ChEMBL data.

Modifications: filter missing structures, validity-flagged, censored and missing
pChEMBL records; aggregate compound/target measurements by median; pivot to a
four-target table. Unmeasured affinities remain missing. See `data.py`.

`snapshot.json` records the original repository commit and file SHA256 hashes.
The original cache omitted release and retrieval timestamps; those fields are
explicitly unknown. The four original raw CSVs are archived byte-for-byte in
`source_snapshot/`. `python scripts/audit_sources.py` verifies their checksums and
rebuilds the curated labels offline, checking compound identities and missingness
as well as numerical agreement to 1e-7 pKi.

Metadata for all 292 cached assay IDs were subsequently retrieved from the
current ChEMBL 37 API on October 31, 2026 and archived in
`source_snapshot/assay_metadata.json`. This establishes the metadata retrieval
date/version, not the original dataset's release. There are 244 confidence-9
and 48 confidence-8 assays. Among 9,707 retained raw rows, 1,805 have confidence 8
(homologous rather than direct single-protein assignment); one assay explicitly
annotates a JAK2 V617F variant. The benchmark retains the original curated table
to evaluate the corrections transparently; it is not a purified wild-type human
assay panel. Shared-document support does not establish identical experimental
conditions or independence. See `results/source_audit.json`.

The committed CSV, rather than the current live API contents, defines the corrected
benchmark. Source and construct heterogeneity remain limitations.
