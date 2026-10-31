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
explicitly unknown. The committed CSV, rather than the current live API contents,
defines the corrected benchmark. Source assay heterogeneity remains a limitation.
