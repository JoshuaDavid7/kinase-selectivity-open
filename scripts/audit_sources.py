"""Verify original raw-to-curated lineage and recover public assay metadata.

Metadata are retrieved now for the cached assay IDs; this does not establish the
original ChEMBL release, and is not used to change the benchmark after seeing it.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from benchmark import ROOT, TARGETS, digest, write_json
from kinase_open.data import BASE, curate

def get(session, endpoint, params=None):
    for attempt in range(4):
        try:
            response = session.get(f'{BASE}/{endpoint}', params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'data/source_snapshot'
    files = sorted(source.glob('*_Ki_raw.csv'))
    if len(files) != 4:
        raise RuntimeError('expected the four original raw snapshot CSVs')
    snapshot = json.loads((ROOT / 'data/snapshot.json').read_text())
    for path in files:
        if digest(path) != snapshot['files'][f'data/chembl_cache/{path.name}']['sha256']:
            raise AssertionError(f'original raw snapshot checksum mismatch: {path.name}')
    if digest(ROOT / 'data/jak_pki.csv') != snapshot['files']['data/jak_pki.csv']['sha256']:
        raise AssertionError('original curated snapshot checksum mismatch')
    raw = pd.concat([pd.read_csv(p).assign(target=p.name.split('_')[0]) for p in files], ignore_index=True)
    rebuilt, curation = curate(raw)
    existing = pd.read_csv(ROOT / 'data/jak_pki.csv')
    rebuilt = rebuilt.set_index('canonical_smiles').sort_index()
    existing = existing.set_index('canonical_smiles').sort_index()
    if not rebuilt.index.equals(existing.index):
        raise AssertionError('compound identities differ after offline rebuild')
    if not np.allclose(rebuilt[TARGETS].to_numpy(float), existing[TARGETS].to_numpy(float), rtol=0, atol=1e-7, equal_nan=True):
        raise AssertionError('labels differ after offline rebuild')
    metadata_path = source / 'assay_metadata.json'
    if args.fetch:
        if metadata_path.exists():
            raise RuntimeError('metadata snapshot exists; do not silently overwrite it')
        ids = sorted(raw.assay_chembl_id.dropna().unique())
        session = requests.Session()
        status = get(session, 'status.json')
        assays = []
        for offset in range(0, len(ids), 50):
            payload = get(session, 'assay.json', {'assay_chembl_id__in': ','.join(ids[offset:offset+50]), 'limit': 1000})
            assays.extend(payload['assays'])
            print(f'fetched assay metadata: {len(assays)}/{len(ids)}', flush=True)
        write_json(metadata_path, {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'api_base': BASE, 'current_api_status': status, 'requested_assay_ids': ids, 'assays': assays,
            'scope': 'current metadata for original cached assay IDs; original dataset release remains unknown'})
    metadata = json.loads(metadata_path.read_text())
    recorded = snapshot.get('recovered_assay_metadata')
    if recorded and digest(metadata_path) != recorded['sha256']:
        raise AssertionError('assay metadata checksum mismatch')
    assays = {a['assay_chembl_id']: a for a in metadata['assays']}
    unresolved = sorted(set(raw.assay_chembl_id.dropna()) - set(assays))
    if unresolved:
        raise RuntimeError(f'unresolved assay IDs: {unresolved}')
    numeric = pd.to_numeric(raw.pchembl_value, errors='coerce')
    keep = raw.canonical_smiles.notna() & raw.data_validity_comment.isna() & (raw.standard_relation == '=') & np.isfinite(numeric)
    observed = raw[keep].copy()
    observed['document'] = observed.assay_chembl_id.map(lambda a: assays[a].get('document_chembl_id'))
    observed['confidence'] = observed.assay_chembl_id.map(lambda a: assays[a].get('confidence_score'))
    documents = {}
    for (smiles, target), block in observed.groupby(['canonical_smiles', 'target']):
        documents[smiles, target] = set(block.document.dropna())
    report = {
        'raw_to_curated_labels_reproduced_atol': 1e-7, 'raw_to_curated_verified': True,
        'raw_rows': len(raw), 'curation': curation.to_dict(),
        'raw_snapshot_sha256': {p.name: digest(p) for p in files},
        'curated_sha256': digest(ROOT / 'data/jak_pki.csv'), 'assay_metadata_sha256': digest(metadata_path),
        'metadata_retrieved_at_utc': metadata['retrieved_at_utc'],
        'unique_cached_assays': len(assays),
        'assays_by_target_assignment_confidence': dict(Counter(str(a.get('confidence_score')) for a in assays.values())),
        'kept_raw_rows_by_confidence': {str(k): int(v) for k,v in observed.confidence.value_counts(dropna=False).items()},
        'unique_documents_among_kept_rows': int(observed.document.nunique()),
        'assays_with_variant_sequence_metadata': sum(bool(a.get('variant_sequence')) for a in assays.values()),
        'pairwise_shared_document_support': {},
        'limitations': ['posthoc provenance audit, not a new model evaluation',
            'shared document does not establish identical assay conditions or independent experiments',
            'the pooled median can include other documents as well',
            'a high target-assignment confidence score does not establish wild-type protein or comparable constructs'],
    }
    for a,b in itertools.combinations(TARGETS, 2):
        paired = existing[[a,b]].dropna()
        shared = sum(bool(documents.get((s,a),set()) & documents.get((s,b),set())) for s in paired.index)
        identical = paired[paired[a] == paired[b]]
        equal_shared = sum(bool(documents.get((s,a),set()) & documents.get((s,b),set())) for s in identical.index)
        report['pairwise_shared_document_support'][f'{a}/{b}'] = {'paired': len(paired),
            'shared_document_support': shared, 'identical_pooled_pKi': len(identical),
            'identical_with_shared_document_support': equal_shared}
    write_json(ROOT / 'results/source_audit.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
