"""Verify a released ensemble reproduces its frozen predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import ROOT, digest, write_json
from kinase_open.inference import predict_from_checkpoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'results/audit_v2/scaffold_hybrid_seed1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    split = json.loads((args.run / 'split.json').read_text())
    indices = split['indices']['test'][:16]
    smiles = [split['smiles'][i] for i in indices]
    mu, sigma = predict_from_checkpoints(args.run / 'models', smiles)
    with np.load(args.run / 'predictions.npz', allow_pickle=False) as frozen:
        mu_error = float(np.max(np.abs(mu - frozen['test_mu'][:len(indices)])))
        sigma_error = float(np.max(np.abs(sigma - frozen['test_sigma'][:len(indices)])))
    tolerance = 1e-5
    if max(mu_error, sigma_error) > tolerance:
        raise AssertionError(f'checkpoint replay differs: mu {mu_error}, sigma {sigma_error}')
    result = {'n_compounds': len(indices), 'max_abs_mean_error': mu_error,
        'max_abs_std_error': sigma_error, 'tolerance': tolerance, 'passed': True,
        'checkpoints_sha256': {p.name: digest(p) for p in sorted((args.run / 'models').glob('*.pt'))}}
    write_json(args.run / 'checkpoint_validation.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
