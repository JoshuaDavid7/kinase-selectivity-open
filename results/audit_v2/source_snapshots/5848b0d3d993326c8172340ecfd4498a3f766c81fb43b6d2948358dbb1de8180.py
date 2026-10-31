"""Repeat the audited protocol over initialization seeds on one fixed split."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from benchmark import ROOT, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--kinds', nargs='+', choices=['hybrid', 'fp', 'graph'], default=['hybrid', 'fp', 'graph'])
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'audit_v2')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = args.out / 'plan.json'
    if plan.exists():
        raise SystemExit('Plan already exists. Use a new output directory; existing runs are never overwritten.')
    write_json(plan, {'protocol': 'audit_v2', 'split': 'fixed scaffold partition',
        'interpretation': 'initialization/batch-order replication, not new chemical splits',
        'test_already_seen_in_legacy_study': True,
        'settings_fixed_before_corrected_runs': True,
        'config': vars(args)})
    for seed in args.seeds:
        for kind in args.kinds:
            command = [sys.executable, str(ROOT / 'scripts' / 'benchmark.py'), '--seed', str(seed),
                '--kind', kind, '--members', str(args.members), '--epochs', str(args.epochs),
                '--threads', str(args.threads), '--out', str(args.out / f'scaffold_{kind}_seed{seed}')]
            if seed == args.seeds[0] and kind == 'hybrid':
                command.append('--save-models')
            print(f'Starting {kind}, seed {seed}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    print('All declared runs completed. Run scripts/verify_results.py and scripts/render_results.py.', flush=True)

if __name__ == '__main__':
    main()
