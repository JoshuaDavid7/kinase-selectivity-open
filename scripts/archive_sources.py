"""Archive exact model/evaluation source bytes so ZIP downloads are verifiable."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from benchmark import ROOT, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT / 'results/audit_v2')
    args = parser.parse_args()
    destination = args.study / 'source_snapshots'
    destination.mkdir(exist_ok=True)
    index = {}
    for provenance_path in sorted(args.study.glob('*/provenance.json')):
        p = json.loads(provenance_path.read_text())
        for name, sha in p['source_sha256'].items():
            if not (name.startswith('src/') or name in {'scripts/benchmark.py', 'scripts/run_experiment.py', 'scripts/replicate.py'}):
                continue
            target = destination / f'{sha}.py'
            if not target.exists():
                historical = subprocess.run(['git', 'show', f"{p['git_base_commit']}:{name}"], cwd=ROOT, capture_output=True)
                content = historical.stdout
                if historical.returncode != 0 or hashlib.sha256(content).hexdigest() != sha:
                    content = (ROOT / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != sha:
                    raise AssertionError(f'no exact source bytes available for {name}: {sha}')
                target.write_bytes(content)
            elif hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise AssertionError(f'corrupted source snapshot: {target}')
            index.setdefault(sha, {'files': [], 'runs': []})
            if name not in index[sha]['files']:
                index[sha]['files'].append(name)
            index[sha]['runs'].append(provenance_path.parent.name)
    write_json(destination / 'index.json', index)
    print(f'archived {len(index)} unique source files')

if __name__ == '__main__':
    main()
