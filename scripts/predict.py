"""Predict affinities with a trusted released ensemble; no retraining required."""
import argparse
import json
from pathlib import Path
import torch
from benchmark import ROOT, TARGETS, conformal, write_json
from kinase_open.inference import predict_from_checkpoints
from kinase_open.splits import nearest_train_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    mu, sigma = predict_from_checkpoints(args.run / "models", args.smiles)
    frozen = json.loads((args.run / "frozen.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    train_smiles = [split["smiles"][i] for i in split["indices"]["train"]]
    similarities = nearest_train_similarity(train_smiles, args.smiles)
    rows = []
    for i, smi in enumerate(args.smiles):
        entry = {"smiles": smi, "max_training_morgan_tanimoto": float(similarities[i]), "targets": {}}
        for t, name in enumerate(TARGETS):
            result = {"predicted_pKi": float(mu[i,t]), "ensemble_std": float(sigma[i,t])}
            if name in frozen["calibrators"]:
                values = dict(frozen["calibrators"][name])
                if values["q"] is None:
                    values["q"] = float("inf")
                cal = conformal.ConformalCalibrator(**values)
                lo, hi = cal.interval(mu[i:i+1,t], sigma[i:i+1,t])
                result |= {"nominal_90_interval": [float(lo[0]), float(hi[0])], "unbounded": not torch.isfinite(torch.tensor(cal.q)).item()}
            entry["targets"][name] = result
        rows.append(entry)
    write_json(args.out, {"predictions": rows,
        "scope": "retrospective ligand-based affinity model; no prospective or whole-kinome validation",
        "interval_caveat": "empirical calibration, no automatic coverage guarantee on new chemistry",
        "similarity_caveat": "fingerprint similarity describes proximity to training data; it is not a reliability guarantee",
        "null_interval_endpoint_means_unbounded": True})
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
