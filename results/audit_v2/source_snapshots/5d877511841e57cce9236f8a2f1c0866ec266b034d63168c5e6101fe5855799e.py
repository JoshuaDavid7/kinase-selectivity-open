"""Reload the saved ensemble with its training-only feature/target scaling."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import MolecularNetwork
from .train import TARGETS, featurize_frame, predict
from .featurize import parse

def predict_from_checkpoints(checkpoint_dir: Path, smiles: list[str]):
    if not smiles or any(not s or parse(s) is None or parse(s).GetNumAtoms() == 0 for s in smiles):
        raise ValueError("provide nonempty, valid SMILES")
    paths = sorted(Path(checkpoint_dir).glob("member_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no ensemble checkpoints in {checkpoint_dir}")
    frame = pd.DataFrame({"canonical_smiles": smiles, **{t: np.nan for t in TARGETS}})
    data = featurize_frame(frame)
    predictions = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = MolecularNetwork(**{k: payload[k] for k in ["kind", "hidden", "depth", "dropout", "n_targets"]})
        model.load_state_dict(payload["state_dict"], strict=True)
        mean, std = payload["standardizer_mean"].numpy(), payload["standardizer_std"].numpy()
        predictions.append(predict(model, data, np.arange(len(data))) * std + mean)
    stacked = np.stack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0)
