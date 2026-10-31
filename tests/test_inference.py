import numpy as np
import pandas as pd
import pytest
import torch
from kinase_open.inference import predict_from_checkpoints
from kinase_open.model import MolecularNetwork
from kinase_open.featurize import collate, mol_to_graph, parse
from kinase_open.train import featurize_frame, train_ensemble_multi

def test_released_checkpoint_reproduces_predictions(tmp_path):
    torch.set_num_threads(1)
    smiles = ["C", "CC", "CCC", "CO", "CCO", "CCCO", "CN", "CCN"]
    frame = pd.DataFrame({"canonical_smiles": smiles, **{t: np.arange(8.) + 5 for t in ["JAK1", "JAK2", "JAK3", "TYK2"]}})
    data = featurize_frame(frame)
    expected = train_ensemble_multi(data, np.arange(4), np.arange(4,6), [np.arange(6,8)],
        n_members=2, epochs=2, checkpoint_dir=tmp_path, base_seed=8)[0]
    mu, sigma = predict_from_checkpoints(tmp_path, smiles[6:])
    assert np.allclose(mu, expected.mu, atol=1e-6, rtol=0)
    assert np.allclose(sigma, expected.sigma, atol=1e-6, rtol=0)

def test_invalid_input_is_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError):
        predict_from_checkpoints(tmp_path, ["not a molecule"])

def test_graph_predictions_do_not_depend_on_batch_companions():
    torch.manual_seed(7)
    net = MolecularNetwork(kind="graph", dropout=0).eval()
    first = mol_to_graph(parse("CCO"))
    companion = mol_to_graph(parse("c1ccccc1"))
    with torch.no_grad():
        alone = net(collate([first]), None)[0]
        together = net(collate([first, companion]), None)[0]
    assert torch.allclose(alone, together, atol=1e-6, rtol=0)
