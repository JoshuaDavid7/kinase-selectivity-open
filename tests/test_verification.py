import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import ROOT
from verify_results import same, verify_run

def test_verifier_rejects_tampered_prediction_file_before_scoring(tmp_path):
    data_hash = hashlib.sha256((ROOT / "data/jak_pki.csv").read_bytes()).hexdigest()
    split = tmp_path / "split.json"
    split.write_text('{}')
    pred = tmp_path / "predictions.npz"
    pred.write_bytes(b'original artifact')
    frozen = {"predictions_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(),
              "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest()}
    (tmp_path / "provenance.json").write_text(json.dumps({"data_sha256": data_hash}))
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    (tmp_path / "results.json").write_text('{}')
    pred.write_bytes(b'changed artifact')
    with pytest.raises(AssertionError, match='checksum mismatch'):
        verify_run(tmp_path)

def test_verifier_rejects_metric_drift():
    with pytest.raises(AssertionError, match='!='):
        same({"mae": .5}, {"mae": .6}, 'metrics')
