"""Splits that decide what "generalization" is allowed to mean.

A random split of a medicinal-chemistry dataset mostly asks the model to
interpolate within series it has already seen, because analogues of the same
scaffold land on both sides. A scaffold split asks it to leave the series.
Both are reported here: the random split is kept deliberately, as the
interpolation comparator that shows how much of a headline number came from
the split rather than the model.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold, or the empty string if it cannot be derived."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    except Exception:  # noqa: BLE001 - malformed input is data, not a crash
        return ""


def scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)
    return dict(groups)


def scaffold_split(
    smiles: list[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole scaffold groups to train/val/test, largest first.

    Filling largest-first makes the split deterministic and puts the common
    series in training, which is the honest version: the test set is then the
    rarer chemistry, not a lucky draw of singletons.
    """
    groups = scaffold_groups(smiles)
    ordered = sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    n = len(smiles)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for group in ordered:
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            test.extend(group)

    rng = np.random.default_rng(seed)
    return tuple(np.sort(rng.permutation(np.array(part, dtype=int))) for part in (train, val, test))


def random_split(
    n: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The interpolation comparator. Not a baseline to beat - a ceiling to contrast."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    return (
        np.sort(order[:n_train]),
        np.sort(order[n_train : n_train + n_val]),
        np.sort(order[n_train + n_val :]),
    )


def butina_clusters(smiles: list[str], cutoff: float = 0.4) -> np.ndarray:
    """Cluster by Tanimoto distance; returns a cluster id per compound.

    Used to describe how much of the test set sits far from any training
    compound, which is what makes a scaffold split hard or easy.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [_GEN.GetFingerprint(m) if m is not None else None for m in mols]
    valid = [i for i, f in enumerate(fps) if f is not None]
    kept = [fps[i] for i in valid]

    distances: list[float] = []
    for i in range(1, len(kept)):
        sims = DataStructs.BulkTanimotoSimilarity(kept[i], kept[:i])
        distances.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(distances, len(kept), cutoff, isDistData=True)
    labels = np.full(len(smiles), -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[valid[m]] = cid
    return labels


def nearest_train_similarity(train_smiles: list[str], test_smiles: list[str]) -> np.ndarray:
    """Max Tanimoto of each test compound to anything in training.

    This is the number that says whether a scaffold split actually moved the
    test set away from training, rather than just relabelling it.
    """
    train_fps = [_GEN.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in train_smiles) if m is not None]
    out = np.zeros(len(test_smiles), dtype=float)
    for i, smi in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not train_fps:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(mol), train_fps))
    return out
