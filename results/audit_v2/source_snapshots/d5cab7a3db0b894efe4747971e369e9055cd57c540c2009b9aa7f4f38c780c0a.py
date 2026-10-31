"""Training a deep ensemble on sparse multitask labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .featurize import BatchedGraph, collate, fingerprint, mol_to_graph, parse
from .model import MolecularNetwork, masked_mse

TARGETS = ["JAK1", "JAK2", "JAK3", "TYK2"]


@dataclass
class Featurized:
    """Everything the model needs, computed once and reused across the ensemble."""

    smiles: list[str]
    graphs: list
    fps: np.ndarray
    y: np.ndarray  # (n, n_targets), NaN where unmeasured
    mask: np.ndarray  # 1.0 where measured

    def __len__(self) -> int:
        return len(self.smiles)


def featurize_frame(df, targets: list[str] = TARGETS) -> Featurized:
    """Parse once. Rows whose SMILES RDKit rejects are dropped, not patched."""
    smiles, graphs, fps, ys = [], [], [], []
    for _, row in df.iterrows():
        mol = parse(row["canonical_smiles"])
        if mol is None:
            continue
        smiles.append(row["canonical_smiles"])
        graphs.append(mol_to_graph(mol))
        fps.append(fingerprint(mol))
        ys.append([row[t] for t in targets])

    y = np.array(ys, dtype=np.float32)
    return Featurized(
        smiles=smiles,
        graphs=graphs,
        fps=np.stack(fps).astype(np.float32),
        y=y,
        mask=np.isfinite(y).astype(np.float32),
    )


class _Subset(Dataset):
    def __init__(self, data: Featurized, idx: np.ndarray):
        self.data = data
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        return self.data.graphs[j], self.data.fps[j], self.data.y[j], self.data.mask[j]


def _collate_batch(items):
    graphs, fps, ys, masks = zip(*items)
    return (
        collate(list(graphs)),
        torch.from_numpy(np.stack(fps)),
        torch.from_numpy(np.nan_to_num(np.stack(ys))),
        torch.from_numpy(np.stack(masks)),
    )


@dataclass
class Standardizer:
    """Per-target mean and std, computed on training labels only.

    Each target gets its own scaling because the heads share a trunk: without
    it a target with a wider pKi range dominates the gradient purely through
    its units.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray, mask: np.ndarray) -> "Standardizer":
        mean = np.zeros(y.shape[1], dtype=np.float32)
        std = np.ones(y.shape[1], dtype=np.float32)
        for t in range(y.shape[1]):
            observed = y[mask[:, t] > 0, t]
            if len(observed):
                mean[t] = observed.mean()
                s = observed.std()
                std[t] = s if s > 1e-6 else 1.0
        return cls(mean=mean, std=std)

    def encode(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def train_one(
    data: Featurized,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    standardizer: Standardizer,
    kind: str = "hybrid",
    hidden: int = 96,
    depth: int = 3,
    dropout: float = 0.1,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 12,
    seed: int = 0,
) -> MolecularNetwork:
    """Train a single member, early-stopping on masked validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaled = Featurized(
        smiles=data.smiles,
        graphs=data.graphs,
        fps=data.fps,
        y=np.nan_to_num(standardizer.encode(data.y)).astype(np.float32),
        mask=data.mask,
    )

    model = MolecularNetwork(n_targets=data.y.shape[1], kind=kind, hidden=hidden, depth=depth, dropout=dropout)
    if model.fp_encoder is not None:
        model.fp_encoder.fit_descriptors(torch.from_numpy(data.fps[train_idx]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    train_loader = DataLoader(
        _Subset(scaled, train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate_batch
    )
    val_loader = DataLoader(
        _Subset(scaled, val_idx), batch_size=batch_size * 2, shuffle=False, collate_fn=_collate_batch
    )

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0

    for _ in range(epochs):
        model.train()
        for graph, fp, y, mask in train_loader:
            optimizer.zero_grad()
            loss = masked_mse(model(graph, fp), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        total, weight = 0.0, 0.0
        with torch.no_grad():
            for graph, fp, y, mask in val_loader:
                n = float(mask.sum())
                if n == 0:
                    continue
                total += float(masked_mse(model(graph, fp), y, mask)) * n
                weight += n
        val_loss = total / weight if weight else float("inf")
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: MolecularNetwork, data: Featurized, idx: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Standardized-space predictions for a subset."""
    loader = DataLoader(_Subset(data, idx), batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)
    out = []
    model.eval()
    with torch.no_grad():
        for graph, fp, _, _ in loader:
            out.append(model(graph, fp).numpy())
    return np.concatenate(out) if out else np.zeros((0, data.y.shape[1]), dtype=np.float32)


@dataclass
class EnsemblePrediction:
    mu: np.ndarray
    sigma: np.ndarray
    members: list[np.ndarray]


def train_ensemble(
    data: Featurized,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    predict_idx: np.ndarray,
    n_members: int = 5,
    base_seed: int = 0,
    **kwargs,
) -> tuple[EnsemblePrediction, list[MolecularNetwork], Standardizer]:
    """Train `n_members` independently and return their mean and spread.

    Members differ only in initialization and batch order. That is enough for a
    useful *relative* spread; it is not enough for a calibrated absolute one,
    which is what `conformal.py` is for.
    """
    standardizer = Standardizer.fit(data.y[train_idx], data.mask[train_idx])

    models, preds = [], []
    for i in range(n_members):
        model = train_one(data, train_idx, val_idx, standardizer, seed=base_seed + i, **kwargs)
        models.append(model)
        preds.append(standardizer.decode(predict(model, data, predict_idx)))

    stack = np.stack(preds)
    return (
        EnsemblePrediction(mu=stack.mean(axis=0), sigma=stack.std(axis=0), members=preds),
        models,
        standardizer,
    )


def train_ensemble_multi(
    data: Featurized,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    predict_sets: list[np.ndarray],
    n_members: int = 5,
    base_seed: int = 0,
    checkpoint_dir: Path | None = None,
    **kwargs,
) -> list[EnsemblePrediction]:
    """Train one ensemble and predict several index sets with it.

    Calibration and test predictions must come from the *same* fitted members.
    Training a second ensemble for the second set - even with identical seeds -
    couples the conformal quantile to a separate training run, and any drift
    between them shows up as an unstable interval width rather than as an
    obviously wrong number.
    """
    standardizer = Standardizer.fit(data.y[train_idx], data.mask[train_idx])

    per_set: list[list[np.ndarray]] = [[] for _ in predict_sets]
    for i in range(n_members):
        model = train_one(data, train_idx, val_idx, standardizer, seed=base_seed + i, **kwargs)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "kind": kwargs.get("kind", "hybrid"),
                "hidden": kwargs.get("hidden", 96), "depth": kwargs.get("depth", 3),
                "dropout": kwargs.get("dropout", 0.1), "n_targets": data.y.shape[1],
                "standardizer_mean": torch.from_numpy(standardizer.mean),
                "standardizer_std": torch.from_numpy(standardizer.std),
                "seed": base_seed + i,
            }, checkpoint_dir / f"member_{i}.pt")
        for s, idx in enumerate(predict_sets):
            per_set[s].append(standardizer.decode(predict(model, data, idx)))

    out = []
    for preds in per_set:
        stack = np.stack(preds)
        out.append(EnsemblePrediction(mu=stack.mean(axis=0), sigma=stack.std(axis=0), members=preds))
    return out
