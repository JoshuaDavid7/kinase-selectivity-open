"""A directed message-passing network fused with a fingerprint branch.

Written from the algorithm rather than taken from a library: no chemprop, no
torch-geometric, no DGL. The message passing is plain scatter-add over directed
bonds, which is the whole of a D-MPNN once the bookkeeping is in place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .featurize import ATOM_DIM, BOND_DIM, FP_DIM, MORGAN_BITS, BatchedGraph


class DMPNNEncoder(nn.Module):
    """Message passing on directed bonds, pooled to one vector per molecule.

    The message on bond (u->v) at step t+1 is built from the messages on every
    bond entering u, *except* the reverse bond (v->u). Excluding the reverse is
    the point of the directed formulation: without it, information bounces back
    and forth along a single bond and the depth stops meaning anything.
    """

    def __init__(self, hidden: int = 96, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ATOM_DIM + hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, batch: BatchedGraph) -> torch.Tensor:
        atom_f, bond_f = batch.atom_f, batch.bond_f
        src, dst, rev = batch.src, batch.dst, batch.rev
        n_atoms = atom_f.size(0)

        if bond_f.size(0) == 0:
            # A molecule with no bonds still needs a readout.
            atom_h = self.act(self.W_o(torch.cat([atom_f, atom_f.new_zeros(n_atoms, self.hidden)], dim=1)))
            return self._pool(atom_h, batch)

        # Initial bond message: the originating atom plus the bond itself.
        h0 = self.act(self.W_i(torch.cat([atom_f[src], bond_f], dim=1)))
        h = h0

        for _ in range(self.depth - 1):
            # Sum incoming messages at each atom, then subtract the reverse bond
            # so bond (u->v) never sees what it just sent back from v.
            atom_sum = h.new_zeros(n_atoms, self.hidden).index_add_(0, dst, h)
            message = atom_sum[src] - h[rev]
            h = self.act(h0 + self.W_h(message))
            h = self.dropout(h)

        atom_sum = h.new_zeros(n_atoms, self.hidden).index_add_(0, dst, h)
        atom_h = self.act(self.W_o(torch.cat([atom_f, atom_sum], dim=1)))
        atom_h = self.dropout(atom_h)
        return self._pool(atom_h, batch)

    @staticmethod
    def _pool(atom_h: torch.Tensor, batch: BatchedGraph) -> torch.Tensor:
        """Mean over each molecule's atoms."""
        totals = atom_h.new_zeros(batch.n_mols, atom_h.size(1))
        totals.index_add_(0, batch.atom_to_mol, atom_h)
        counts = atom_h.new_zeros(batch.n_mols).index_add_(
            0, batch.atom_to_mol, torch.ones_like(batch.atom_to_mol, dtype=atom_h.dtype)
        )
        return totals / counts.clamp(min=1).unsqueeze(1)


class FingerprintEncoder(nn.Module):
    def __init__(self, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(FP_DIM))
        self.register_buffer("feature_scale", torch.ones(FP_DIM))
        self.net = nn.Sequential(
            nn.Linear(FP_DIM, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net((fp - self.feature_mean) / self.feature_scale)

    def fit_descriptors(self, train_fp: torch.Tensor) -> None:
        """Fit descriptor scaling on training molecules; leave Morgan bits alone."""
        with torch.no_grad():
            desc = train_fp[:, MORGAN_BITS:]
            self.feature_mean[MORGAN_BITS:] = desc.mean(dim=0)
            scale = desc.std(dim=0, unbiased=False)
            self.feature_scale[MORGAN_BITS:] = scale.clamp(min=1e-6)


class MolecularNetwork(nn.Module):
    """Hybrid encoder with a multitask head.

    `kind` selects which encoders are active, so the same class provides the
    fingerprint-only and graph-only ablations rather than needing three models.
    Outputs are one head per target; a target a compound was not measured
    against is masked out of the loss rather than imputed.
    """

    def __init__(
        self,
        n_targets: int = 4,
        kind: str = "hybrid",
        hidden: int = 96,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kind not in {"hybrid", "fp", "graph"}:
            raise ValueError(f"unknown kind: {kind}")
        self.kind = kind

        self.graph_encoder = DMPNNEncoder(hidden, depth, dropout) if kind in {"hybrid", "graph"} else None
        self.fp_encoder = FingerprintEncoder(hidden, dropout) if kind in {"hybrid", "fp"} else None

        joint = hidden * (2 if kind == "hybrid" else 1)
        self.head = nn.Sequential(
            nn.Linear(joint, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, graph: BatchedGraph | None, fp: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if self.graph_encoder is not None:
            parts.append(self.graph_encoder(graph))
        if self.fp_encoder is not None:
            parts.append(self.fp_encoder(fp))
        return self.head(torch.cat(parts, dim=1))

    def embed(self, graph: BatchedGraph | None, fp: torch.Tensor | None) -> torch.Tensor:
        """The joint representation before the head, for similarity work."""
        parts = []
        if self.graph_encoder is not None:
            parts.append(self.graph_encoder(graph))
        if self.fp_encoder is not None:
            parts.append(self.fp_encoder(fp))
        return torch.cat(parts, dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over observed entries only.

    `mask` is 1 where a label exists. Missing labels contribute nothing - they
    are not zeros, and treating them as zeros would teach the model that an
    unmeasured compound is inactive.
    """
    observed = mask > 0
    if not observed.any():
        return pred.sum() * 0.0
    return (((pred[observed] - target[observed]) ** 2) * mask[observed]).sum() / mask[observed].sum()
