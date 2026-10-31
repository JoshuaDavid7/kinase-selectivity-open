"""Molecular featurization: fingerprints, descriptors, and graph tensors.

Two views of the same molecule. The fingerprint branch sees a fixed-length
substructure count; the graph branch sees atoms and bonds and learns its own
representation. The model in `model.py` uses both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

# Twelve cheap, interpretable descriptors. Kept short on purpose: these exist to
# give the fingerprint branch a few global properties it cannot count locally,
# not to be a descriptor model in their own right.
DESCRIPTORS: list[tuple[str, callable]] = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
]

FP_DIM = MORGAN_BITS + len(DESCRIPTORS)

# Atom feature vocabulary.
ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_DIM = (
    len(ATOMS) + 1 + len(DEGREES) + 1 + len(CHARGES) + 1 + len(HYBRIDIZATIONS) + 1 + len(NUM_HS) + 1 + 2
)
BOND_DIM = len(BOND_TYPES) + 1 + 2


def _one_hot(value, choices: list) -> list[float]:
    """One-hot with a trailing catch-all bucket, so an unseen value never crashes."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOMS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Morgan bits plus descriptors, as one float32 vector."""
    bits = np.zeros(MORGAN_BITS, dtype=np.float32)
    on = _GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
    bits[: len(on)] = on
    desc = np.array([fn(mol) for _, fn in DESCRIPTORS], dtype=np.float32)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([bits, desc])


@dataclass
class MolGraph:
    """A molecule as directed bond messages.

    `atom_f` is (n_atoms, ATOM_DIM); `bond_f` is (n_directed_bonds, BOND_DIM).
    Each undirected bond becomes two directed edges, because a D-MPNN passes
    messages along bonds with a direction and needs to exclude the reverse edge
    when it aggregates - `rev` records where that reverse edge lives.
    """

    atom_f: np.ndarray
    bond_f: np.ndarray
    src: np.ndarray  # originating atom of each directed bond
    dst: np.ndarray  # receiving atom of each directed bond
    rev: np.ndarray  # index of the opposite directed bond

    @property
    def n_atoms(self) -> int:
        return len(self.atom_f)

    @property
    def n_bonds(self) -> int:
        return len(self.bond_f)


def mol_to_graph(mol: Chem.Mol) -> MolGraph:
    atom_f = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    if len(atom_f) == 0:
        atom_f = np.zeros((1, ATOM_DIM), dtype=np.float32)

    bond_f: list[list[float]] = []
    src: list[int] = []
    dst: list[int] = []
    rev: list[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        # forward edge i->j, then reverse j->i; each records the other's index
        forward = len(bond_f)
        bond_f.append(features)
        src.append(i)
        dst.append(j)
        backward = len(bond_f)
        bond_f.append(features)
        src.append(j)
        dst.append(i)
        rev.extend([backward, forward])

    return MolGraph(
        atom_f=atom_f,
        bond_f=np.array(bond_f, dtype=np.float32).reshape(-1, BOND_DIM),
        src=np.array(src, dtype=np.int64),
        dst=np.array(dst, dtype=np.int64),
        rev=np.array(rev, dtype=np.int64),
    )


def parse(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


@dataclass
class BatchedGraph:
    """Several molecules packed into flat tensors with a scatter index."""

    atom_f: torch.Tensor
    bond_f: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    rev: torch.Tensor
    atom_to_mol: torch.Tensor
    n_mols: int

    def to(self, device: torch.device) -> "BatchedGraph":
        return BatchedGraph(
            atom_f=self.atom_f.to(device),
            bond_f=self.bond_f.to(device),
            src=self.src.to(device),
            dst=self.dst.to(device),
            rev=self.rev.to(device),
            atom_to_mol=self.atom_to_mol.to(device),
            n_mols=self.n_mols,
        )


def collate(graphs: list[MolGraph]) -> BatchedGraph:
    """Offset each molecule's atom and bond indices, then concatenate."""
    atom_f, bond_f, src, dst, rev, atom_to_mol = [], [], [], [], [], []
    atom_offset = 0
    bond_offset = 0
    for i, g in enumerate(graphs):
        atom_f.append(g.atom_f)
        bond_f.append(g.bond_f)
        src.append(g.src + atom_offset)
        dst.append(g.dst + atom_offset)
        rev.append(g.rev + bond_offset)
        atom_to_mol.append(np.full(g.n_atoms, i, dtype=np.int64))
        atom_offset += g.n_atoms
        bond_offset += g.n_bonds

    return BatchedGraph(
        atom_f=torch.from_numpy(np.concatenate(atom_f)),
        bond_f=torch.from_numpy(np.concatenate(bond_f)) if bond_offset else torch.zeros((0, BOND_DIM)),
        src=torch.from_numpy(np.concatenate(src)) if bond_offset else torch.zeros(0, dtype=torch.long),
        dst=torch.from_numpy(np.concatenate(dst)) if bond_offset else torch.zeros(0, dtype=torch.long),
        rev=torch.from_numpy(np.concatenate(rev)) if bond_offset else torch.zeros(0, dtype=torch.long),
        atom_to_mol=torch.from_numpy(np.concatenate(atom_to_mol)),
        n_mols=len(graphs),
    )
