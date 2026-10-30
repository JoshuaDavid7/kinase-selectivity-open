"""Pull and curate JAK-family binding affinities from ChEMBL.

Everything here reads from the public ChEMBL REST API, so the dataset is
reproducible from a clean checkout by anyone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"

# UniProt accession -> the single-protein human target ChEMBL resolves it to.
# The ChEMBL ids are recorded so a reader can check them, but the accession is
# what we query on: accessions are stable, ChEMBL ids are not guaranteed to be.
TARGETS: dict[str, dict[str, str]] = {
    "JAK1": {"accession": "P23458", "chembl_id": "CHEMBL2835"},
    "JAK2": {"accession": "O60674", "chembl_id": "CHEMBL2971"},
    "JAK3": {"accession": "P52333", "chembl_id": "CHEMBL2148"},
    "TYK2": {"accession": "P29597", "chembl_id": "CHEMBL3553"},
}

PAGE_SIZE = 1000
RETRIES = 4


def _get(path: str, params: dict) -> dict:
    """GET with a bounded retry, so a transient 5xx does not lose a long pull."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = requests.get(f"{BASE}/{path}", params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - re-raised below after retries
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"ChEMBL request failed after {RETRIES} attempts: {path}") from last


def resolve_target(accession: str) -> str:
    """Return the single-protein human target id for a UniProt accession."""
    payload = _get(
        "target.json",
        {
            "target_components__accession": accession,
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
        },
    )
    targets = payload["targets"]
    if len(targets) != 1:
        found = [t["target_chembl_id"] for t in targets]
        raise RuntimeError(f"expected exactly one target for {accession}, got {found}")
    return targets[0]["target_chembl_id"]


def fetch_activities(target_chembl_id: str, standard_type: str = "Ki") -> pd.DataFrame:
    """Page through every activity of one type for one target."""
    rows: list[dict] = []
    offset = 0
    while True:
        payload = _get(
            "activity.json",
            {
                "target_chembl_id": target_chembl_id,
                "standard_type": standard_type,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        batch = payload["activities"]
        rows.extend(batch)
        total = payload["page_meta"]["total_count"]
        offset += len(batch)
        if not batch or offset >= total:
            break
    return pd.DataFrame(rows)


KEEP = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "data_validity_comment",
    "activity_comment",
]


def pull_raw(cache_dir: Path, standard_type: str = "Ki") -> pd.DataFrame:
    """Fetch all four targets, caching each to disk so a rerun is free."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, meta in TARGETS.items():
        path = cache_dir / f"{name}_{standard_type}_raw.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            resolved = resolve_target(meta["accession"])
            if resolved != meta["chembl_id"]:
                raise RuntimeError(
                    f"{name}: accession {meta['accession']} now resolves to {resolved}, "
                    f"not the recorded {meta['chembl_id']}"
                )
            frame = fetch_activities(resolved, standard_type)
            frame = frame[[c for c in KEEP if c in frame.columns]]
            frame.to_csv(path, index=False)
        frame = frame.assign(target=name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@dataclass
class CurationReport:
    """What the curation threw away, and why. Printed into the README."""

    raw_rows: int
    dropped_no_smiles: int
    dropped_flagged: int
    dropped_censored: int
    dropped_no_pchembl: int
    kept_measurements: int
    unique_compounds: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def curate(raw: pd.DataFrame) -> tuple[pd.DataFrame, CurationReport]:
    """Reduce raw activities to one pKi per (compound, target).

    The rules, stated so they can be argued with:
      * a measurement needs a structure and a ChEMBL pChEMBL value
      * ChEMBL's own validity flag is trusted - flagged rows go
      * censored values ('>', '<') are dropped rather than imputed at the bound
      * replicate measurements of the same pair collapse to their median
    """
    n_raw = len(raw)

    has_smiles = raw["canonical_smiles"].notna()
    dropped_no_smiles = int((~has_smiles).sum())
    df = raw[has_smiles]

    unflagged = df["data_validity_comment"].isna()
    dropped_flagged = int((~unflagged).sum())
    df = df[unflagged]

    exact = df["standard_relation"] == "="
    dropped_censored = int((~exact).sum())
    df = df[exact]

    has_p = df["pchembl_value"].notna()
    dropped_no_pchembl = int((~has_p).sum())
    df = df[has_p]

    df = df.assign(pKi=pd.to_numeric(df["pchembl_value"], errors="coerce")).dropna(subset=["pKi"])

    # One structure can appear under several ChEMBL ids; key on the SMILES that
    # the model will actually see, not on the accession number.
    collapsed = (
        df.groupby(["canonical_smiles", "target"], as_index=False)
        .agg(pKi=("pKi", "median"), n_measurements=("pKi", "size"))
    )

    wide = collapsed.pivot(index="canonical_smiles", columns="target", values="pKi").reset_index()
    wide.columns.name = None
    for name in TARGETS:
        if name not in wide.columns:
            wide[name] = pd.NA
    wide = wide[["canonical_smiles", *TARGETS]]

    report = CurationReport(
        raw_rows=n_raw,
        dropped_no_smiles=dropped_no_smiles,
        dropped_flagged=dropped_flagged,
        dropped_censored=dropped_censored,
        dropped_no_pchembl=dropped_no_pchembl,
        kept_measurements=len(collapsed),
        unique_compounds=len(wide),
    )
    return wide, report


def build(cache_dir: Path, out_path: Path, standard_type: str = "Ki") -> CurationReport:
    raw = pull_raw(cache_dir, standard_type)
    table, report = curate(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    (out_path.parent / "curation_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report
