"""Build the public JAK-family pKi table from ChEMBL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinase_open.data import build  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = build(cache_dir=root / "data" / "chembl_cache", out_path=root / "data" / "jak_pki.csv")
    for key, value in report.to_dict().items():
        print(f"{key:22s} {value}")
