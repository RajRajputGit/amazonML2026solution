from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import SOURCE_FILES, Paths


ENTITY_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string", usecols=columns, keep_default_na=False)


def verify_training_files(paths: Paths) -> dict:
    report = {}
    for label, filename in SOURCE_FILES.items():
        path = (paths.train_dir / filename).resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        df = read_tsv(path)
        report[label] = {
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            "columns": list(df.columns),
            "rows": int(len(df)),
        }
    (paths.eda_dir / "data_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_training(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = read_tsv(paths.train_dir / SOURCE_FILES["S1"], ENTITY_COLUMNS)
    s2 = read_tsv(paths.train_dir / SOURCE_FILES["S2"], ENTITY_COLUMNS)
    s3 = read_tsv(paths.train_dir / SOURCE_FILES["S3"], ENTITY_COLUMNS)
    gt = read_tsv(paths.train_dir / SOURCE_FILES["GT"])
    return s1, s2, s3, gt

