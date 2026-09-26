from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 2026


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    artifacts_dir: Path

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def eda_dir(self) -> Path:
        return self.artifacts_dir / "eda"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"


def make_paths(data_dir: str = "dataset", artifacts_dir: str = "artifacts") -> Paths:
    root = Path.cwd()
    paths = Paths(root=root, data_dir=root / data_dir, artifacts_dir=root / artifacts_dir)
    paths.eda_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    return paths


SOURCE_FILES = {
    "S1": "train_source1.tsv",
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
    "GT": "train_ground_truth.tsv",
}

