from typing import Optional

from pydantic.dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str


@dataclass
class GCSConfig:
    bucket: str


@dataclass
class CreateFoldsConfig:
    dir: DirConfig
    gcs: GCSConfig


@dataclass
class ReducerConfig:
    name: str
    n_components: int
    pca_whiten: Optional[bool]


@dataclass
class TrainConfig:
    dir: DirConfig
    gcs: GCSConfig
    reducer: ReducerConfig
    seed: int
