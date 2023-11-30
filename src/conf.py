from pydantic.dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str


@dataclass
class CreateFoldsConfig:
    dir: DirConfig
