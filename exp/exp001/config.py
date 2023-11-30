from pydantic.dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str


@dataclass
class ReducerConfig:
    n_components: int
    whiten: bool


@dataclass
class TargetEncoderConfig:
    min_samples_leaf: int
    smoothing: float


@dataclass
class ModelConfig:
    name: str
    alpha: float


@dataclass
class WandbConfig:
    project: str
    use_wandb: bool


@dataclass
class TrainConfig:
    dir: DirConfig
    reducer: ReducerConfig
    seed: int
    target_encoder: TargetEncoderConfig
    model: ModelConfig
    wandb: WandbConfig
