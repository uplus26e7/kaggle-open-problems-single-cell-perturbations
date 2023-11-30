import pickle
from pathlib import Path

import numpy as np
from pydantic import RootModel
from sklearn.decomposition import PCA  # type:ignore

from src.conf import ReducerConfig


def get_dir_name(cfg: ReducerConfig) -> str:
    params: dict = RootModel(cfg).model_dump()  # type:ignore
    dir_name = ""
    for k, v in params.items():
        if dir_name != "":
            dir_name += "_"
        dir_name += f"{k}={v}"
    return dir_name


def get_reducer(cfg: ReducerConfig) -> PCA:
    if cfg.name == "PCA":
        reducer = PCA(
            n_components=cfg.n_components, whiten=cfg.pca_whiten, random_state=3
        )
    else:
        raise NotImplementedError(f"Reducer: {cfg.name}")
    return reducer


def transform(y: np.ndarray, cfg: ReducerConfig, processed_dir: Path) -> np.ndarray:
    dir_name = get_dir_name(cfg)
    output_dir = processed_dir / "reducer" / dir_name
    if output_dir.exists():
        y_transformed = np.load(output_dir / "y_transformed.npz")["y_transformed"]
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        reducer = get_reducer(cfg=cfg)
        y_transformed = reducer.fit_transform(y)

        np.savez_compressed(output_dir / "y_transformed", y_transformed=y_transformed)
        pickle.dump(reducer, open(str(output_dir / "reducer.pkl"), "wb"))

    return y_transformed


def reverse_transform(
    y_transformed: np.ndarray, cfg: ReducerConfig, processed_dir: Path
) -> np.ndarray:
    dir_name = get_dir_name(cfg)
    output_dir = processed_dir / "reducer" / dir_name
    reducer = pickle.load(open(str(output_dir / "reducer.pkl"), "rb"))
    y_reversed = reducer.inverse_transform(y_transformed)
    return y_reversed
