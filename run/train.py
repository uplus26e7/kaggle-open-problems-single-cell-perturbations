import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dacite import from_dict
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from src.conf import TrainConfig
from src.metrics import mrrmse
from src.reducer.functions import reverse_transform as reducer_inverse_transform
from src.reducer.functions import transform as reducer_transform
from src.utils.gcs import gsutil_rsync

warnings.simplefilter("ignore")


@hydra.main(version_base="1.2", config_path="conf", config_name="train")
def main(dict_cfg: DictConfig) -> None:
    cfg = from_dict(data_class=TrainConfig, data=dict_cfg)

    # prepare data
    data_dir = Path(cfg.dir.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    gsutil_rsync(source=f"{cfg.gcs.bucket}/data", target=str(data_dir))

    # prepare processed data
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    gsutil_rsync(source=f"{cfg.gcs.bucket}/processed", target=str(processed_dir))

    # read data
    de_train = pd.read_parquet(data_dir / "de_train.parquet")
    id_map = pd.read_csv(data_dir / "id_map.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")
    print(f"de_train: {de_train.shape}")
    print(f"id_map: {id_map.shape}")
    print(f"folds: {folds.shape}")

    df = de_train.merge(folds, on=["cell_type", "sm_name"])
    X = df[["cell_type", "sm_name"]].to_numpy()
    y = df.drop(
        columns=["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control", "fold"]
    ).to_numpy()
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    # preprocess features
    ohe = OneHotEncoder(sparse=False)
    X = ohe.fit_transform(X)

    # reduce targets
    y_reduced = reducer_transform(y=y, cfg=cfg.reducer, processed_dir=processed_dir)
    print(f"y_reduced: {y_reduced.shape}")

    # train
    oof_preds = np.zeros(y_reduced.shape)
    for fold in range(4):
        train_idx = df["fold"] != fold
        valid_idx = df["fold"] == fold
        X_train = X[train_idx, :]
        X_valid = X[valid_idx, :]
        y_train = y_reduced[train_idx, :]
        y_valid = y_reduced[valid_idx, :]

        model = Ridge()
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        oof_preds[valid_idx, :] = preds_valid

    # reverse targets
    y_reversed = reducer_inverse_transform(
        y_transformed=oof_preds, cfg=cfg.reducer, processed_dir=processed_dir
    )

    # evaluate
    score = mrrmse(y_pred=y_reversed, y_true=y)
    print(f"score: {score:.06f}")

    raise NotImplementedError

    # sync processed data
    gsutil_rsync(source=str(processed_dir), target=f"{cfg.gcs.bucket}/processed")


if __name__ == "__main__":
    main()
