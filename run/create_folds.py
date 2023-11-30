import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dacite import from_dict
from omegaconf import DictConfig

from src.conf import CreateFoldsConfig
from src.utils.gcs import gsutil_rsync

warnings.simplefilter("ignore")


def create_folds(de_train: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    test_cell_types = id_map["cell_type"].unique()
    test_sm_names = id_map["sm_name"].unique()
    train_sm_names = np.array(
        [
            sm_name
            for sm_name in de_train["sm_name"].unique()
            if sm_name not in test_sm_names
        ]
    )

    df = de_train[["cell_type", "sm_name"]]
    df["fold"] = -1

    for fold, cell_type in enumerate(df["cell_type"].unique()):
        if cell_type in test_cell_types:
            continue
        df.loc[
            (df["cell_type"] == cell_type) & (df["sm_name"].isin(test_sm_names)), "fold"
        ] = fold

    for fold, sm_names in enumerate(np.array_split(np.array(train_sm_names), 4)):
        df.loc[(df["fold"] == -1) & (df["sm_name"].isin(sm_names)), "fold"] = fold

    return df


@hydra.main(version_base="1.2", config_path="conf", config_name="create_folds")
def main(dict_cfg: DictConfig) -> None:
    cfg = from_dict(data_class=CreateFoldsConfig, data=dict_cfg)

    # prepare data
    data_dir = Path(cfg.dir.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    gsutil_rsync(source=f"{cfg.gcs.bucket}/data", target=str(data_dir))

    # cross validation
    de_train = pd.read_parquet(data_dir / "de_train.parquet")
    id_map = pd.read_csv(data_dir / "id_map.csv")
    df = create_folds(de_train, id_map)

    # save & upload data
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "folds.csv", index=False)
    gsutil_rsync(source=str(processed_dir), target=f"{cfg.gcs.bucket}/processed")


if __name__ == "__main__":
    main()
