import warnings
from pathlib import Path

import hydra
from dacite import from_dict
from omegaconf import DictConfig

from src.conf import CreateFoldsConfig
from src.utils.gcs import gsutil_rsync

warnings.simplefilter("ignore")


@hydra.main(version_base="1.2", config_path="conf", config_name="create_folds")
def main(dict_cfg: DictConfig) -> None:
    cfg = from_dict(data_class=CreateFoldsConfig, data=dict_cfg)

    # prepare data
    data_dir = Path(cfg.dir.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    gsutil_rsync(source=f"{cfg.gcs.bucket}/data", target=str(data_dir))

    # prepare processed data
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    gsutil_rsync(source=f"{cfg.gcs.bucket}/processed", target=str(processed_dir))


if __name__ == "__main__":
    main()
