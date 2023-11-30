import hydra
from dacite import from_dict
from omegaconf import DictConfig

from src.conf import CreateFoldsConfig


@hydra.main(version_base="1.2", config_path="conf", config_name="create_folds")
def main(dict_cfg: DictConfig) -> None:
    cfg = from_dict(data_class=CreateFoldsConfig, data=dict_cfg)
    print(cfg)


if __name__ == "__main__":
    main()
