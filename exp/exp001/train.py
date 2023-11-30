from pathlib import Path

import category_encoders as ce  # type:ignore
import hydra
import numpy as np
import pandas as pd
from config import TrainConfig
from dacite import from_dict
from omegaconf import DictConfig
from pydantic import RootModel
from sklearn.decomposition import PCA  # type:ignore
from sklearn.linear_model import Ridge  # type: ignore

import wandb


def mrrmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(dict_cfg: DictConfig) -> None:
    cfg = from_dict(data_class=TrainConfig, data=dict_cfg)
    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            config=RootModel(cfg).model_dump(),  # type: ignore
        )

    data_dir = Path(cfg.dir.data_dir)
    processed_dir = Path(cfg.dir.processed_dir)
    output_dir = Path(cfg.dir.output_dir) / "exp001"
    output_dir.mkdir(parents=True, exist_ok=True)

    # read data
    de_train = pd.read_parquet(data_dir / "de_train.parquet")
    id_map = pd.read_csv(data_dir / "id_map.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")

    # prepare inputs & targets
    df = de_train.merge(folds, on=["cell_type", "sm_name"])
    X = df[["cell_type", "sm_name"]]
    X_test = id_map[["cell_type", "sm_name"]]
    y = df.drop(
        columns=["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control", "fold"]
    ).to_numpy()
    print(f"X: {X.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y: {y.shape}")

    # reduce targets
    reducer = PCA(
        n_components=cfg.reducer.n_components,
        whiten=cfg.reducer.whiten,
        random_state=cfg.seed,
    )
    y_reduced = reducer.fit_transform(y)

    # train
    oof_preds_reduced = np.zeros(y_reduced.shape)
    test_preds_reduced = np.zeros([4, X_test.shape[0], y_reduced.shape[1]])
    for fold in range(4):
        train_idx = df["fold"] != fold
        valid_idx = df["fold"] == fold
        X_train = X[train_idx]
        X_valid = X[valid_idx]
        y_train = y_reduced[train_idx, :]
        # y_valid = y_reduced[valid_idx, :]

        # target encoding
        for i in range(y_reduced.shape[1]):
            encoder = ce.TargetEncoder(
                cols=["cell_type", "sm_name"],
                min_samples_leaf=cfg.target_encoder.min_samples_leaf,
                smoothing=cfg.target_encoder.smoothing,
            )
            if i == 0:
                X_train_encoded = encoder.fit_transform(X_train, y_train[:, i])
                X_valid_encoded = encoder.transform(X_valid)
                X_test_encoded = encoder.transform(X_test)
            else:
                X_tmp = encoder.fit_transform(X_train, y_train[:, i])
                X_train_encoded = np.concatenate([X_train_encoded, X_tmp], axis=1)
                X_tmp = encoder.transform(X_valid)
                X_valid_encoded = np.concatenate([X_valid_encoded, X_tmp], axis=1)
                X_tmp = encoder.transform(X_test)
                X_test_encoded = np.concatenate([X_test_encoded, X_tmp], axis=1)

        # modeling
        model = Ridge(cfg.model.alpha)
        model.fit(X_train_encoded, y_train)

        # inference
        oof_preds_reduced[valid_idx, :] = model.predict(X_valid_encoded)
        test_preds_reduced[fold, :, :] = model.predict(X_test_encoded)

    # inverse transform targets
    oof_preds = reducer.inverse_transform(oof_preds_reduced)
    test_preds = np.zeros([4, X_test.shape[0], y.shape[1]])
    for fold in range(4):
        test_preds[fold, :, :] = reducer.inverse_transform(
            test_preds_reduced[fold, :, :]
        )

    # scoring
    score = mrrmse(oof_preds, y)
    print(f"Score: {score:06f}")
    if cfg.wandb.use_wandb:
        wandb.log({"score": score})

    # save
    np.savez_compressed(
        output_dir / "preds",
        oof_preds_reduced=oof_preds_reduced,
        test_preds_reduced=test_preds_reduced,
        oof_preds=oof_preds,
        test_preds=test_preds,
    )

    # make submission
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    submission.iloc[:, 1:] = test_preds.mean(axis=0)
    submission.to_csv(output_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()
