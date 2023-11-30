import numpy as np


def mrrmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()
