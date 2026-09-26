from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


CLASSICAL = ["lr", "xgb", "lgbm", "catboost", "rf"]
PRIMARY_FAMILIES = CLASSICAL + [
    "ftt", "tabpfn", "tab", "tabmfm_pre030_colid1", "tabmfm_nopre_colid1"
]
EXTRA_CONFIGS = ["tabmfm_pre015_colid1", "tabmfm_pre030_colid0"]


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def configure_torch_runtime(config: dict) -> None:
    """Apply the recorded CPU runtime controls before any Torch model is fit."""
    import torch
    runtime = config.get("runtime", {})
    torch.set_num_threads(int(runtime.get("torch_num_threads", 8)))
    if bool(runtime.get("torch_deterministic_algorithms", True)):
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(True)


def classical_model(name: str, y_train: np.ndarray, config: dict):
    params = config["models"][name]
    if name == "lr":
        return LogisticRegression(**params)
    if name == "rf":
        return RandomForestClassifier(**params)
    if name == "xgb":
        from xgboost import XGBClassifier
        values = dict(params)
        values["scale_pos_weight"] = float((y_train == 0).sum() / (y_train == 1).sum())
        return XGBClassifier(**values)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    raise KeyError(name)
