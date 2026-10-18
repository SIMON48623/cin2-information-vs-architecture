from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cin2.cv import _tensor_loader, _train_tabpfn_oof


def _load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loader_order(seed: int) -> list[int]:
    x_num = np.arange(16, dtype=np.float32).reshape(-1, 1)
    x_cat = np.zeros((16, 0), dtype=np.int64)
    y = np.zeros(16, dtype=int)
    loader = _tensor_loader(x_num, x_cat, y, batch_size=4, shuffle=True, seed=seed)
    return [int(value) for batch in loader for value in batch[0][:, 0].tolist()]


def test_tensor_loader_shuffle_uses_supplied_model_seed():
    assert _loader_order(29) == _loader_order(29)
    assert _loader_order(29) != _loader_order(31)


def test_tabpfn_receives_configured_model_seed(tmp_path, monkeypatch):
    observed_random_states: list[int] = []

    class FakeTabPFNClassifier:
        def __init__(self, **kwargs):
            observed_random_states.append(kwargs["random_state"])

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            probability = observed_random_states[-1] / 100.0
            return np.tile([1.0 - probability, probability], (len(x), 1))

    monkeypatch.setitem(
        sys.modules,
        "tabpfn",
        SimpleNamespace(TabPFNClassifier=FakeTabPFNClassifier),
    )
    weights = tmp_path / "local.ckpt"
    weights.write_bytes(b"local-only test checkpoint")
    monkeypatch.setenv("TABPFN_MODEL_PATH", str(weights))

    frame = pd.DataFrame({"age": np.arange(8, dtype=float)})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = [
        (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
    ]
    config = {
        "features": {"numeric": ["age"]},
        "seeds": {"model": 29},
    }

    predictions, metadata = _train_tabpfn_oof(
        frame, y, folds, ["age"], config, "cpu"
    )

    assert metadata["status"] == "SUCCESS"
    assert observed_random_states == [29, 29]
    np.testing.assert_allclose(predictions, 0.29)


def test_repeat_config_sets_every_protocol_seed_without_mutating_base():
    repeated = _load_script("scripts/04_repeated_cv.py", "repeated_cv_runner")
    base = {
        "cross_validation": {"random_state": 42, "inner_random_state": 13},
        "seeds": {"model": 13},
        "models": {
            "lr": {"random_state": 13},
            "rf": {"random_state": 13},
            "xgb": {"random_state": 13},
            "lgbm": {"random_state": 13},
            "catboost": {"random_seed": 13},
            "tabpfn": {"random_state": 13},
        },
    }

    configured = repeated.config_for_repeat(base, 7)

    assert configured["cross_validation"] == {
        "random_state": 1007,
        "inner_random_state": 2007,
    }
    assert configured["seeds"]["model"] == 2007
    assert configured["models"]["lr"]["random_state"] == 2007
    assert configured["models"]["rf"]["random_state"] == 2007
    assert configured["models"]["xgb"]["random_state"] == 2007
    assert configured["models"]["lgbm"]["random_state"] == 2007
    assert configured["models"]["catboost"]["random_seed"] == 2007
    assert configured["models"]["tabpfn"]["random_state"] == 2007
    assert base["cross_validation"]["random_state"] == 42
    assert base["seeds"]["model"] == 13


def test_combining_device_runs_requires_identical_y_and_folds():
    repeated = _load_script("scripts/04_repeated_cv.py", "repeated_cv_combiner")
    classical = pd.DataFrame(
        {"y": [0, 1, 0, 1], "fold": [1, 1, 2, 2], "lr": [0.1, 0.8, 0.2, 0.7]}
    )
    neural = pd.DataFrame(
        {"y": [0, 1, 0, 1], "fold": [1, 1, 2, 2], "tab": [0.2, 0.7, 0.3, 0.6]}
    )

    combined = repeated.combine_predictions(classical, neural)

    assert list(combined.columns) == ["y", "fold", "lr__full", "tab__full"]
    mismatched = neural.copy()
    mismatched.loc[0, "fold"] = 2
    try:
        repeated.combine_predictions(classical, mismatched)
    except RuntimeError as error:
        assert "y and fold" in str(error)
    else:
        raise AssertionError("Mismatched fold assignments were accepted")


def test_bouckaert_frank_formula_matches_hand_calculation():
    analysis = _load_script("analysis/15_repeated_cv.py", "repeated_cv_analysis")
    statistic, degrees_freedom, p_value = analysis.corrected_repeated_kfold_ttest(
        np.array([0.1, 0.2, 0.3, 0.4]),
        n_test_over_n_train=0.25,
    )

    assert abs(statistic - 2.7386127875258306) < 1e-12
    assert degrees_freedom == 3
    assert abs(p_value - 0.07142153823145017) < 1e-12
