"""Fold-nested calibration helpers used by the fixed analysis protocol."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


EPSILON = 1e-6


def fit_calibrator(kind: str, probabilities: np.ndarray, outcome: np.ndarray):
    probabilities = np.clip(np.asarray(probabilities, float), EPSILON, 1 - EPSILON)
    if kind == "sigmoid":
        model = LogisticRegression(C=1e10, max_iter=1000).fit(probabilities.reshape(-1, 1), outcome)
        return lambda values: model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1]
    model = IsotonicRegression(out_of_bounds="clip").fit(probabilities, outcome)
    return model.predict


def choose_family(probabilities: np.ndarray, outcome: np.ndarray, seed: int = 13) -> str:
    scores = {}
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    for kind in ("sigmoid", "isotonic"):
        values = []
        for train, valid in splitter.split(probabilities.reshape(-1, 1), outcome):
            predict = fit_calibrator(kind, probabilities[train], outcome[train])
            calibrated = np.clip(predict(probabilities[valid]), EPSILON, 1 - EPSILON)
            values.append(brier_score_loss(outcome[valid], calibrated))
        scores[kind] = float(np.mean(values))
    return min(scores, key=scores.get)


def nested_calibration(outcome: np.ndarray, probabilities: np.ndarray, folds: np.ndarray):
    output = np.empty(len(outcome)); choices = []
    for fold in np.unique(folds):
        valid = folds == fold
        kind = choose_family(probabilities[~valid], outcome[~valid])
        choices.append(kind)
        predict = fit_calibrator(kind, probabilities[~valid], outcome[~valid])
        output[valid] = np.clip(predict(probabilities[valid]), EPSILON, 1 - EPSILON)
    return output, choices

