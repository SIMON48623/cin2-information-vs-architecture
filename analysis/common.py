from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((index == b).mean() * abs(y[index == b].mean() - p[index == b].mean())
                     for b in range(bins) if (index == b).any()))


def calibration_parameters(y, p):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs").fit(x, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def auc_variance(y, p):
    """DeLong variance using midranks (Sun and Xu fast formulation)."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    comparisons = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = comparisons.mean(1), comparisons.mean(0)
    auc = float(comparisons.mean())
    variance = float(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg))
    return auc, variance


def delong_ci(y, p):
    auc, variance = auc_variance(y, p)
    half = 1.959963984540054 * np.sqrt(max(variance, 0))
    return auc, max(0.0, auc - half), min(1.0, auc + half)


def delong_compare(y, a, b):
    y = np.asarray(y, int); a = np.asarray(a); b = np.asarray(b)
    pos, neg = y == 1, y == 0
    def influence(p):
        matrix = (p[pos, None] > p[neg][None, :]).astype(float) + 0.5 * (p[pos, None] == p[neg][None, :])
        return matrix.mean(), matrix.mean(1), matrix.mean(0)
    aa, ap, an = influence(a); bb, bp, bn = influence(b)
    variance = np.var(ap - bp, ddof=1) / pos.sum() + np.var(an - bn, ddof=1) / neg.sum()
    z = (aa - bb) / np.sqrt(max(variance, 1e-15))
    return float(aa - bb), float(2 * norm.sf(abs(z)))


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def shared_bootstrap_indices(y, n_boot=2000, seed=13):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    indices = []
    while len(indices) < n_boot:
        sample = rng.integers(0, len(y), len(y))
        if np.unique(y[sample]).size == 2:
            indices.append(sample)
    return indices


def bootstrap_auc_difference(y, a, b, indices):
    values = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in indices]
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    return float(point), *[float(x) for x in np.quantile(values, [0.025, 0.975])]


def _fit_calibrator(kind, p, y):
    if kind == "sigmoid":
        fit = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
        return lambda x: fit.predict_proba(logit(np.clip(x, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
    fit = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return fit.predict


def nested_calibration(y, p, folds):
    out = np.zeros(len(y)); picks = []
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        inner = StratifiedKFold(4, shuffle=True, random_state=13)
        losses = {kind: [] for kind in ("sigmoid", "isotonic")}
        train_indices = np.flatnonzero(train)
        for inner_train, inner_valid in inner.split(p[train], y[train]):
            for kind in losses:
                predictor = _fit_calibrator(kind, p[train_indices[inner_train]], y[train_indices[inner_train]])
                predicted = predictor(p[train_indices[inner_valid]])
                losses[kind].append(brier_score_loss(y[train_indices[inner_valid]], predicted))
        chosen = min(losses, key=lambda k: np.mean(losses[k]))
        picks.append(chosen)
        out[valid] = _fit_calibrator(chosen, p[train], y[train])(p[valid])
    return np.clip(out, 1e-6, 1 - 1e-6), picks


def youden_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[np.argmax(tpr - fpr)])


def sensitivity_threshold(y, p, target=0.95):
    positives = np.sort(p[y == 1])
    return float(positives[max(0, int(np.floor((1 - target) * len(positives))))])


def wilson(successes, total):
    if total == 0: return [None, None]
    z = 1.959963984540054; phat = successes / total
    den = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / den
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / den
    return [float(center - half), float(center + half)]


def nested_threshold_report(y, p, folds, rule):
    predicted = np.zeros(len(y), dtype=bool); thresholds = []
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        threshold = youden_threshold(y[train], p[train]) if rule == "youden" else sensitivity_threshold(y[train], p[train])
        thresholds.append(threshold); predicted[valid] = p[valid] >= threshold
    tp = int(((predicted == 1) & (y == 1)).sum()); fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum()); fn = int(((predicted == 0) & (y == 1)).sum())
    return {"thresholds": thresholds, "threshold_median": float(np.median(thresholds)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tp / (tp + fn), "sensitivity_ci": wilson(tp, tp + fn),
            "specificity": tn / (tn + fp), "specificity_ci": wilson(tn, tn + fp),
            "ppv": tp / max(1, tp + fp), "npv": tn / max(1, tn + fn),
            "flagged": float(predicted.mean())}


def net_benefit(y, p, thresholds):
    values = []
    n = len(y)
    for threshold in thresholds:
        positive = p >= threshold
        tp = ((positive == 1) & (y == 1)).sum(); fp = ((positive == 1) & (y == 0)).sum()
        values.append(float(tp / n - fp / n * threshold / (1 - threshold)))
    return values

