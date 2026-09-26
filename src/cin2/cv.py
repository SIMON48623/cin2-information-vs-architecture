"""Leakage-controlled cross-validation using the article's actual estimators."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from .models import CLASSICAL, classical_model, configure_torch_runtime, set_seed
from .preprocessing import classical_preprocessor


@dataclass
class CVResult:
    predictions: pd.DataFrame
    metadata: dict


def fixed_folds(y: np.ndarray, config: dict):
    c = config["cross_validation"]
    splitter = StratifiedKFold(c["n_splits"], shuffle=c["shuffle"], random_state=c["random_state"])
    assignment = np.zeros(len(y), dtype=int)
    folds = []
    for number, (train, valid) in enumerate(splitter.split(np.zeros((len(y), 1)), y), 1):
        folds.append((train, valid))
        assignment[valid] = number
    return folds, assignment


def _split_columns(columns: list[str], config: dict) -> tuple[list[str], list[str]]:
    numeric = [c for c in config["features"]["numeric"] if c in columns]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical


def _clean_cat(value):
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "-", "--", "/", "\\", "nan", "none", "null", "unknown", "未知", "不详"}:
        return np.nan
    return text


def _fit_numeric(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame], columns: list[str]):
    if not columns:
        return [np.zeros((len(d), 0), dtype=np.float32) for d in [train_df, *other_dfs]]
    raw = train_df[columns].to_numpy(dtype=np.float32)
    median = np.nanmedian(raw, axis=0)
    filled = np.where(np.isnan(raw), median, raw)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    result = []
    for frame in [train_df, *other_dfs]:
        values = frame[columns].to_numpy(dtype=np.float32)
        values = np.where(np.isnan(values), median, values)
        result.append(((values - mean) / std).astype(np.float32))
    return result


def _fit_categories(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame], columns: list[str],
                    mapping_start: int = 1, unknown: int = 0):
    mappings = []
    train_arrays = []
    other_arrays = [[] for _ in other_dfs]
    for column in columns:
        values = train_df[column].map(_clean_cat).fillna("__MISSING__").astype(str)
        mapping = {value: i + mapping_start for i, value in enumerate(pd.unique(values))}
        mappings.append(mapping)
        train_arrays.append(values.map(mapping).to_numpy(dtype=np.int64))
        for index, frame in enumerate(other_dfs):
            series = frame[column].map(_clean_cat).fillna("__MISSING__").astype(str)
            other_arrays[index].append(series.map(lambda x: mapping.get(x, unknown)).to_numpy(dtype=np.int64))
    train = np.stack(train_arrays, axis=1) if columns else np.zeros((len(train_df), 0), dtype=np.int64)
    others = [np.stack(values, axis=1) if columns else np.zeros((len(frame), 0), dtype=np.int64)
              for values, frame in zip(other_arrays, other_dfs)]
    return [train, *others], mappings


def _tensor_loader(x_num, x_cat, y, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(
        torch.from_numpy(x_num).float(),
        torch.from_numpy(x_cat).long(),
        torch.from_numpy(y.astype(np.float32)).float(),
    )
    generator = torch.Generator().manual_seed(13)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _inner_split(outer_train: np.ndarray, y: np.ndarray, config: dict):
    cv = config["cross_validation"]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cv["inner_validation_fraction"],
        random_state=cv["inner_random_state"],
    )
    fit, stop = next(splitter.split(np.zeros((len(outer_train), 1)), y[outer_train]))
    return outer_train[fit], outer_train[stop]


def _make_deep_arrays(frame, fit_idx, stop_idx, valid_idx, columns, config, kind):
    numeric, categorical = _split_columns(columns, config)
    fit, stop, valid = frame.iloc[fit_idx], frame.iloc[stop_idx], frame.iloc[valid_idx]
    nums = _fit_numeric(fit, [stop, valid], numeric)
    if kind == "ftt":
        cats, mappings = _fit_categories(fit, [stop, valid], categorical, mapping_start=0, unknown=-1)
        for j, mapping in enumerate(mappings):
            unknown_index = len(mapping)
            for values in cats[1:]:
                values[:, j] = np.where(values[:, j] < 0, unknown_index, values[:, j])
        cards = [len(mapping) + 1 for mapping in mappings]
    else:
        cats, mappings = _fit_categories(fit, [stop, valid], categorical, mapping_start=1, unknown=0)
        cards = [len(mapping) for mapping in mappings] if kind == "tab" else [len(mapping) + 2 for mapping in mappings]
    return nums, cats, cards, numeric, categorical


def _predict_torch(model, loader, kind):
    import torch
    model.eval()
    result = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            if kind == "ftt":
                logits = model(None if x_num.shape[1] == 0 else x_num, x_cat).squeeze(-1)
            elif kind == "tab":
                logits = model(x_num, x_cat)
            else:
                logits = model.classify(x_num, x_cat)
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def _supervised_epoch(model, loader, optimizer, loss_fn, kind):
    model.train()
    device = next(model.parameters()).device
    for x_num, x_cat, outcome in loader:
        x_num, x_cat, outcome = x_num.to(device), x_cat.to(device), outcome.to(device)
        optimizer.zero_grad()
        if kind == "ftt":
            logits = model(None if x_num.shape[1] == 0 else x_num, x_cat).squeeze(-1)
        elif kind == "tab":
            logits = model(x_num, x_cat)
        else:
            logits = model.classify(x_num, x_cat)
        loss = loss_fn(logits, outcome)
        loss.backward()
        optimizer.step()


def _pretrain_mfm_epoch(model, loader, optimizer, cat_cards, mask_ratio):
    import torch
    from torch import nn
    from .tabmfm import apply_mfm_mask
    model.train()
    device = next(model.parameters()).device
    mse, ce = nn.MSELoss(reduction="none"), nn.CrossEntropyLoss(reduction="none")
    trained = False
    for x_num, x_cat, _ in loader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        masked_num, masked_cat, mask = apply_mfm_mask(x_num, x_cat, cat_cards, mask_ratio)
        hidden = model.encode(masked_num, masked_cat)
        terms = []
        if model.n_num:
            numeric_mask = mask[:, :model.n_num]
            if numeric_mask.any():
                terms.append(mse(model.recon_num(hidden[:, :model.n_num, :]), x_num)[numeric_mask].mean())
        if model.n_cat:
            categorical_mask = mask[:, model.n_num:]
            categorical_terms = []
            for j, logits in enumerate(model.recon_cat(hidden[:, model.n_num:, :])):
                selected = categorical_mask[:, j]
                if selected.any():
                    categorical_terms.append(ce(logits, x_cat[:, j])[selected].mean())
            if categorical_terms:
                terms.append(torch.stack(categorical_terms).mean())
        if not terms:
            continue
        loss = torch.stack(terms).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trained = True
    if not trained:
        raise RuntimeError("No masked elements were sampled during the Tab-MFM pretraining epoch")


def _family_kind(family: str) -> str:
    if family == "ftt":
        return "ftt"
    if family == "tab":
        return "tab"
    if family.startswith("tabmfm"):
        return "tabmfm"
    raise KeyError(family)


def _model_class_name(family: str) -> str:
    names = {
        "lr": "sklearn.linear_model.LogisticRegression",
        "rf": "sklearn.ensemble.RandomForestClassifier",
        "xgb": "xgboost.XGBClassifier",
        "lgbm": "lightgbm.LGBMClassifier",
        "catboost": "catboost.CatBoostClassifier",
        "ftt": "rtdl_revisiting_models.FTTransformer",
        "tab": "cin2.tab_transformer.FTTransformer",
        "tabpfn": "tabpfn.TabPFNClassifier",
    }
    return names.get(family, "cin2.tabmfm.TabTokTransformer")


def _train_deep_oof(frame, y, folds, columns, family, config, device, fast):
    import torch
    from torch import nn
    from rtdl_revisiting_models import FTTransformer as OfficialFTTransformer
    from .tab_transformer import FTTransformer as ArticleTabTransformer
    from .tabmfm import TabTokTransformer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    kind = _family_kind(family)
    tab_cfg = config["models"]["tab"]
    mfm_cfg = config["models"]["tabmfm"]
    ftt_cfg = config["models"]["ftt"]
    family_cfg = mfm_cfg.get("configurations", {}).get(family, {})
    do_pretrain = bool(family_cfg.get("pretrain", False))
    mask_ratio = float(family_cfg.get("mask_ratio", 0.30))
    use_column_identity = bool(family_cfg.get("column_identity", True))
    max_epochs = 2 if fast else int(ftt_cfg["epochs"] if kind == "ftt" else tab_cfg["epochs"] if kind == "tab" else mfm_cfg["finetune_epochs"])
    pretrain_epochs = 1 if fast else int(mfm_cfg["pretrain_epochs"])
    patience = int(ftt_cfg["patience"] if kind == "ftt" else tab_cfg["patience"] if kind == "tab" else mfm_cfg["patience"])
    batch_size = int(ftt_cfg["batch_size"] if kind == "ftt" else tab_cfg["batch_size"] if kind == "tab" else mfm_cfg["batch_size"])
    predictions = np.full(len(y), np.nan)
    selected_epochs, parameter_counts = [], []
    set_seed(13)
    for fold_number, (outer_train, outer_valid) in enumerate(folds, 1):
        fit_idx, stop_idx = _inner_split(outer_train, y, config)
        nums, cats, cards, numeric, _ = _make_deep_arrays(
            frame, fit_idx, stop_idx, outer_valid, columns, config, kind
        )
        fit_loader = _tensor_loader(nums[0], cats[0], y[fit_idx], batch_size, True)
        stop_loader = _tensor_loader(nums[1], cats[1], y[stop_idx], batch_size, False)
        valid_loader = _tensor_loader(nums[2], cats[2], y[outer_valid], batch_size, False)
        set_seed(13)
        if kind == "ftt":
            model = OfficialFTTransformer(
                n_cont_features=len(numeric), cat_cardinalities=cards, d_out=1,
                **OfficialFTTransformer.get_default_kwargs(),
            ).to(device)
            optimizer = model.make_default_optimizer()
        elif kind == "tab":
            model = ArticleTabTransformer(
                n_num=len(numeric), cat_cardinalities=cards,
                d_token=int(tab_cfg["d_token"]), n_head=int(tab_cfg["n_heads"]),
                n_layers=int(tab_cfg["n_layers"]), d_ff=int(tab_cfg["d_ff"]),
                dropout=float(tab_cfg["dropout"]),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(tab_cfg["learning_rate"]),
                                          weight_decay=float(tab_cfg["weight_decay"]))
        else:
            model = TabTokTransformer(
                n_num=len(numeric), cat_cards=cards,
                d_model=int(mfm_cfg["d_model"]), n_head=int(mfm_cfg["n_heads"]),
                n_layers=int(mfm_cfg["n_layers"]), d_ff=int(mfm_cfg["d_ff"]),
                dropout=float(mfm_cfg["dropout"]), use_col_id_emb=use_column_identity,
            ).to(device)
            if do_pretrain:
                pre_optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(mfm_cfg["pretrain_learning_rate"]),
                    weight_decay=float(mfm_cfg["pretrain_weight_decay"]),
                )
                for _ in range(pretrain_epochs):
                    _pretrain_mfm_epoch(model, fit_loader, pre_optimizer, cards, mask_ratio)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(mfm_cfg["finetune_learning_rate"]),
                weight_decay=float(mfm_cfg["finetune_weight_decay"]),
            )
        parameter_counts.append(sum(value.numel() for value in model.parameters() if value.requires_grad))
        positives = float(y[fit_idx].sum())
        negatives = float(len(fit_idx) - positives)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        )
        best_auc, best_epoch, best_state, bad = -np.inf, 0, None, 0
        for epoch in range(1, max_epochs + 1):
            _supervised_epoch(model, fit_loader, optimizer, loss_fn, kind)
            auc = roc_auc_score(y[stop_idx], _predict_torch(model, stop_loader, kind))
            if auc > best_auc + 1e-6:
                best_auc, best_epoch, bad = auc, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"No valid early-stopping checkpoint for {family}, fold {fold_number}")
        model.load_state_dict(best_state)
        predictions[outer_valid] = _predict_torch(model, valid_loader, kind)
        selected_epochs.append(best_epoch)
    return predictions, {
        "status": "SUCCESS",
        "model_class": _model_class_name(family),
        "selected_epochs": selected_epochs,
        "parameter_counts": parameter_counts,
        "pretrain_epochs": pretrain_epochs if do_pretrain else 0,
        "finetune_epoch_limit": max_epochs,
    }


def _tabpfn_arrays(train_df, valid_df, columns, config):
    numeric, categorical = _split_columns(columns, config)
    if numeric:
        raw_train = train_df[numeric].to_numpy(dtype=np.float32)
        median = np.nanmedian(raw_train, axis=0)
        raw_train = np.where(np.isnan(raw_train), median, raw_train)
        raw_valid = valid_df[numeric].to_numpy(dtype=np.float32)
        raw_valid = np.where(np.isnan(raw_valid), median, raw_valid)
    else:
        raw_train = np.zeros((len(train_df), 0), dtype=np.float32)
        raw_valid = np.zeros((len(valid_df), 0), dtype=np.float32)
    cats, _ = _fit_categories(train_df, [valid_df], categorical, mapping_start=1, unknown=0)
    return np.column_stack([raw_train, cats[0]]), np.column_stack([raw_valid, cats[1]]), len(numeric)


def _local_tabpfn_weights() -> Path | None:
    value = os.environ.get("TABPFN_MODEL_PATH")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _train_tabpfn_oof(frame, y, folds, columns, config, device):
    from tabpfn import TabPFNClassifier
    weights = _local_tabpfn_weights()
    if weights is None:
        return None, {
            "status": "SKIPPED (no local weights)",
            "model_class": _model_class_name("tabpfn"),
        }
    predictions = np.full(len(y), np.nan)
    set_seed(13)
    for train, valid in folds:
        x_train, x_valid, n_numeric = _tabpfn_arrays(frame.iloc[train], frame.iloc[valid], columns, config)
        model = TabPFNClassifier(
            categorical_features_indices=list(range(n_numeric, x_train.shape[1])),
            random_state=13,
            device=device,
            model_path=weights,
        )
        model.fit(x_train, y[train])
        predictions[valid] = model.predict_proba(x_valid)[:, 1]
    return predictions, {
        "status": "SUCCESS",
        "model_class": _model_class_name("tabpfn"),
        "weights_sha256_not_recorded": "Weight files are local-only and are not distributed.",
    }


def run_oof(frame: pd.DataFrame, y: np.ndarray, columns: list[str], families: list[str],
            config: dict, device: str = "cpu", fast: bool = False) -> CVResult:
    if any(family not in CLASSICAL for family in families):
        configure_torch_runtime(config)
    folds, assignment = fixed_folds(y, config)
    result = pd.DataFrame({"y": y.astype(int), "fold": assignment})
    family_metadata = {}
    for family in families:
        if family in CLASSICAL:
            values = np.full(len(y), np.nan)
            model_class = None
            for train, valid in folds:
                model = classical_model(family, y[train], config)
                model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
                pipeline = Pipeline([
                    ("preprocess", classical_preprocessor(columns, config)),
                    ("model", model),
                ])
                pipeline.fit(frame.iloc[train][columns], y[train])
                values[valid] = pipeline.predict_proba(frame.iloc[valid][columns])[:, 1]
            result[family] = values
            family_metadata[family] = {"status": "SUCCESS", "model_class": model_class}
        elif family == "tabpfn":
            values, metadata = _train_tabpfn_oof(frame, y, folds, columns, config, device)
            family_metadata[family] = metadata
            if values is not None:
                result[family] = values
        else:
            values, metadata = _train_deep_oof(frame, y, folds, columns, family, config, device, fast)
            result[family] = values
            family_metadata[family] = metadata
    return CVResult(result, {
        "mode": "FAST: real estimators with 1 pretraining epoch and 2 finetuning epochs; not paper results" if fast else "full protocol",
        "device": device,
        "n": len(y),
        "features": columns,
        "requested_families": families,
        "produced_families": [c for c in result.columns if c not in {"y", "fold"}],
        "families": family_metadata,
    })


def fit_final_lr_predict(train, y, external, columns, config):
    model = classical_model("lr", y, config)
    pipeline = Pipeline([("preprocess", classical_preprocessor(columns, config)), ("model", model)])
    pipeline.fit(train[columns], y)
    return pipeline.predict_proba(external[columns])[:, 1], {
        "status": "SUCCESS", "model_class": _model_class_name("lr")
    }


def fit_final_tabmfm_predict(train, y, external, columns, config, epochs, device="cpu", fast=False):
    import torch
    from torch import nn
    from .tabmfm import TabTokTransformer
    configure_torch_runtime(config)
    numeric, categorical = _split_columns(columns, config)
    nums = _fit_numeric(train, [external], numeric)
    cats, mappings = _fit_categories(train, [external], categorical, mapping_start=1, unknown=0)
    cards = [len(mapping) + 2 for mapping in mappings]
    cfg = config["models"]["tabmfm"]
    set_seed(13)
    model = TabTokTransformer(
        n_num=len(numeric), cat_cards=cards, d_model=int(cfg["d_model"]),
        n_head=int(cfg["n_heads"]), n_layers=int(cfg["n_layers"]), d_ff=int(cfg["d_ff"]),
        dropout=float(cfg["dropout"]), use_col_id_emb=True,
    ).to(device)
    train_loader = _tensor_loader(nums[0], cats[0], y, int(cfg["batch_size"]), True)
    external_loader = _tensor_loader(nums[1], cats[1], np.zeros(len(external), dtype=int), int(cfg["batch_size"]), False)
    pre_epochs = 1 if fast else int(cfg["pretrain_epochs"])
    pre_optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["pretrain_learning_rate"]),
                                      weight_decay=float(cfg["pretrain_weight_decay"]))
    for _ in range(pre_epochs):
        _pretrain_mfm_epoch(model, train_loader, pre_optimizer, cards, 0.30)
    positives = float(y.sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - positives) / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["finetune_learning_rate"]),
                                  weight_decay=float(cfg["finetune_weight_decay"]))
    for _ in range(max(1, int(epochs))):
        _supervised_epoch(model, train_loader, optimizer, loss_fn, "tabmfm")
    return _predict_torch(model, external_loader, "tabmfm"), {
        "status": "SUCCESS",
        "model_class": _model_class_name("tabmfm_pre030_colid1"),
        "parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "selected_epoch": int(epochs),
    }


def write_metadata(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
