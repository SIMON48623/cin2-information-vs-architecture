"""Frozen external evaluation matching the article pipeline.

Only logistic regression and the pretrained Tab-MFM were frozen on the full
development cohort in the original analysis. Tab-MFM uses the rounded median
of the five OOF-selected epochs; no external outcome is used during fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cin2.config import feature_sets, load_config
from cin2.cv import fit_final_lr_predict, fit_final_tabmfm_predict, write_metadata
from cin2.data import clean_features, load_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_development.csv")
    parser.add_argument("--external", type=Path, default=ROOT / "data" / "synthetic_external.csv")
    parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
    parser.add_argument("--external-sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    train, y = load_table(args.data, sheet_name=args.sheet)
    external, y_external = load_table(args.external, sheet_name=args.external_sheet)
    train, external = clean_features(train, config), clean_features(external, config)
    fit_metadata_path = args.output / "fit_models_metadata.json"
    if not fit_metadata_path.is_file():
        raise FileNotFoundError("Run 01_fit_models.py first; selected OOF epochs are required")
    fit_metadata = json.loads(fit_metadata_path.read_text(encoding="utf-8"))
    result = pd.DataFrame({"y_true": y_external})
    metadata = {
        "mode": "FAST: real estimators with shortened training; not paper results" if args.fast else "full protocol",
        "device": args.device,
        "models": {},
    }
    for feature_set, columns in feature_sets(config).items():
        lr_predictions, lr_metadata = fit_final_lr_predict(train, y, external, columns, config)
        result[f"lr__{feature_set}"] = lr_predictions
        metadata["models"][f"lr__{feature_set}"] = lr_metadata
        family_record = fit_metadata["runs"][feature_set]["families"]["tabmfm_pre030_colid1"]
        selected = family_record.get("selected_epochs")
        if not selected:
            raise RuntimeError(f"Missing OOF-selected epochs for tabmfm_pre030_colid1__{feature_set}")
        epochs = max(1, int(np.rint(np.median(np.asarray(selected, dtype=float)))))
        predictions, model_metadata = fit_final_tabmfm_predict(
            train, y, external, columns, config, epochs, args.device, args.fast
        )
        result[f"tabmfm_pre030_colid1__{feature_set}"] = predictions
        metadata["models"][f"tabmfm_pre030_colid1__{feature_set}"] = model_metadata
    result.to_csv(args.output / "external_raw.csv", index=False)
    write_metadata(args.output / "external_metadata.json", metadata)


if __name__ == "__main__":
    main()
