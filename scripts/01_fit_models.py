from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cin2.config import feature_sets, load_config
from cin2.cv import run_oof, write_metadata
from cin2.data import clean_features, load_table
from cin2.models import EXTRA_CONFIGS, PRIMARY_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_development.csv")
    parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    frame, y = load_table(args.data, sheet_name=args.sheet)
    frame = clean_features(frame, config)
    sets = feature_sets(config)
    combined = None
    metadata = {"mode": "fast" if args.fast else "full", "runs": {}}
    for name in ["full", "nopf"]:
        families = list(args.families or PRIMARY_FAMILIES)
        if name == "full" and args.families is None:
            families += [x for x in EXTRA_CONFIGS if x not in families]
        if name == "nopf":
            families = [family for family in families if family not in EXTRA_CONFIGS]
        run = run_oof(frame, y, sets[name], families, config, args.device, args.fast)
        produced = [c for c in run.predictions.columns if c not in {"y", "fold"}]
        renamed = run.predictions.rename(columns={family: f"{family}__{name}" for family in produced})
        if combined is None:
            combined = renamed
        else:
            if not combined[["y", "fold"]].equals(renamed[["y", "fold"]]):
                raise RuntimeError("Fold assignments changed between feature sets")
            for column in renamed.columns.difference(["y", "fold"]):
                combined[column] = renamed[column].to_numpy()
        metadata["runs"][name] = run.metadata
    combined.to_csv(args.output / "oof_raw_all.csv", index=False)
    write_metadata(args.output / "fit_models_metadata.json", metadata)


if __name__ == "__main__":
    main()
