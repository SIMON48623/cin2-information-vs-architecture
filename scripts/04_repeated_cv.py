from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cin2.config import feature_sets, load_config
from cin2.cv import CVResult, run_oof
from cin2.data import clean_features, load_table
from cin2.models import CLASSICAL, EXTRA_CONFIGS, PRIMARY_FAMILIES


ALL_FAMILIES = list(PRIMARY_FAMILIES) + [
    family for family in EXTRA_CONFIGS if family not in PRIMARY_FAMILIES
]
CLASSICAL_FAMILIES = [family for family in ALL_FAMILIES if family in CLASSICAL]
NEURAL_FAMILIES = [family for family in ALL_FAMILIES if family not in CLASSICAL]
PACKAGE_NAMES = ["torch", "tabpfn", "scikit-learn", "xgboost", "lightgbm", "catboost"]


def config_for_repeat(base_config: dict, repeat: int) -> dict:
    config = copy.deepcopy(base_config)
    fold_seed = 1000 + repeat
    model_seed = 2000 + repeat
    config["cross_validation"]["random_state"] = fold_seed
    config["cross_validation"]["inner_random_state"] = model_seed
    config["seeds"]["model"] = model_seed
    for family in ["lr", "rf", "xgb", "lgbm"]:
        config["models"][family]["random_state"] = model_seed
    config["models"]["catboost"]["random_seed"] = model_seed
    config["models"]["tabpfn"]["random_state"] = model_seed
    return config


def combine_predictions(classical: pd.DataFrame, neural: pd.DataFrame) -> pd.DataFrame:
    if not classical[["y", "fold"]].equals(neural[["y", "fold"]]):
        raise RuntimeError("Classical and neural y and fold columns are not identical")
    combined = classical[["y", "fold"]].copy()
    for frame in [classical, neural]:
        for column in frame.columns:
            if column not in {"y", "fold"}:
                combined[f"{column}__full"] = frame[column].to_numpy()
    return combined


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def resolve_neural_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def require_local_tabpfn_weights() -> Path:
    value = os.environ.get("TABPFN_MODEL_PATH")
    if not value:
        raise RuntimeError("TABPFN_MODEL_PATH is not set")
    path = Path(value).expanduser()
    if not path.is_file():
        raise RuntimeError("TABPFN_MODEL_PATH does not point to a local file")
    return path


def run_quietly(*args, **kwargs) -> CVResult:
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            return run_oof(*args, **kwargs)


def assert_success(result: CVResult, requested_families: list[str]) -> None:
    for family in requested_families:
        status = result.metadata["families"].get(family, {}).get("status")
        if status != "SUCCESS":
            raise RuntimeError(f"{family} did not finish successfully: {status}")


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_sheet(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated five-fold CV sensitivity analysis")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--sheet", type=parse_sheet, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "repeated_cv")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--neural-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    data_path = args.data
    if args.synthetic:
        data_path = data_path or ROOT / "data" / "synthetic_development.csv"
    elif data_path is None:
        parser.error("--data is required unless --synthetic is used")

    weights = require_local_tabpfn_weights()
    neural_device = resolve_neural_device(args.neural_device)
    args.output.mkdir(parents=True, exist_ok=True)
    base_config = load_config()
    frame, y = load_table(data_path, sheet_name=args.sheet)
    frame = clean_features(frame, base_config)
    if not args.synthetic and (len(y) != 879 or int(y.sum()) != 242):
        raise RuntimeError(
            f"Development cohort check failed: n={len(y)}, CIN2+={int(y.sum())}"
        )
    columns = feature_sets(base_config)["full"]
    versions = package_versions()

    for repeat in range(1, args.repeats + 1):
        oof_path = args.output / f"oof_r{repeat:02d}.csv"
        metadata_path = args.output / f"metadata_r{repeat:02d}.json"
        if oof_path.exists():
            if not metadata_path.exists():
                raise RuntimeError(f"Existing {oof_path.name} has no metadata sidecar")
            print(f"repeat {repeat}/{args.repeats}: existing output skipped", flush=True)
            continue

        print(f"repeat {repeat}/{args.repeats}: started", flush=True)
        started = time.perf_counter()
        config = config_for_repeat(base_config, repeat)
        classical = run_quietly(
            frame,
            y,
            columns,
            CLASSICAL_FAMILIES,
            config,
            device="cpu",
            fast=args.fast,
        )
        assert_success(classical, CLASSICAL_FAMILIES)
        neural = run_quietly(
            frame,
            y,
            columns,
            NEURAL_FAMILIES,
            config,
            device=neural_device,
            fast=args.fast,
        )
        assert_success(neural, NEURAL_FAMILIES)
        combined = combine_predictions(classical.predictions, neural.predictions)
        expected = [f"{family}__full" for family in ALL_FAMILIES]
        if [column for column in combined if column not in {"y", "fold"}] != expected:
            raise RuntimeError("Combined OOF columns do not match the twelve configured models")

        elapsed = time.perf_counter() - started
        metadata = {
            "repeat": repeat,
            "mode": "fast" if args.fast else "full",
            "elapsed_seconds": elapsed,
            "packages": versions,
            "devices": {"classical": "cpu", "neural_and_tabpfn": neural_device},
            "seeds": {
                "fold": 1000 + repeat,
                "inner": 2000 + repeat,
                "model": 2000 + repeat,
            },
            "families": {
                **classical.metadata["families"],
                **neural.metadata["families"],
            },
            "tabpfn_weights_file": weights.name,
        }
        write_json_atomic(metadata_path, metadata)
        write_csv_atomic(oof_path, combined)
        print(f"repeat {repeat}/{args.repeats}: completed in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
