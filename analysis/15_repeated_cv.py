from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def corrected_repeated_kfold_ttest(
    differences: np.ndarray,
    n_test_over_n_train: float,
) -> tuple[float, int, float]:
    differences = np.asarray(differences, dtype=float)
    if differences.size < 2:
        raise ValueError("At least two fold-level differences are required")
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    denominator = np.sqrt((1.0 / differences.size + n_test_over_n_train) * variance)
    degrees_freedom = int(differences.size - 1)
    if denominator == 0:
        statistic = 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
    else:
        statistic = mean / denominator
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return float(statistic), degrees_freedom, p_value


def holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def repeat_number(path: Path) -> int:
    try:
        return int(path.stem.split("r")[-1])
    except ValueError as error:
        raise ValueError(f"Cannot parse repeat number from {path.name}") from error


def load_repeats(input_dir: Path) -> list[tuple[int, pd.DataFrame, dict]]:
    paths = sorted(input_dir.glob("oof_r*.csv"), key=repeat_number)
    if not paths:
        raise FileNotFoundError(f"No oof_r*.csv files found in {input_dir}")
    numbers = [repeat_number(path) for path in paths]
    if numbers != list(range(1, len(numbers) + 1)):
        raise RuntimeError(f"Repeat files are not contiguous from 1: {numbers}")
    repeats = []
    for number, path in zip(numbers, paths):
        metadata_path = input_dir / f"metadata_r{number:02d}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata sidecar: {metadata_path.name}")
        repeats.append(
            (
                number,
                pd.read_csv(path),
                json.loads(metadata_path.read_text(encoding="utf-8")),
            )
        )
    return repeats


def make_run_summary(repeats: list[tuple[int, pd.DataFrame, dict]], configurations: list[str]) -> dict:
    metadata = [item[2] for item in repeats]
    packages = metadata[0]["packages"]
    weights_file = Path(metadata[0]["tabpfn_weights_file"]).name
    if any(item["packages"] != packages for item in metadata):
        raise RuntimeError("Package versions changed between repetitions")
    if any(Path(item["tabpfn_weights_file"]).name != weights_file for item in metadata):
        raise RuntimeError("TabPFN weights changed between repetitions")

    devices = {}
    for configuration in configurations:
        family = configuration.removesuffix("__full")
        devices[configuration] = (
            metadata[0]["devices"]["classical"]
            if family in {"lr", "rf", "xgb", "lgbm", "catboost"}
            else metadata[0]["devices"]["neural_and_tabpfn"]
        )

    epochs: dict[str, list[int]] = {}
    for item in metadata:
        for family, family_metadata in item["families"].items():
            selected = family_metadata.get("selected_epochs", [])
            if selected:
                epochs.setdefault(f"{family}__full", []).extend(int(value) for value in selected)

    return {
        "n_repeats": len(repeats),
        "n_splits": 5,
        "packages": packages,
        "devices_by_configuration": devices,
        "seeds": [item["seeds"] for item in metadata],
        "repeat_elapsed_seconds": {
            f"r{item['repeat']:02d}": item["elapsed_seconds"] for item in metadata
        },
        "neural_selected_epoch_ranges": {
            configuration: {"min": min(values), "max": max(values)}
            for configuration, values in epochs.items()
        },
        "tabpfn_weights_file": weights_file,
    }


def analyse(input_dir: Path, output_dir: Path) -> None:
    repeats = load_repeats(input_dir)
    configurations = [
        column for column in repeats[0][1].columns if column not in {"y", "fold"}
    ]
    if "lr__full" not in configurations:
        raise RuntimeError("The repeated OOF files do not contain lr__full")
    for number, frame, _ in repeats:
        current = [column for column in frame.columns if column not in {"y", "fold"}]
        if current != configurations:
            raise RuntimeError(f"Configuration columns changed in repeat {number}")
        if sorted(frame["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"Repeat {number} does not contain exactly five folds")

    pooled: dict[str, list[float]] = {configuration: [] for configuration in configurations}
    fold_differences: dict[str, list[float]] = {
        configuration: [] for configuration in configurations if configuration != "lr__full"
    }
    fold_ratios: list[float] = []

    for _, frame, _ in repeats:
        y = frame["y"].to_numpy(int)
        for configuration in configurations:
            pooled[configuration].append(
                float(roc_auc_score(y, frame[configuration].to_numpy(float)))
            )
        for fold in [1, 2, 3, 4, 5]:
            selected = frame["fold"].to_numpy(int) == fold
            fold_y = y[selected]
            if np.unique(fold_y).size != 2:
                raise RuntimeError("A validation fold contains only one outcome class")
            lr_auc = float(roc_auc_score(fold_y, frame.loc[selected, "lr__full"]))
            for configuration in fold_differences:
                auc = float(roc_auc_score(fold_y, frame.loc[selected, configuration]))
                fold_differences[configuration].append(auc - lr_auc)
            n_test = int(selected.sum())
            fold_ratios.append(n_test / (len(frame) - n_test))

    ratio = float(np.mean(fold_ratios))
    summary_rows = []
    raw_p_values = []
    for configuration in configurations:
        if configuration == "lr__full":
            continue
        pooled_difference = np.asarray(pooled[configuration]) - np.asarray(pooled["lr__full"])
        statistic, degrees_freedom, p_value = corrected_repeated_kfold_ttest(
            np.asarray(fold_differences[configuration]), ratio
        )
        summary_rows.append(
            {
                "configuration": configuration,
                "mean_delta_auroc": float(pooled_difference.mean()),
                "sd_delta_auroc": float(pooled_difference.std(ddof=1)),
                "min_delta_auroc": float(pooled_difference.min()),
                "max_delta_auroc": float(pooled_difference.max()),
                "n_delta_positive": int((pooled_difference > 0).sum()),
                "corrected_t": statistic,
                "df": degrees_freedom,
                "p_value": p_value,
                "mean_n_test_over_n_train": ratio,
            }
        )
        raw_p_values.append(p_value)
    adjusted = holm_adjust(raw_p_values)
    for row, value in zip(summary_rows, adjusted):
        row["p_holm"] = float(value)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_vs_lr.csv", index=False)
    pd.DataFrame(
        {
            "configuration": configurations,
            **{
                f"r{number:02d}": [pooled[configuration][index] for configuration in configurations]
                for index, (number, _, _) in enumerate(repeats)
            },
            "mean_auroc": [float(np.mean(pooled[configuration])) for configuration in configurations],
            "sd_auroc": [float(np.std(pooled[configuration], ddof=1)) for configuration in configurations],
        }
    ).to_csv(output_dir / "auroc_by_repeat.csv", index=False)
    run_summary = make_run_summary(repeats, configurations)
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise repeated five-fold CV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "repeated_cv")
    args = parser.parse_args()
    analyse(args.input, args.output)
    print("summary_vs_lr.csv written")
    print("auroc_by_repeat.csv written")
    print("run_summary.json written")


if __name__ == "__main__":
    main()
