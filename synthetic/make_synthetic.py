"""Create independent-marginal demonstration data from published summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SEED = 13


def exact_binary(n: int, proportion: float, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(n, dtype=int)
    values[: int(round(n * proportion))] = 1
    rng.shuffle(values)
    return values


def exact_category(n: int, probabilities: list[float], rng: np.random.Generator) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float) * n
    counts = np.floor(raw).astype(int)
    for index in np.argsort(raw - counts)[::-1][: n - counts.sum()]:
        counts[index] += 1
    values = np.concatenate([np.full(count, i) for i, count in enumerate(counts)])
    rng.shuffle(values)
    return values


def truncated_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(45.8, 12.2, n)
    while ((values < 20) | (values > 80)).any():
        mask = (values < 20) | (values > 80)
        values[mask] = rng.normal(45.8, 12.2, int(mask.sum()))
    return np.round(values, 1)


def make_frame(n: int, prevalence: float, rng: np.random.Generator, missing: bool) -> pd.DataFrame:
    age = truncated_normal(n, rng)
    gravidity = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.03, 0.15, 0.38, 0.27, 0.12, 0.05])
    parity = rng.choice([0, 1, 2, 3], n, p=[0.07, 0.55, 0.32, 0.06])
    parity = np.minimum(parity, gravidity)
    hpv = exact_binary(n, 0.692, rng)
    hpv_positive = np.flatnonzero(hpv)
    hpv16 = np.zeros(n, dtype=int)
    hpv18 = np.zeros(n, dtype=int)
    hpv_other = np.zeros(n, dtype=int)
    hpv16[rng.choice(hpv_positive, min(len(hpv_positive), round(n * 0.191)), replace=False)] = 1
    remaining = hpv_positive[hpv16[hpv_positive] == 0]
    hpv18[rng.choice(remaining, min(len(remaining), round(n * 0.060)), replace=False)] = 1
    remaining = hpv_positive[(hpv16[hpv_positive] == 0) & (hpv18[hpv_positive] == 0)]
    if len(remaining):
        hpv_other[rng.choice(remaining, max(1, round(n * 0.001)), replace=False)] = 1
    frame = pd.DataFrame({
        "age": age,
        "gravidity": gravidity.astype(float),
        "parity": parity.astype(float),
        "HPV_overall": hpv,
        "HPV16": hpv16,
        "HPV18": hpv18,
        "HPV_other_hr": hpv_other,
        "iodine_negative": exact_binary(n, 0.329, rng),
        "atypical_vessels": exact_binary(n, 0.002, rng),
        "child_alive": exact_binary(n, 0.894, rng),
        "menopausal_status": exact_binary(n, 0.345, rng),
        "pathology_fig": exact_binary(n, 0.224, rng),
        "cytology_grade": exact_category(n, [0.312, 0.488, 0.023, 0.143, 0.030, 0.005], rng),
        "colpo_impression": exact_category(n, [0.271, 0.611, 0.009, 0.109], rng) + 1,
        "TZ_type": exact_category(n, [0.279, 0.250, 0.471], rng) + 1,
    })

    # Coefficients are human-selected and were not estimated from patient data.
    linear = (
        0.018 * (frame["age"] - 45.8)
        + 0.45 * frame["HPV16"] + 0.25 * frame["HPV18"]
        + 0.18 * frame["HPV_overall"] + 0.22 * frame["iodine_negative"]
        + 0.20 * (frame["cytology_grade"] >= 3)
        + 0.25 * (frame["colpo_impression"] >= 2)
        + 1.10 * frame["pathology_fig"]
    ).to_numpy(float)
    uniforms = rng.random(n)
    target = int(round(n * prevalence))
    low, high = -10.0, 10.0
    for _ in range(100):
        intercept = (low + high) / 2
        count = int((uniforms < 1 / (1 + np.exp(-(linear + intercept)))).sum())
        if count < target:
            low = intercept
        else:
            high = intercept
    probabilities = 1 / (1 + np.exp(-(linear + high)))
    frame["pathology_group"] = (uniforms < probabilities).astype(int)

    if missing:
        for column, count in {"age": 3, "gravidity": 3, "parity": 2, "iodine_negative": 1}.items():
            frame.loc[rng.choice(n, count, replace=False), column] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    make_frame(879, 242 / 879, rng, True).to_csv(args.output_dir / "synthetic_development.csv", index=False)
    make_frame(103, 34 / 103, rng, False).to_csv(args.output_dir / "synthetic_external.csv", index=False)


if __name__ == "__main__":
    main()

