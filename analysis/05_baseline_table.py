"""Generate article Table 2 from a development-cohort table.

Age excludes implausible values above 120 years and uses Welch's t test.
Gravidity and parity use Mann-Whitney U tests. Categorical variables use
chi-square tests (Yates correction for 2x2 tables); a 2x2 table with any
expected frequency below five uses Fisher's exact test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(path: Path, sheet):
    frame = pd.read_excel(path, sheet_name=sheet) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, comment="#")
    frame = frame.drop(columns=[c for c in frame.columns if c.lower() in {"patient_id", "patient_name", "name", "id"}], errors="ignore")
    outcome = pd.to_numeric(frame["pathology_group"], errors="coerce")
    keep = outcome.isin([0, 1])
    frame = frame.loc[keep].reset_index(drop=True)
    frame["pathology_group"] = outcome.loc[keep].astype(int).to_numpy()
    return frame


def categorical_test(values, outcome, yates=True):
    table = pd.crosstab(values, outcome)
    if table.shape == (2, 2):
        expected = stats.contingency.expected_freq(table.to_numpy())
        if (expected < 5).any():
            return stats.fisher_exact(table.to_numpy())[1], "Fisher"
        return stats.chi2_contingency(table.to_numpy(), correction=yates)[1], "chi2-Yates" if yates else "chi2"
    return stats.chi2_contingency(table.to_numpy(), correction=False)[1], "chi2"


def build_table(path: Path, sheet=0, yates=True):
    frame = load(path, sheet)
    outcome = frame["pathology_group"]
    rows = []

    def n_pct(series):
        return f"{int(series.sum())} ({100 * series.mean():.1f})"

    age = pd.to_numeric(frame["age"], errors="coerce").where(lambda x: x <= 120)
    age0, age1 = age[outcome == 0].dropna(), age[outcome == 1].dropna()
    rows.append([
        "Age, years, mean ± SD", f"{age.mean():.1f} ± {age.std():.1f}",
        f"{age0.mean():.1f} ± {age0.std():.1f}", f"{age1.mean():.1f} ± {age1.std():.1f}",
        stats.ttest_ind(age0, age1, equal_var=False).pvalue, "Welch",
    ])
    for column, label in (("gravidity", "Gravidity, median [IQR]"), ("parity", "Parity, median [IQR]")):
        values = pd.to_numeric(frame[column], errors="coerce")

        def median_iqr(series):
            series = series.dropna()
            return f"{series.median():.0f} [{series.quantile(.25):.0f}–{series.quantile(.75):.0f}]"

        rows.append([
            label, median_iqr(values), median_iqr(values[outcome == 0]), median_iqr(values[outcome == 1]),
            stats.mannwhitneyu(values[outcome == 0].dropna(), values[outcome == 1].dropna()).pvalue, "MWU",
        ])
    binary_rows = (
        ("menopausal_status", "Postmenopausal"), ("child_alive", "At least one living child"),
        ("HPV_overall", "High-risk HPV positive"), ("HPV16", "HPV16 positive"),
        ("HPV18", "HPV18 positive"), ("HPV_other_hr", "Other high-risk HPV positive"),
    )
    for column, label in binary_rows:
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        p_value, test = categorical_test(values[present], outcome[present], yates)
        rows.append([label, n_pct(values[present] == 1), n_pct(values[present & (outcome == 0)] == 1),
                     n_pct(values[present & (outcome == 1)] == 1), p_value, test])
    grouped_rows = (
        ("cytology_grade", "Cytology grade", {0: "NILM", 1: "ASC-US", 2: "ASC-H", 3: "LSIL", 4: "HSIL", 5: "AGC"}),
        ("colpo_impression", "Colposcopic impression", {1: "Mild", 2: "Moderate", 3: "Severe", 4: "Highly suspicious for CIN3+"}),
        ("TZ_type", "Transformation zone type", {1: "Type 1", 2: "Type 2", 3: "Type 3"}),
    )
    for column, label, levels in grouped_rows:
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        p_value, test = categorical_test(values[present], outcome[present])
        rows.append([label, "", "", "", p_value, test])
        for key, name in levels.items():
            rows.append([f"  {name}", n_pct(values[present] == key), n_pct(values[present & (outcome == 0)] == key),
                         n_pct(values[present & (outcome == 1)] == key), np.nan, ""])
    for column, label in (("iodine_negative", "Iodine test negative"),
                          ("atypical_vessels", "Atypical vessels present"),
                          ("pathology_fig", "Clinician image-based assessment positive")):
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        p_value, test = categorical_test(values[present], outcome[present], yates)
        rows.append([label, n_pct(values[present] == 1), n_pct(values[present & (outcome == 0)] == 1),
                     n_pct(values[present & (outcome == 1)] == 1), p_value, test])
    table = pd.DataFrame(rows, columns=["characteristic", "overall", "non_cin2", "cin2", "p", "test"])
    table.attrs["n"] = (len(frame), int((outcome == 0).sum()), int((outcome == 1).sum()))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-yates", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = build_table(args.data, args.sheet, not args.no_yates)
    table.to_csv(args.output / "table2_baseline.csv", index=False)
    summary = args.output / "table2_baseline_metadata.txt"
    summary.write_text(
        f"n_total={table.attrs['n'][0]}\nn_non_cin2={table.attrs['n'][1]}\nn_cin2={table.attrs['n'][2]}\n"
        f"mode={'FAST synthetic smoke test; not paper results' if args.fast else 'full protocol'}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
