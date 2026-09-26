from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CPU_FAMILIES = ["lr", "xgb", "lgbm", "catboost", "rf"]
NEURAL_FAMILIES = ["ftt", "tab", "tabpfn", "tabmfm_pre030_colid1",
                   "tabmfm_pre015_colid1", "tabmfm_pre030_colid0", "tabmfm_nopre_colid1"]


def numeric_differences(left, right, prefix=""):
    out = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys(): out += numeric_differences(left[key], right[key], f"{prefix}.{key}" if prefix else key)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        out.append((prefix, abs(float(left) - float(right))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only reproduction comparison; do not commit row-level inputs.")
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--candidate-external", type=Path)
    parser.add_argument("--reference-external", type=Path)
    parser.add_argument("--reference-results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "reference")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidate, reference = pd.read_csv(args.candidate_oof), pd.read_csv(args.reference_oof)
    if len(candidate) != len(reference): raise SystemExit("Row count mismatch")
    report = {"classical_cpu": {}, "neural_and_tabpfn_cpu": {}, "external": {}, "summary": {}}
    failed = False
    for column in [c for c in reference if c not in {"y", "fold"} and c in candidate]:
        difference = float(np.max(np.abs(candidate[column].to_numpy(float) - reference[column].to_numpy(float))))
        item = {"max_abs_diff": difference, "candidate_auroc": float(roc_auc_score(reference["y"], candidate[column])),
                "reference_auroc": float(roc_auc_score(reference["y"], reference[column]))}
        family = column.split("__")[0]
        if family in CPU_FAMILIES:
            report["classical_cpu"][column] = item
            failed |= difference > 1e-6
        elif family in NEURAL_FAMILIES:
            item["auroc_abs_diff"] = abs(item["candidate_auroc"] - item["reference_auroc"])
            report["neural_and_tabpfn_cpu"][column] = item
            failed |= item["auroc_abs_diff"] > 0.001
    if args.candidate_external or args.reference_external:
        if not args.candidate_external or not args.reference_external:
            raise SystemExit("Both external comparison paths are required")
        candidate_external = pd.read_csv(args.candidate_external)
        reference_external = pd.read_csv(args.reference_external)
        if len(candidate_external) != len(reference_external):
            raise SystemExit("External row count mismatch")
        outcome = reference_external["y_true"].to_numpy(int)
        for column in [c for c in reference_external if c != "y_true" and c in candidate_external]:
            candidate_probability = candidate_external[column].to_numpy(float)
            reference_probability = reference_external[column].to_numpy(float)
            candidate_auc = float(roc_auc_score(outcome, candidate_probability))
            reference_auc = float(roc_auc_score(outcome, reference_probability))
            item = {
                "max_abs_diff": float(np.max(np.abs(candidate_probability - reference_probability))),
                "candidate_auroc": candidate_auc,
                "reference_auroc": reference_auc,
                "auroc_abs_diff": abs(candidate_auc - reference_auc),
            }
            report["external"][column] = item
            family = column.split("__")[0]
            failed |= item["max_abs_diff"] > 1e-6 if family in CPU_FAMILIES else item["auroc_abs_diff"] > 0.001
    if args.candidate_results:
        for name in ["numbers.json", "numbers_A.json", "numbers_B.json", "sens_age.json", "strata_calibration.json"]:
            a, b = args.candidate_results / name, args.reference_results / name
            if a.exists() and b.exists():
                differences = numeric_differences(json.loads(a.read_text()), json.loads(b.read_text()))
                report["summary"][name] = {"compared_values": len(differences), "max_abs_diff": max((d for _, d in differences), default=0.0)}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("Reproduction gate failed: classical probability difference >1e-6 or neural AUROC difference >0.001")


if __name__ == "__main__": main()
