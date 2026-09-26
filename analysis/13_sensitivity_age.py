"""Sensitivity analysis that treats recorded ages above 100 as missing."""
import argparse, json, sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--oof", type=Path, required=True)
parser.add_argument("--external", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
parser.add_argument("--fast", action="store_true")
S_ARGS = parser.parse_args(); S_ARGS.output.mkdir(parents=True, exist_ok=True)
saved_argv = sys.argv
sys.argv = [saved_argv[0], "--oof", str(S_ARGS.oof), "--external", str(S_ARGS.external),
            "--data", str(S_ARGS.data), "--sheet", str(S_ARGS.sheet), "--output", str(S_ARGS.output),
            "--bootstrap", str(S_ARGS.bootstrap)] + (["--fast"] if S_ARGS.fast else [])
source = (Path(__file__).resolve().parent / "10_main_analysis.py").read_text(encoding="utf-8")
exec(source.split("# ---------------------------------------------------------------- calibration of every configuration")[0], globals())
raw_block = source.split("# ---------------------------------------------------------------- raw data")[1].split("rep = {}")[0]
exec("# " + raw_block, globals())
sys.argv = saved_argv

from scipy import stats
age = X["age"]
print("age>100:", int((age > 100).sum()))
trimmed = age[age <= 100]
print("Table 2 age: mean %.1f sd %.1f; Welch p %.3f" % (
    trimmed.mean(), trimmed.std(),
    stats.ttest_ind(trimmed[y[trimmed.index] == 0], trimmed[y[trimmed.index] == 1], equal_var=False).pvalue,
))
base = {feature_set: oof_pred("lr", feature_set) for feature_set in ("full", "nopf")}
X.loc[X["age"] > 100, "age"] = np.nan
fixed = {feature_set: oof_pred("lr", feature_set) for feature_set in ("full", "nopf")}
result = {feature_set: {
    "auroc_as_recorded": float(roc_auc_score(y, base[feature_set])),
    "auroc_set_missing": float(roc_auc_score(y, fixed[feature_set])),
    "max_abs_prob_change": float(np.abs(base[feature_set] - fixed[feature_set]).max()),
} for feature_set in base}
if S_ARGS.fast:
    result["mode"] = "FAST synthetic smoke test; estimates are not paper results"
with (S_ARGS.output / "sens_age.json").open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=1)
print(result)
