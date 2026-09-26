"""Observed-to-expected CIN2+ counts within clinician strata, from nested-calibrated OOF predictions (out/arrays.npz)."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
parser.add_argument("--fast", action="store_true")
ARGS = parser.parse_args(); ARGS.output.mkdir(parents=True, exist_ok=True)
A = np.load(ARGS.output / "arrays.npz"); y = A["y"].astype(int)
if ARGS.data.suffix.lower() in (".xlsx", ".xls"):
    df = pd.read_excel(ARGS.data, sheet_name=ARGS.sheet)
else:
    df = pd.read_csv(ARGS.data, comment="#")
lab = pd.to_numeric(df["pathology_group"], errors="coerce"); keep = lab.isin([0, 1])
assert (lab[keep].astype(int).values == y).all()
ciba = pd.to_numeric(df.loc[keep, "pathology_fig"], errors="coerce").values.astype(int)
out = {}
for m in [name for name in ("lr__full", "catboost__full", "tabpfn__full") if f"cal_{name}" in A.files]:
    p = A[f"cal_{m}"]; out[m] = {}
    for name, mask in (("negative", ciba == 0), ("positive", ciba == 1), ("all", np.ones_like(y, bool))):
        o, e = int(y[mask].sum()), float(p[mask].sum())
        out[m][name] = {"observed": o, "expected": e, "oe": o / e}
if ARGS.fast:
    out["mode"] = "FAST synthetic smoke test; estimates are not paper results"
with (ARGS.output / "strata_calibration.json").open("w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=1)
print(json.dumps({m: {k: round(v["oe"], 4) for k, v in d.items()}
                  for m, d in out.items() if isinstance(d, dict)}))
