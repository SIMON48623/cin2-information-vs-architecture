"""Analysis B exactly as pre-specified in B_analysis_plan.md (sha256 1c13ec8b...)."""
import argparse, json, sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--oof", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--external", type=Path, required=True)
parser.add_argument("--external-data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
parser.add_argument("--external-sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
parser.add_argument("--fast", action="store_true")
B_ARGS = parser.parse_args(); B_ARGS.output.mkdir(parents=True, exist_ok=True)
saved_argv = sys.argv
sys.argv = [saved_argv[0], "--oof", str(B_ARGS.oof), "--external", str(B_ARGS.external),
            "--data", str(B_ARGS.data), "--sheet", str(B_ARGS.sheet), "--output", str(B_ARGS.output),
            "--bootstrap", str(B_ARGS.bootstrap)] + (["--fast"] if B_ARGS.fast else [])
source = (Path(__file__).resolve().parent / "10_main_analysis.py").read_text(encoding="utf-8")
exec(source.split("# ---------------------------------------------------------------- calibration of every configuration")[0], globals())
sys.argv = saved_argv

A = np.load(B_ARGS.output / "arrays.npz")
if B_ARGS.data.suffix.lower() in (".xlsx", ".xls"):
    df = pd.read_excel(B_ARGS.data, sheet_name=B_ARGS.sheet)
else:
    df = pd.read_csv(B_ARGS.data, comment="#")
lab = pd.to_numeric(df["pathology_group"], errors="coerce"); keep = lab.isin([0, 1])
ciba = pd.to_numeric(df.loc[keep, "pathology_fig"], errors="coerce").values.astype(int)
assert (lab[keep].astype(int).values == y).all()
neg, pos = ciba == 0, ciba == 1
R = {}
R["strata"] = dict(neg_n=int(neg.sum()), neg_events=int(y[neg].sum()), pos_n=int(pos.sum()), pos_events=int(y[pos].sum()))
R["clinician_alone"] = dict(sens=float(y[pos].sum() / y.sum()), spec=float(((~pos) & (y == 0)).sum() / (y == 0).sum()),
                            ppv=float(y[pos].mean()), npv=float(1 - y[neg].mean()))
FAMS = [("lr", "Logistic regression"), ("xgb", "XGBoost"), ("lgbm", "LightGBM"), ("catboost", "CatBoost"),
        ("rf", "Random forest"), ("ftt", "FT-Transformer"), ("tabpfn", "TabPFN"), ("tab", "Tabular Transformer"),
        ("tabmfm_pre030_colid1", "Tab-MFM (pretrained)"), ("tabmfm_nopre_colid1", "Tab-MFM (no pretraining)")]
FAMS = [(key, name) for key, name in FAMS
        if f"raw_{key}__full" in A.files and f"raw_{key}__nopf" in A.files]

# ---- B1
b1 = []
for key, name in FAMS:
    p = A[f"raw_{key}__nopf"]
    an, Sn = delong(y[neg], p[neg]); ap, Sp = delong(y[pos], p[pos])
    z = (an[0] - ap[0]) / np.sqrt(Sn[0, 0] + Sp[0, 0])
    b1.append(dict(family=name, auroc_neg=float(an[0]), neg_lo=float(an[0] - 1.96 * np.sqrt(Sn[0, 0])), neg_hi=float(an[0] + 1.96 * np.sqrt(Sn[0, 0])),
                   auroc_pos=float(ap[0]), pos_lo=float(ap[0] - 1.96 * np.sqrt(Sp[0, 0])), pos_hi=float(ap[0] + 1.96 * np.sqrt(Sp[0, 0])),
                   p_diff=float(2 * stats.norm.sf(abs(z)))))
R["B1"] = b1

# ---- B2: label-free workload rule inside each held-out fold
def rescue_flags(p, w):
    flag = np.zeros(len(y), bool)
    for k in np.unique(fold):
        idx = np.where((fold == k) & neg)[0]
        n_ref = int(round(w * len(idx)))
        top = idx[np.argsort(-p[idx], kind="mergesort")[:n_ref]]
        flag[top] = True
    return flag
def b2_stats(flag, ii=None):
    ii = np.arange(len(y)) if ii is None else ii
    yy, nn, ff, pp = y[ii], neg[ii], flag[ii], pos[ii]
    found = int((ff & nn & (yy == 1)).sum()); refs = int((ff & nn).sum()); negev = int((nn & (yy == 1)).sum())
    refer = pp | (ff & nn)
    return dict(found=found, referrals=refs, neg_events=negev,
                rescue_rate=found / negev if negev else np.nan,
                refs_per_case=refs / found if found else np.nan,
                pathway_sens=float((refer & (yy == 1)).sum() / (yy == 1).sum()),
                pathway_spec=float(((~refer) & (yy == 0)).sum() / (yy == 0).sum()))
b2 = {}
for key, name in FAMS:
    p = A[f"raw_{key}__nopf"]
    for w in (0.10, 0.20, 0.30):
        fl = rescue_flags(p, w); s = b2_stats(fl)
        s["random_expected_found"] = float(w * R["strata"]["neg_events"])
        bs = np.array([[b2_stats(fl, i)[m] for m in ("rescue_rate", "pathway_sens", "pathway_spec")] for i in BOOT_IDX])
        s["rescue_rate_ci"] = np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist()
        s["pathway_sens_ci"] = np.percentile(bs[:, 1], [2.5, 97.5]).tolist()
        s["pathway_spec_ci"] = np.percentile(bs[:, 2], [2.5, 97.5]).tolist()
        b2[f"{key}|{w:.2f}"] = s
R["B2"] = b2

# ---- B3
def stratum_cal(praw):
    praw = np.clip(praw, EPS, 1 - EPS); out = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = (fold == k) & neg; tr = (fold != k) & neg
        kind = choose_family(praw[tr], y[tr])
        out[te] = np.clip(fit_cal(kind, praw[tr], y[tr])(praw[te]), EPS, 1 - EPS)
    return out
def nb_rule(refer, yy):
    n = len(yy); return lambda t: ((refer & (yy == 1)).sum() / n) - ((refer & (yy == 0)).sum() / n) * (t / (1 - t))
PTS = np.round(np.arange(0.05, 0.4001, 0.01), 2)
def strategies(key, ii=None):
    ii = np.arange(len(y)) if ii is None else ii
    yy = y[ii]; ps = pos[ii]
    sc = SC[key][ii]; cn = A[f"cal_{key}__nopf"][ii]; cf = A[f"cal_{key}__full"][ii]
    out = {}
    for t in PTS:
        n = len(yy); f = lambda r: ((r & (yy == 1)).sum() / n) - ((r & (yy == 0)).sum() / n) * (t / (1 - t))
        out[t] = dict(S1=f(ps), S2=f(cn >= t), S3=f(ps | ((~ps) & (sc >= t))), S4=f(cf >= t),
                      all=float(yy.mean() - (1 - yy.mean()) * t / (1 - t)))
    return out
SC = {key: stratum_cal(A[f"raw_{key}__nopf"]) for key, _ in FAMS}
b3 = {"pt": PTS.tolist()}
full_lr = strategies("lr")
for s in ("S1", "S2", "S3", "S4", "all"):
    b3[s] = [float(full_lr[t][s]) for t in PTS]
contr = {}
for key, name in FAMS:
    for t in (0.10, 0.20, 0.30):
        pt = float(np.round(t, 2)); base = strategies(key)[pt]
        pairs = [("S3", "S1")] + ([("S3", "S4")] if key == "lr" else [])
        for a_, b_ in pairs:
            bs = []
            for i in BOOT_IDX:
                s = strategies(key, i)[pt]; bs.append(s[a_] - s[b_])
            contr[f"{key}|{a_}-{b_}|{pt:.2f}"] = dict(point=float(base[a_] - base[b_]),
                                                    lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
b3["contrasts"] = contr
R["B3"] = b3

# ---- B4 external, descriptive
ext = pd.read_csv(B_ARGS.external)
if B_ARGS.external_data.suffix.lower() in (".xlsx", ".xls"):
    exf = pd.read_excel(B_ARGS.external_data, sheet_name=B_ARGS.external_sheet)
else:
    exf = pd.read_csv(B_ARGS.external_data, comment="#")
external_outcome = "y_true" if "y_true" in exf.columns else "pathology_group"
assert (ext.y_true.values == pd.to_numeric(exf[external_outcome], errors="coerce").values.astype(int)).all()
cx = pd.to_numeric(exf.pathology_fig, errors="coerce").values.astype(int); yx = ext.y_true.values.astype(int); px = ext["lr__nopf"].values
b4 = dict(neg_n=int((cx == 0).sum()), neg_events=int(((cx == 0) & (yx == 1)).sum()),
          pos_n=int((cx == 1).sum()), pos_events=int(((cx == 1) & (yx == 1)).sum()),
          clinician_sens=float(((cx == 1) & (yx == 1)).sum() / yx.sum()),
          clinician_spec=float(((cx == 0) & (yx == 0)).sum() / (yx == 0).sum()))
idx = np.where(cx == 0)[0]
for w in (0.10, 0.20, 0.30):
    top = idx[np.argsort(-px[idx], kind="mergesort")[:int(round(w * len(idx)))]]
    b4[f"found_{w:.2f}"] = int(yx[top].sum()); b4[f"refs_{w:.2f}"] = int(len(top))
R["B4"] = b4
if B_ARGS.fast:
    R["mode"] = "FAST synthetic smoke test; estimates are not paper results"
with (B_ARGS.output / "numbers_B.json").open("w", encoding="utf-8") as handle:
    json.dump(R, handle, indent=1, default=float)
print("done")
