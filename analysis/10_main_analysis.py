"""
Single source of truth for every number in the revised Frontiers manuscript.

Inputs
  rerun/rerun_outputs/oof_raw_all.csv    uncalibrated pooled OOF probabilities, 22 configurations
  rerun/rerun_outputs/external_raw.csv   frozen-model probabilities on the 103-case external cohort
  proj/.../data_modelA#U7ec8.xlsx        raw development data (identifiers dropped on load, never exported)

Conventions (fixed before computing)
  - Discrimination (AUROC, AUPRC) is computed on UNCALIBRATED pooled OOF predictions.
    Post-hoc calibration is monotone within a fold, and isotonic regression introduces ties,
    so computing AUROC after calibration would mix calibration choice into discrimination.
  - Probability quality (Brier, ECE, calibration intercept/slope), thresholds and DCA use
    NESTED-calibrated probabilities: for each held-out fold, the calibration family is chosen
    by inner CV on the remaining folds only, the calibrator is fitted on those folds only.
  - Paired uncertainty: DeLong for AUROC differences; patient-level paired bootstrap
    (2,000 resamples, seed 13) for all differences.
Outputs
  out/numbers.json, out/*.csv, out/arrays.npz
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--oof", type=Path, required=True)
parser.add_argument("--external", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--sheet", type=lambda value: int(value) if str(value).isdigit() else value, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--fast", action="store_true")
ARGS = parser.parse_args()
OUT = ARGS.output
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6
B = ARGS.bootstrap
SEED = 13
N = {}  # numbers.json

# ---------------------------------------------------------------- load
oof = pd.read_csv(ARGS.oof)
y = oof["y"].values.astype(int)
fold = oof["fold"].values.astype(int)
MODELS = [c for c in oof.columns if c not in ("y", "fold")]
ext = pd.read_csv(ARGS.external)
yx = ext["y_true"].values.astype(int)
N["n"] = int(len(y)); N["events"] = int(y.sum()); N["prevalence"] = float(y.mean())
N["ext_n"] = int(len(yx)); N["ext_events"] = int(yx.sum()); N["ext_prevalence"] = float(yx.mean())

# ---------------------------------------------------------------- helpers
def ece(yv, p, bins=10):
    e = np.linspace(0, 1, bins + 1); t = 0.0
    for i in range(bins):
        m = (p > e[i]) & (p <= e[i + 1]) if i else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            t += m.sum() / len(p) * abs(yv[m].mean() - p[m].mean())
    return t

def fit_cal(kind, p, yy):
    if kind == "sigmoid":
        m = LogisticRegression(C=1e10, max_iter=1000).fit(p.reshape(-1, 1), yy)
        return lambda q: m.predict_proba(q.reshape(-1, 1))[:, 1]
    m = IsotonicRegression(out_of_bounds="clip").fit(p, yy)
    return lambda q: m.predict(q)

def choose_family(ptr, ytr, seed=SEED):
    sc = {}
    for kind in ("sigmoid", "isotonic"):
        s = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(ptr.reshape(-1, 1), ytr):
            f = fit_cal(kind, ptr[a], ytr[a])
            s.append(brier_score_loss(ytr[b], np.clip(f(ptr[b]), EPS, 1 - EPS)))
        sc[kind] = float(np.mean(s))
    return min(sc, key=sc.get)

def nested_cal(praw):
    praw = np.clip(praw, EPS, 1 - EPS); out = np.empty(len(y)); picks = []
    for k in np.unique(fold):
        te = fold == k
        kind = choose_family(praw[~te], y[~te]); picks.append(kind[0])
        out[te] = np.clip(fit_cal(kind, praw[~te], y[~te])(praw[te]), EPS, 1 - EPS)
    return out, "".join(picks)

def legacy_cal(praw):
    """Previous procedure: fold-wise calibration, one family chosen by pooled-OOF Brier."""
    praw = np.clip(praw, EPS, 1 - EPS); res = {}
    for kind in ("sigmoid", "isotonic"):
        o = np.empty(len(y))
        for k in np.unique(fold):
            te = fold == k
            o[te] = np.clip(fit_cal(kind, praw[~te], y[~te])(praw[te]), EPS, 1 - EPS)
        res[kind] = o
    best = min(res, key=lambda k: brier_score_loss(y, res[k]))
    return res[best], best

def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))

def cal_slope_intercept(yv, p):
    x = logit(p)
    # slope: y ~ a + b x
    beta = np.zeros(2); X = np.c_[np.ones_like(x), x]
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta)); W = mu * (1 - mu)
        g = X.T @ (yv - mu); H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-10: break
    # calibration-in-the-large: y ~ a + offset(x)
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + x))); g = (yv - mu).sum(); h = (mu * (1 - mu)).sum()
        a += g / h
        if abs(g / h) < 1e-12: break
    return float(a), float(beta[1])

# DeLong (Sun & Xu 2014 fast algorithm)
def _midrank(x):
    J = np.argsort(x, kind="mergesort"); Z = x[J]; n = len(x); T = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(n); out[J] = T; return out

def delong(yv, preds):
    preds = np.atleast_2d(preds)
    pos, neg = preds[:, yv == 1], preds[:, yv == 0]; m, n = pos.shape[1], neg.shape[1]; k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    return auc, S

def auc_ci(yv, p):
    auc, S = delong(yv, p); se = np.sqrt(S[0, 0])
    return float(auc[0]), float(auc[0] - 1.959964 * se), float(auc[0] + 1.959964 * se)

def delong_diff(yv, p1, p2):
    auc, S = delong(yv, np.vstack([p1, p2])); L = np.array([1, -1.0]); v = L @ S @ L
    d = auc[0] - auc[1]
    return float(d), float(2 * stats.norm.sf(abs(d) / np.sqrt(v))) if v > 0 else 1.0

_rng = np.random.default_rng(SEED)
BOOT_IDX = []
while len(BOOT_IDX) < B:
    i = _rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(y): BOOT_IDX.append(i)
_rngx = np.random.default_rng(SEED)
BOOT_IDX_X = []
while len(BOOT_IDX_X) < B:
    i = _rngx.integers(0, len(yx), len(yx))
    if 0 < yx[i].sum() < len(yx): BOOT_IDX_X.append(i)

def boot_diff(yv, p1, p2, fn, idxs=None):
    idxs = BOOT_IDX if idxs is None else idxs
    d = np.array([fn(yv[i], p1[i]) - fn(yv[i], p2[i]) for i in idxs])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(fn(yv, p1) - fn(yv, p2)), float(lo), float(hi), float(p)

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.959964; ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)

# ---------------------------------------------------------------- calibration of every configuration
RAW = {m: np.clip(oof[m].values, 0, 1) for m in MODELS}
CAL, PICKS = {}, {}
for m in MODELS:
    CAL[m], PICKS[m] = nested_cal(RAW[m])

# ---------------------------------------------------------------- Table: performance of every configuration
rows = []
for m in MODELS:
    a, lo, hi = auc_ci(y, RAW[m])
    ci, cs = cal_slope_intercept(y, CAL[m])
    rows.append(dict(model=m, auroc=a, auroc_lo=lo, auroc_hi=hi,
                     auprc=float(average_precision_score(y, RAW[m])),
                     brier=float(brier_score_loss(y, CAL[m])), ece10=float(ece(y, CAL[m])),
                     cal_intercept=ci, cal_slope=cs, picks=PICKS[m],
                     auroc_after_nested_cal=float(roc_auc_score(y, CAL[m]))))
perf = pd.DataFrame(rows).sort_values("auroc", ascending=False)
perf.to_csv(OUT / "perf_all.csv", index=False)
N["perf"] = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in rows}

# ---------------------------------------------------------------- contrasts
def contrast(a, b, label):
    d, p_dl = delong_diff(y, RAW[a], RAW[b])
    _, lo, hi, p_bs = boot_diff(y, RAW[a], RAW[b], roc_auc_score)
    db, blo, bhi, bp = boot_diff(y, CAL[a], CAL[b], brier_score_loss)
    return dict(label=label, a=a, b=b, d_auroc=d, lo=lo, hi=hi, p_delong=p_dl, p_boot=p_bs,
                d_brier=db, brier_lo=blo, brier_hi=bhi, p_brier=bp)

FULL = ["catboost__full", "tabpfn__full", "xgb__full", "lgbm__full", "rf__full", "ftt__full",
        "tab__full", "tabmfm_pre030_colid1__full", "tabmfm_pre015_colid1__full",
        "tabmfm_pre030_colid0__full", "tabmfm_nopre_colid1__full"]
FULL = [model for model in FULL if model in RAW]
vs_lr = [contrast(m, "lr__full", f"{m} vs LR") for m in FULL]
adj = holm([c["p_delong"] for c in vs_lr])
for c, h in zip(vs_lr, adj): c["p_holm"] = float(h)
ablation_specs = [
    ("tabmfm_pre030_colid1__full", "tabmfm_nopre_colid1__full", "pretraining"),
    ("tabmfm_pre030_colid1__full", "tabmfm_pre030_colid0__full", "column identity"),
    ("tabmfm_pre030_colid1__full", "tabmfm_pre015_colid1__full", "mask ratio 0.30 vs 0.15"),
]
ablation = [contrast(a, b, label) for a, b, label in ablation_specs if a in RAW and b in RAW]
FAM = [("lr", "Logistic regression"), ("xgb", "XGBoost"), ("lgbm", "LightGBM"), ("catboost", "CatBoost"),
       ("rf", "Random forest"), ("ftt", "FT-Transformer"), ("tabpfn", "TabPFN"),
       ("tab", "Tabular Transformer"), ("tabmfm_pre030_colid1", "Tab-MFM (pretrained)"),
       ("tabmfm_nopre_colid1", "Tab-MFM (no pretraining)")]
pf = []
for key, name in FAM:
    if f"{key}__full" not in RAW or f"{key}__nopf" not in RAW:
        continue
    c = contrast(f"{key}__full", f"{key}__nopf", name)
    c["auroc_with"] = float(roc_auc_score(y, RAW[f"{key}__full"]))
    c["auroc_without"] = float(roc_auc_score(y, RAW[f"{key}__nopf"]))
    pf.append(c)
pd.DataFrame(vs_lr).to_csv(OUT / "contrasts_vs_lr.csv", index=False)
pd.DataFrame(ablation).to_csv(OUT / "contrasts_ablation.csv", index=False)
pd.DataFrame(pf).to_csv(OUT / "contrasts_pathfig.csv", index=False)
N["vs_lr"] = vs_lr; N["ablation"] = ablation; N["pathfig"] = pf

# weak-signal observation: does TabPFN's advantage grow when the variable is removed?
weak = {}
for comp in (("lr", "catboost") if "tabpfn__full" in RAW and "tabpfn__nopf" in RAW else ()):
    for fs in ("full", "nopf"):
        weak[f"tabpfn_vs_{comp}_{fs}"] = contrast(f"tabpfn__{fs}", f"{comp}__{fs}", f"TabPFN vs {comp} ({fs})")
    did = np.array([(roc_auc_score(y[i], RAW[f"tabpfn__nopf"][i]) - roc_auc_score(y[i], RAW[f"{comp}__nopf"][i]))
                    - (roc_auc_score(y[i], RAW[f"tabpfn__full"][i]) - roc_auc_score(y[i], RAW[f"{comp}__full"][i]))
                    for i in BOOT_IDX])
    point = (roc_auc_score(y, RAW["tabpfn__nopf"]) - roc_auc_score(y, RAW[f"{comp}__nopf"])) - \
            (roc_auc_score(y, RAW["tabpfn__full"]) - roc_auc_score(y, RAW[f"{comp}__full"]))
    weak[f"did_{comp}"] = dict(point=float(point), lo=float(np.percentile(did, 2.5)), hi=float(np.percentile(did, 97.5)),
                               p=float(min(1, 2 * min((did <= 0).mean(), (did >= 0).mean()))))
N["weak_signal"] = weak

# ---------------------------------------------------------------- calibration-selection optimism
opt = {}
for m in MODELS:
    leg, fam = legacy_cal(RAW[m])
    d, lo, hi, p = boot_diff(y, CAL[m], leg, brier_score_loss)
    opt[m] = dict(legacy_family=fam, brier_legacy=float(brier_score_loss(y, leg)),
                  brier_nested=float(brier_score_loss(y, CAL[m])), d=d, lo=lo, hi=hi, p=p,
                  ece_legacy=float(ece(y, leg)), ece_nested=float(ece(y, CAL[m])))
N["cal_optimism"] = opt

# ---------------------------------------------------------------- nested thresholds (selection never sees the held-out fold)
def nested_thresholds(praw, rule):
    praw = np.clip(praw, EPS, 1 - EPS); pred = np.zeros(len(y), int); ths = []
    for k in np.unique(fold):
        te = fold == k; tr_idx = np.where(~te)[0]
        kind = choose_family(praw[~te], y[~te])
        q = np.empty(len(tr_idx)); ftr = fold[~te]
        for j in np.unique(ftr):                       # inner calibrated probabilities on training folds only
            inj = ftr == j
            q[inj] = fit_cal(kind, praw[tr_idx][~inj], y[tr_idx][~inj])(praw[tr_idx][inj])
        ytr = y[tr_idx]; cand = np.unique(q)
        if rule == "youden":
            J = [((q >= t) & (ytr == 1)).sum() / ytr.sum() + ((q < t) & (ytr == 0)).sum() / (1 - ytr).sum() - 1 for t in cand]
            t = float(cand[int(np.argmax(J))])
        else:
            ok = [t for t in cand if ((q >= t) & (ytr == 1)).sum() / ytr.sum() >= 0.95]
            t = float(max(ok))
        ths.append(t)
        pk = np.clip(fit_cal(kind, praw[~te], y[~te])(praw[te]), EPS, 1 - EPS)
        pred[te] = (pk >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(thresholds=ths, thr_median=float(np.median(ths)), thr_min=float(min(ths)), thr_max=float(max(ths)),
                tp=tp, fp=fp, tn=tn, fn=fn,
                sens=tp / (tp + fn), sens_ci=wilson(tp, tp + fn), spec=tn / (tn + fp), spec_ci=wilson(tn, tn + fp),
                ppv=tp / (tp + fp), npv=tn / (tn + fn), flagged=(tp + fp) / len(y))
thr = {}
for m in [name for name in ("lr__full", "catboost__full", "tabpfn__full") if name in RAW]:
    for rule in ("youden", "sens95"):
        thr[f"{m}|{rule}"] = nested_thresholds(RAW[m], rule)
N["thresholds"] = thr

# ---------------------------------------------------------------- decision curves
def nb(yv, p, t):
    pr = p >= t; n = len(yv)
    return ((pr & (yv == 1)).sum() / n) - ((pr & (yv == 0)).sum() / n) * (t / (1 - t))
PTS = np.round(np.arange(0.05, 0.4001, 0.01), 2)
dca = {"pt": PTS.tolist(), "treat_all": [float(y.mean() - (1 - y.mean()) * t / (1 - t)) for t in PTS]}
for m in [name for name in ("lr__full", "catboost__full", "tabpfn__full") if name in CAL]:
    dca[m] = [float(nb(y, CAL[m], t)) for t in PTS]
dnb = {}
for m in [name for name in ("catboost__full", "tabpfn__full") if name in CAL]:
    for t in (0.10, 0.20, 0.30):
        d = np.array([nb(y[i], CAL[m][i], t) - nb(y[i], CAL["lr__full"][i], t) for i in BOOT_IDX])
        dnb[f"{m}|{t:.2f}"] = dict(point=float(nb(y, CAL[m], t) - nb(y, CAL["lr__full"], t)),
                                   lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
N["dca"] = dca; N["dnb_vs_lr"] = dnb

# ---------------------------------------------------------------- external cohort (frozen inference, discrimination only)
exto = {}
for c in [c for c in ext.columns if c != "y_true"]:
    a, lo, hi = auc_ci(yx, ext[c].values)
    exto[c] = dict(auroc=a, lo=lo, hi=hi, auprc=float(average_precision_score(yx, ext[c].values)))
if {"tabmfm_pre030_colid1__full", "tabmfm_pre030_colid1__nopf"} <= set(ext.columns):
    exto["pf_tabmfm"] = dict(zip(("d", "p"), delong_diff(yx, ext["tabmfm_pre030_colid1__full"].values, ext["tabmfm_pre030_colid1__nopf"].values)))
if {"lr__full", "lr__nopf"} <= set(ext.columns):
    exto["pf_lr"] = dict(zip(("d", "p"), delong_diff(yx, ext["lr__full"].values, ext["lr__nopf"].values)))
if {"tabmfm_pre030_colid1__full", "lr__full"} <= set(ext.columns):
    exto["tabmfm_vs_lr_full"] = dict(zip(("d", "p"), delong_diff(yx, ext["tabmfm_pre030_colid1__full"].values, ext["lr__full"].values)))
N["external"] = exto

# ---------------------------------------------------------------- raw data: missingness, parameters, reproduction, learning curves, univariable
if ARGS.data.suffix.lower() in (".xlsx", ".xls"):
    sheet = ARGS.sheet
    df = pd.read_excel(ARGS.data, sheet_name=sheet)
else:
    df = pd.read_csv(ARGS.data, comment="#")
N["screened_records"] = int(len(df))
lab = pd.to_numeric(df["pathology_group"], errors="coerce"); keep = lab.isin([0, 1])
df = df.loc[keep].drop(columns=[c for c in ("patient_id", "patient_name") if c in df.columns]).reset_index(drop=True)
yr = lab[keep].astype(int).values
FEAT = ["age", "menopausal_status", "gravidity", "parity", "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
        "cytology_grade", "colpo_impression", "TZ_type", "iodine_negative", "atypical_vessels", "child_alive", "pathology_fig"]
TOK = ["_", " ", "", "NA", "N/A", "na", "None", "none", "NULL", "null", "unknown", "UNKNOWN"]
X = df[FEAT].replace(TOK, np.nan)
for c in FEAT: X[c] = pd.to_numeric(X[c], errors="coerce")
assert len(yr) == len(y) and (yr == y).all(), "row order / labels differ from rerun OOF"
N["excluded_records"] = N["screened_records"] - len(yr)
N["label_order_matches_rerun"] = True
miss = X.isna().sum(); N["missing"] = {c: int(v) for c, v in miss.items() if v > 0}
N["n_candidate_variables_full"] = len(FEAT); N["n_candidate_variables_nopf"] = len(FEAT) - 1

# fold reproduction
myfold = np.zeros(len(y), int)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1): myfold[te] = k
N["fold_assignment_reproduced"] = bool((myfold == fold).all())

NUM = ["age", "gravidity", "parity"]
def pre(cols):
    cat = [c for c in cols if c not in NUM]
    return ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                              ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                              ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
COLS = {"full": FEAT, "nopf": [c for c in FEAT if c != "pathology_fig"]}
N["n_parameters_onehot"] = {fs: int(pre(COLS[fs]).fit_transform(X[COLS[fs]]).shape[1]) for fs in COLS}
N["events_per_variable_full"] = N["events"] / len(FEAT)
N["events_per_parameter_full"] = N["events"] / N["n_parameters_onehot"]["full"]

def mk(model):
    if model == "lr":
        return LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=13)
    return RandomForestClassifier(n_estimators=100, random_state=13)

def oof_pred(model, fs, frac=1.0, sub_seed=0):
    out = np.full(len(y), np.nan); rng = np.random.default_rng(sub_seed)
    for k in range(1, 6):
        te = fold == k; tr = np.where(~te)[0]
        if frac < 1.0:          # stratified subsample of the training fold
            keep_idx = []
            for cls in (0, 1):
                ids = tr[y[tr] == cls]; keep_idx.append(rng.choice(ids, int(round(len(ids) * frac)), replace=False))
            tr = np.sort(np.concatenate(keep_idx))
        P = pre(COLS[fs]); Xt = P.fit_transform(X.iloc[tr][COLS[fs]]); Xv = P.transform(X.loc[te, COLS[fs]])
        out[te] = mk(model).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]
    return out

rep = {}
for model in ("lr", "rf"):
    for fs in ("full", "nopf"):
        mine = oof_pred(model, fs)
        ref = RAW[f"{model}__{fs}"]
        rep[f"{model}__{fs}"] = dict(max_abs_diff=float(np.abs(mine - ref).max()),
                                     auroc_mine=float(roc_auc_score(y, mine)), auroc_rerun=float(roc_auc_score(y, ref)))
N["reproduction"] = rep

FR = [0.2, 0.4, 0.6, 0.8, 1.0]
lc = {}
for model in ("lr", "rf"):
    for fs in ("full", "nopf"):
        pts = []
        for f in FR:
            vals = [roc_auc_score(y, oof_pred(model, fs, f, s)) for s in (range(5) if f < 1 else [0])]
            pts.append(dict(frac=f, n_train=int(round(len(y) * 0.8 * f)), mean=float(np.mean(vals)), sd=float(np.std(vals))))
        lc[f"{model}__{fs}"] = pts
N["learning_curve"] = lc

# univariable (nested) discrimination of each candidate variable
uni = {}
for c in FEAT:
    sc = np.empty(len(y))
    for k in range(1, 6):
        te = fold == k; tr = ~te
        med = X.loc[tr, c].median(); xtr = X.loc[tr, c].fillna(med).values; xte = X.loc[te, c].fillna(med).values
        if len(np.unique(xtr)) <= 12:   # discrete: training-fold event rate per level
            rate = pd.Series(y[tr]).groupby(xtr).mean(); base = y[tr].mean()
            sc[te] = pd.Series(xte).map(rate).fillna(base).values
        else:                            # continuous: direction from training fold, training-fold percentile
            s = 1 if roc_auc_score(y[tr], xtr) >= 0.5 else -1; ref = np.sort(s * xtr)
            sc[te] = np.searchsorted(ref, s * xte, side="right") / len(ref)
    a, lo, hi = auc_ci(y, sc)
    uni[c] = dict(auroc=a, lo=lo, hi=hi, positive_n=int((X[c] == 1).sum()) if set(X[c].dropna().unique()) <= {0, 1} else None)
N["univariable"] = uni
lift = np.array([max(v["auroc"] - 0.5, 0) for v in uni.values()])
N["n_eff_15_nested"] = float(lift.sum() ** 2 / (lift ** 2).sum())
lift14 = np.array([max(uni[c]["auroc"] - 0.5, 0) for c in FEAT if c != "pathology_fig"])
N["n_eff_14_nested_nopf"] = float(lift14.sum() ** 2 / (lift14 ** 2).sum())

# baseline table cross-check against the original Table 2
chk = {"HPV_overall": 608, "HPV16": 168, "HPV18": 53, "HPV_other_hr": 1, "atypical_vessels": 2,
       "pathology_fig": 197, "iodine_negative": 289, "menopausal_status": 303, "child_alive": 786}
N["table2_check"] = {c: dict(original=v, data=int((X[c] == 1).sum())) for c, v in chk.items()}
N["pathfig_by_outcome"] = dict(nonevent=int(((X["pathology_fig"] == 1) & (y == 0)).sum()),
                                event=int(((X["pathology_fig"] == 1) & (y == 1)).sum()))
N["colpo_counts"] = {str(int(k)): [int(((X["colpo_impression"] == k) & (y == 0)).sum()), int(((X["colpo_impression"] == k) & (y == 1)).sum())]
                     for k in sorted(X["colpo_impression"].dropna().unique())}

np.savez(OUT / "arrays.npz", y=y, fold=fold, **{f"cal_{m}": CAL[m] for m in MODELS}, **{f"raw_{m}": RAW[m] for m in MODELS})
if ARGS.fast:
    N["mode"] = "FAST synthetic smoke test; estimates are not paper results"
with (OUT / "numbers.json").open("w", encoding="utf-8") as handle:
    json.dump(N, handle, indent=1, default=float)
print("done")
