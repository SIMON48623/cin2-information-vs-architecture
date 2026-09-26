"""Analysis A: exact variable-group Shapley attribution of discrimination across ten model families.

Implements the downstream plan written into analysisA_task_brief.md before any subset model was run:
  v(S) = uncalibrated pooled OOF AUROC(S) - 0.5, v(empty) = 0; exact Shapley per family;
  2,000 patient-level bootstrap resamples (seed 13) of the fixed OOF predictions, recomputing all v(S);
  Kendall's W of the group rankings across the ten families with a 10,000-permutation test (seed 13);
  whether all families rank the same group first; range of each group's share across families.
Items marked 'descriptive' below were not pre-specified and are reported as such.
"""
import argparse, json, math, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

parser = argparse.ArgumentParser()
parser.add_argument("--groups", type=Path, required=True)
parser.add_argument("--oof", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--permutations", type=int, default=10000)
parser.add_argument("--fast", action="store_true")
ARGS = parser.parse_args(); ARGS.output.mkdir(parents=True, exist_ok=True)
SEED, B, NPERM = 13, ARGS.bootstrap, ARGS.permutations
A = pd.read_csv(ARGS.groups)
R = pd.read_csv(ARGS.oof)
y = A["y"].to_numpy().astype(int)
assert len(y) == 879 and y.sum() == 242 and (A["fold"].to_numpy() == R["fold"].to_numpy()).all() and (y == R["y"].to_numpy()).all()

REQUESTED_FAMS = ["lr", "xgb", "lgbm", "catboost", "rf", "ftt", "tabpfn", "tab", "tabmfm_pre030_colid1", "tabmfm_nopre_colid1"]
GROUPS = ["G1", "G2", "G3", "G4", "G5"]
SUBS = [format(i, "05b") for i in range(1, 32)]            # leftmost bit = G1
FAMS = [family for family in REQUESTED_FAMS if all(f"{family}__{subset}" in A.columns for subset in SUBS)]
if not FAMS:
    raise RuntimeError("No model family has all 31 subset predictions")
NG = 5

# ---- Shapley weight matrix: phi = M @ v, v indexed by integer code 0..31 (bit for G1 is the most significant)
def members(code):  # code int -> set of group indices
    s = format(code, "05b"); return {g for g in range(NG) if s[g] == "1"}
def code_of(gs): return int("".join("1" if g in gs else "0" for g in range(NG)), 2)
M = np.zeros((NG, 32))
for g in range(NG):
    others = [h for h in range(NG) if h != g]
    for k in range(len(others) + 1):
        w = math.factorial(k) * math.factorial(NG - k - 1) / math.factorial(NG)
        for S in itertools.combinations(others, k):
            M[g, code_of(set(S) | {g})] += w
            M[g, code_of(set(S))] -= w
assert np.allclose(M.sum(0)[1:31], 0) and np.allclose(M.sum(0)[31], 1)   # efficiency: sum(phi) = v(N)

cols = [f"{f}__{s}" for f in FAMS for s in SUBS]
P = A[cols].to_numpy()                                     # 879 x 310

def aucs(yv, Pm):
    """Rank-based AUROC (midranks for ties) for every column of Pm."""
    r = rankdata(Pm, axis=0)                               # average ranks handle ties
    m = yv.sum(); n = len(yv) - m
    return (r[yv == 1].sum(0) - m * (m + 1) / 2) / (m * n)

def shapley_from_auc(a):                                   # a: (..., 310) -> (..., 10 fams, 5 groups), v(N)
    a = a.reshape(a.shape[:-1] + (len(FAMS), 31))
    v = np.concatenate([np.zeros(a.shape[:-1] + (1,)), a - 0.5], axis=-1)   # index 0 = empty set
    return v @ M.T, v[..., 31]

a_obs = aucs(y, P)
# sanity: agrees with sklearn-free check against the rerun columns for the CPU families
phi, vN = shapley_from_auc(a_obs)
assert np.allclose(phi.sum(-1), vN)

# ---- bootstrap (same generator and filter as analysis.py)
rng = np.random.default_rng(SEED); IDX = []
while len(IDX) < B:
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(y): IDX.append(i)
phi_b = np.empty((B, len(FAMS), NG)); vN_b = np.empty((B, len(FAMS)))
for b, i in enumerate(IDX):
    phi_b[b], vN_b[b] = shapley_from_auc(aucs(y[i], P[i]))
share_b = phi_b / vN_b[..., None]

# ---- Kendall's W (rank 1 = largest Shapley value)
def ranks_desc(x): return rankdata(-x, axis=-1)             # average ranks for ties
def kendall_w(rk):                                          # rk: (m raters, n objects)
    m, n = rk.shape; Rj = rk.sum(0); S = ((Rj - Rj.mean()) ** 2).sum()
    T = sum(((c ** 3 - c).sum() for c in (np.unique(r, return_counts=True)[1] for r in rk)))
    return 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
rk_obs = ranks_desc(phi)
W_obs = kendall_w(rk_obs)
prng = np.random.default_rng(SEED)
perm_keys = prng.random((NPERM, len(FAMS), NG))
rk_perm = np.take_along_axis(np.broadcast_to(rk_obs, perm_keys.shape), np.argsort(perm_keys, axis=-1), axis=-1)
Rj = rk_perm.sum(1); Sp = ((Rj - Rj.mean(-1, keepdims=True)) ** 2).sum(-1)
W_perm = 12 * Sp / (len(FAMS) ** 2 * (NG ** 3 - NG))        # no ties in observed ranks (checked below)
assert all(len(set(r)) == NG for r in rk_obs.tolist())
p_perm = (1 + (W_perm >= W_obs - 1e-12).sum()) / (NPERM + 1)

top = [GROUPS[int(np.argmax(p))] for p in phi]
same_top = len(set(top)) == 1

# ---- descriptive (not pre-specified): bootstrap stability of the ranking
W_b = np.array([kendall_w(ranks_desc(p)) for p in phi_b])
top_b = phi_b.argmax(-1)                                    # (B, fams)
top_all_same_b = (top_b == top_b[:, :1]).all(1)
modal = int(np.bincount(np.argmax(phi, -1)).argmax())
p_first_by_family = (top_b == modal).mean(0)
p_all_first = (top_b == modal).all(1).mean()
# pairwise difference between the first- and second-ranked groups (by mean rank), per family
order_by_meanrank = np.argsort(rk_obs.mean(0))
g1, g2 = int(order_by_meanrank[0]), int(order_by_meanrank[1])
diff12 = phi[:, g1] - phi[:, g2]; diff12_b = phi_b[:, :, g1] - phi_b[:, :, g2]

def pci(x): lo, hi = np.percentile(x, [2.5, 97.5], axis=0); return lo, hi
phi_lo, phi_hi = pci(phi_b); sh_lo, sh_hi = pci(share_b); d_lo, d_hi = pci(diff12_b)
share = phi / vN[:, None]

out = {"families": FAMS, "groups": GROUPS, "B": B, "n_perm": NPERM, "seed": SEED,
       "auc_subsets": {c: float(a) for c, a in zip(cols, a_obs)},
       "vN": {f: float(v) for f, v in zip(FAMS, vN)},
       "phi": {f: {g: {"est": float(phi[i, j]), "lo": float(phi_lo[i, j]), "hi": float(phi_hi[i, j]),
                       "share": float(share[i, j]), "share_lo": float(sh_lo[i, j]), "share_hi": float(sh_hi[i, j]),
                       "rank": float(rk_obs[i, j])} for j, g in enumerate(GROUPS)} for i, f in enumerate(FAMS)},
       "kendall_W": float(W_obs), "kendall_p_perm": float(p_perm), "perm_W_max": float(W_perm.max()),
       "top_group_by_family": dict(zip(FAMS, top)), "same_top_all": same_top,
       "share_range": {g: [float(share[:, j].min()), float(share[:, j].max())] for j, g in enumerate(GROUPS)},
       "phi_range": {g: [float(phi[:, j].min()), float(phi[:, j].max())] for j, g in enumerate(GROUPS)},
       "mean_rank": {g: float(rk_obs[:, j].mean()) for j, g in enumerate(GROUPS)},
       "descriptive": {"W_boot_ci": [float(x) for x in np.percentile(W_b, [2.5, 97.5])],
                       "p_modal_first_by_family": {f: float(v) for f, v in zip(FAMS, p_first_by_family)},
                       "p_modal_first_all": float(p_all_first), "p_same_top_all_boot": float(top_all_same_b.mean()),
                       "first_vs_second": {"first": GROUPS[g1], "second": GROUPS[g2],
                                           **{f: {"d": float(diff12[i]), "lo": float(d_lo[i]), "hi": float(d_hi[i])} for i, f in enumerate(FAMS)}}},
       "rerun_vs_phaseA_full_auc": {f: {"phaseA": float(a_obs[cols.index(f"{f}__11111")]),
                                        "rerun": float(aucs(y, R[[f"{f}__full"]].to_numpy())[0])} for f in FAMS}}
if ARGS.fast:
    out["mode"] = "FAST synthetic smoke test; estimates are not paper results"
with (ARGS.output / "numbers_A.json").open("w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=1)
np.savez_compressed(ARGS.output / "arrays_A.npz", phi=phi, phi_b=phi_b, share_b=share_b, W_b=W_b, W_perm=W_perm)
pd.DataFrame([
    {"family": family, "group": group, "shapley": float(phi[i, j]), "share": float(share[i, j]),
     "lo": float(phi_lo[i, j]), "hi": float(phi_hi[i, j])}
    for i, family in enumerate(FAMS) for j, group in enumerate(GROUPS)
]).to_csv(ARGS.output / "group_shapley.csv", index=False)

pd.set_option("display.width", 200)
print(pd.DataFrame(phi, index=FAMS, columns=GROUPS).round(4))
print(pd.DataFrame(share, index=FAMS, columns=GROUPS).round(3))
print(pd.DataFrame(rk_obs, index=FAMS, columns=GROUPS))
print("W", round(W_obs, 4), "p", p_perm, "top", top, "same", same_top)
print("desc", json.dumps(out["descriptive"], indent=0)[:1500])
