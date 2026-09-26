import argparse, json, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fast", action="store_true")
ARGS = parser.parse_args(); ARGS.output.mkdir(parents=True, exist_ok=True)
N = json.loads((ARGS.results / "numbers.json").read_text(encoding="utf-8"))
R = json.loads((ARGS.results / "numbers_B.json").read_text(encoding="utf-8"))
A = np.load(ARGS.results / "arrays.npz")
y = A["y"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GREY, LIGHT = "#0b0b0b", "#52514e", "#8A8F98", "#d4d4d0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#999999",
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
def save(fig, name):
    fig.savefig(ARGS.output / f"{name}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(ARGS.output / f"{name}.pdf", bbox_inches="tight", facecolor="white"); plt.close(fig)
def clean(ax, left=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if not left: ax.spines["left"].set_visible(False)
def ptxt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

NAMES = {"lr": "Logistic regression", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost", "rf": "Random forest",
         "ftt": "FT-Transformer", "tabpfn": "TabPFN", "tab": "Tabular Transformer",
         "tabmfm_pre030_colid1": "Tab-MFM, pretrained", "tabmfm_nopre_colid1": "Tab-MFM, no pretraining",
         "tabmfm_pre015_colid1": "Tab-MFM, pretrained, mask 0.15", "tabmfm_pre030_colid0": "Tab-MFM, no column identity"}

# ------------------------------------------------ Figure 1: cohort flow
fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.set_xlim(-2, 100); ax.set_ylim(0, 70); ax.axis("off")
def box(x, y0, w, h, txt, fc="white", ec="#555555", bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    lines = txt.split("\n")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y0 + h / 2 + (len(lines) - 1) * 1.9 - i * 3.8, l, ha="center", va="center",
                fontsize=8.2, color=INK, fontweight="bold" if (i == 0 and bold_first) else "normal")
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9, shrinkA=0, shrinkB=0))
box(4, 57, 44, 10, "Development dataset\nRecords screened, n = 887")
box(56, 55, 40, 14, "Excluded, n = 8\nMissing or invalid\nhistopathology label", fc="#f6f6f4")
arrow(48.8, 62, 55.2, 62)
box(4, 38, 44, 12, "Development cohort, n = 879\nCIN2+ 242 (27.5%)\nNon-CIN2+ 637", fc="#eef4fc", ec=BLUE)
arrow(26, 56.2, 26, 50.8)
box(-1, 13, 26, 15, "Clinician assessment\npositive, n = 197\nCIN2+ 122", fc="white")
box(28, 13, 26, 15, "Clinician assessment\nnegative, n = 682\nCIN2+ 120", fc="white")
arrow(18, 37.2, 12, 28.8); arrow(34, 37.2, 41, 28.8)
box(60, 34, 36, 16, "External cohort, n = 103\nTwo independent institutions\nCIN2+ 34 (33.0%)\nFrozen-model evaluation", fc="#f6f6f4")
ax.text(26, 6.5, "Stratification used in the complementarity analysis", ha="center", fontsize=7.8, color=INK2, style="italic")
save(fig, "Figure1_cohort")

# ------------------------------------------------ Figure 2: effect sizes
vs = sorted(N["vs_lr"], key=lambda c: c["d_auroc"])
ab = N["ablation"]; pf = sorted(N["pathfig"], key=lambda c: c["d_auroc"])
panels = [("a  Each model versus logistic regression (full predictor set)",
           [(NAMES[c["a"].split("__")[0]], c) for c in vs], GREY),
          ("b  Matched ablations within Tab-MFM",
           [("Pretraining vs none", ab[0]), ("Column identity on vs off", ab[1]), ("Mask ratio 0.30 vs 0.15", ab[2])], GREY),
          ("c  Removing the clinician image-based assessment, by model family",
           [(NAMES[c["a"].split("__")[0]], c) for c in pf], BLUE)]
heights = [len(p[1]) + 1.6 for p in panels]
fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), gridspec_kw={"height_ratios": heights}, sharex=True)
for ax, (title, rows, col) in zip(axes, panels):
    for i, (lab, c) in enumerate(rows):
        excl = c["lo"] > 0 or c["hi"] < 0
        ax.plot([c["lo"], c["hi"]], [i, i], color=col, lw=1.6, solid_capstyle="butt")
        ax.scatter([c["d_auroc"]], [i], s=30, color=col if excl else "white", edgecolor=col, lw=1.3, zorder=3)
        ax.text(1.01, i, ptxt(c["p_delong"]), va="center", ha="left", fontsize=7.8, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.axvline(0, color="#333333", lw=0.8, ls=(0, (4, 3)))
    ax.set_ylim(-0.8, len(rows) - 0.2); ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK, pad=4)
    ax.tick_params(axis="y", length=0); clean(ax, left=False); ax.grid(axis="x", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
axes[-1].set_xlim(-0.07, 0.125); axes[-1].set_xlabel("Difference in AUROC (95% CI)")
axes[-1].set_xticks([-0.06, -0.03, 0, 0.03, 0.06, 0.09, 0.12])
fig.text(0.99, 0.005, "Filled marker: 95% CI excludes zero. p: DeLong test, unadjusted.", ha="right", fontsize=7.2, color=INK2)
fig.tight_layout(h_pad=1.2); save(fig, "Figure2_effect_sizes")

# ------------------------------------------------ Figure 3: calibration and decision curves (reference models)
REF = [("lr__full", "Logistic regression", BLUE, "-", "o"), ("catboost__full", "CatBoost", ORANGE, "--", "s"),
       ("tabpfn__full", "TabPFN", AQUA, ":", "^")]
REF = [item for item in REF if f"cal_{item[0]}" in A.files and item[0] in N["dca"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.2))
a1.plot([0, 0.8], [0, 0.8], color="#999999", lw=0.8, ls=(0, (4, 3)))
for key, lab, col, ls, mk in REF:
    p = A[f"cal_{key}"]; q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    mp = [p[b == i].mean() for i in range(10)]; ob = [y[b == i].mean() for i in range(10)]
    a1.plot(mp, ob, color=col, ls=ls, lw=1.4, marker=mk, ms=4.2, label=lab, mec="white", mew=0.6)
a1.set_xlim(0, 0.8); a1.set_ylim(0, 0.8); a1.set_xlabel("Mean predicted risk (decile)"); a1.set_ylabel("Observed CIN2+ proportion")
a1.set_title("a  Calibration", loc="left", fontsize=9, fontweight="bold"); a1.legend(frameon=False, fontsize=7.6, loc="upper left"); clean(a1)
d = N["dca"]; pt = np.array(d["pt"])
a2.plot(pt, d["treat_all"], color=GREY, lw=1.0, label="Refer all"); a2.axhline(0, color="#555555", lw=0.8, label="Refer none")
for key, lab, col, ls, mk in REF: a2.plot(pt, d[key], color=col, ls=ls, lw=1.5, label=lab)
a2.set_xlim(0.05, 0.40); a2.set_ylim(-0.02, 0.26); a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
a2.set_title("b  Decision curves", loc="left", fontsize=9, fontweight="bold"); a2.legend(frameon=False, fontsize=7.4, loc="upper right"); clean(a2)
fig.tight_layout(w_pad=2.2); save(fig, "Figure5_calibration_dca")

# ------------------------------------------------ Figure 4: learning curve (logistic regression, exact pipeline)
fig, ax = plt.subplots(figsize=(4.2, 3.1))
for key, lab, col, ls, mk in [("lr__full", "With clinician assessment", BLUE, "-", "o"), ("lr__nopf", "Without", ORANGE, "--", "s")]:
    pts = N["learning_curve"][key]; x = [p["n_train"] for p in pts]; m = np.array([p["mean"] for p in pts]); s = np.array([p["sd"] for p in pts])
    ax.fill_between(x, m - s, m + s, color=col, alpha=0.12, lw=0); ax.plot(x, m, color=col, ls=ls, lw=1.5, marker=mk, ms=4.5, label=lab, mec="white", mew=0.6)
ax.set_xlabel("Training patients per fold"); ax.set_ylabel("Out-of-fold AUROC"); ax.set_ylim(0.62, 0.78)
ax.legend(frameon=False, fontsize=7.6, loc="lower right"); clean(ax); ax.grid(axis="y", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "FigureS1_learning_curve")

# ------------------------------------------------ Figure 5: complementarity
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.3))
W = [0.10, 0.20, 0.30]; FAMS = ["xgb", "lgbm", "catboost", "rf", "ftt", "tabpfn", "tab", "tabmfm_pre030_colid1", "tabmfm_nopre_colid1"]
FAMS = [family for family in FAMS if f"{family}|0.10" in R["B2"]]
a1.plot([0, 30], [0, 30], color="#999999", lw=0.9, ls=(0, (4, 3)), label="Random referral")
for f in FAMS:
    a1.plot([0] + [w * 100 for w in W], [0] + [R["B2"][f"{f}|{w:.2f}"]["rescue_rate"] * 100 for w in W], color=LIGHT, lw=1.0, zorder=1)
lr = [R["B2"][f"lr|{w:.2f}"] for w in W]
a1.errorbar([w * 100 for w in W], [s["rescue_rate"] * 100 for s in lr],
            yerr=[[s["rescue_rate"] * 100 - s["rescue_rate_ci"][0] * 100 for s in lr], [s["rescue_rate_ci"][1] * 100 - s["rescue_rate"] * 100 for s in lr]],
            color=BLUE, lw=1.6, marker="o", ms=4.8, capsize=2.5, label="Logistic regression", zorder=3, mec="white", mew=0.6)
a1.plot([], [], color=LIGHT, lw=1.0, label=f"{len(FAMS)} other model families")
a1.set_xlim(0, 32); a1.set_ylim(0, 62); a1.set_xticks([0, 10, 20, 30])
a1.set_xlabel("Clinician-negative women re-reviewed (%)"); a1.set_ylabel("Clinician-missed CIN2+ found (%)")
a1.set_title("a  Rescue yield", loc="left", fontsize=9, fontweight="bold"); a1.legend(frameon=False, fontsize=7.2, loc="upper left"); clean(a1)
b = R["B3"]; pt = np.array(b["pt"])
a2.plot(pt, b["all"], color=GREY, lw=1.0, label="Refer all"); a2.axhline(0, color="#555555", lw=0.8, label="Refer none")
a2.plot(pt, b["S1"], color=INK2, lw=1.4, ls=(0, (1.5, 1.5)), label="Clinician alone")
a2.plot(pt, b["S4"], color=ORANGE, lw=1.4, ls="--", label="Integrated model")
a2.plot(pt, b["S3"], color=BLUE, lw=1.8, label="Clinician, then model")
a2.set_xlim(0.05, 0.40); a2.set_ylim(-0.02, 0.26); a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
a2.set_title("b  Referral strategies (logistic regression)", loc="left", fontsize=9, fontweight="bold"); a2.legend(frameon=False, fontsize=7.2, loc="upper right"); clean(a2)
fig.tight_layout(w_pad=2.2); save(fig, "Figure4_complementarity")
# ------------------------------------------------ Figure 6: group Shapley attribution across ten families
NA = json.loads((ARGS.results / "numbers_A.json").read_text(encoding="utf-8"))
FAM10 = NA["families"]
GORD = [("G5", "Clinician\nassessment"), ("G3", "Cytology"), ("G2", "HPV testing"), ("G4", "Colposcopic\nfindings"), ("G1", "Demographic and\nreproductive history")]
fig, ax = plt.subplots(figsize=(6.8, 3.9))
off = np.linspace(0.32, -0.32, len(FAM10))
for gi, (g, lab) in enumerate(GORD):
    yc = -gi
    if gi % 2 == 0: ax.axhspan(yc - 0.45, yc + 0.45, color="#f6f6f4", lw=0, zorder=0)
    for fi, f in enumerate(FAM10):
        x = NA["phi"][f][g]; yy = yc + off[fi]; is_lr = f == "lr"
        ax.plot([x["lo"], x["hi"]], [yy, yy], color=BLUE if is_lr else "#b9b9b5", lw=1.5 if is_lr else 0.9, zorder=3 if is_lr else 2, solid_capstyle="round")
        ax.plot(x["est"], yy, "o", ms=4.6 if is_lr else 3.4, color=BLUE if is_lr else GREY, mec="white", mew=0.5, zorder=4 if is_lr else 2)
    lo, hi = NA["share_range"][g]
    ax.text(0.172, yc, f"{100*lo:.0f}% to {100*hi:.0f}%".replace("-", "\u2212"), va="center", ha="left", fontsize=7.8, color=INK2)
ax.axvline(0, color="#777777", lw=0.8, zorder=1)
ax.set_yticks([-i for i in range(len(GORD))]); ax.set_yticklabels([l for _, l in GORD], fontsize=8.2)
ax.set_ylim(-len(GORD) + 0.5, 0.5); ax.set_xlim(-0.035, 0.168)
ax.text(0.172, 0.62, "Share of\nAUROC − 0.5", va="bottom", ha="left", fontsize=7.6, color=INK2, fontweight="bold")
ax.set_xlabel("Shapley value (AUROC units)")
ax.plot([], [], "o-", color=BLUE, ms=4.6, lw=1.5, mec="white", label="Logistic regression (95% CI)")
ax.plot([], [], "o-", color=GREY, ms=3.4, lw=0.9, mec="white", label=f"{max(0, len(FAM10)-1)} other model families")
ax.legend(frameon=False, fontsize=7.4, loc="lower right", bbox_to_anchor=(1.0, 0.02))
clean(ax); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
fig.tight_layout(); fig.subplots_adjust(right=0.84); save(fig, "Figure3_group_attribution")
print("fig6 ok")

if ARGS.fast:
    (ARGS.output / "FIGURE_MODE.txt").write_text("FAST synthetic smoke test; not paper estimates\n", encoding="utf-8")
print("ok")
