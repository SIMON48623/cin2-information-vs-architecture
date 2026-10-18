"""Figure 4 (clinician-model complementarity), drawn from the summary results in
results/reference/numbers_B.json. Writes vector PDF/EPS and a 300-dpi TIFF.

usage: python figure4.py path/to/numbers_B.json output_dir
"""
import json
import sys
from pathlib import Path

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads(SRC.read_text(encoding="utf-8"))

# Times-style serif as in the other figures; TeX Gyre Termes if present, else Times New Roman
for f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf"):
    p = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre") / f
    if p.exists():
        fm.fontManager.addfont(str(p))

BLUE, ORANGE = "#2B5C8A", "#C8704A"
LIGHT, MID, DARK, TICK = "#D6D6D6", "#929292", "#4B4B4B", "#3A3A3A"
SPINE = "#A6A6A6"
plt.rcParams.update({
    "font.family": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "serif"], "font.size": 9.6,
    "axes.edgecolor": SPINE, "axes.linewidth": 0.9, "axes.labelcolor": "black",
    "xtick.color": SPINE, "ytick.color": SPINE, "xtick.labelcolor": TICK, "ytick.labelcolor": TICK,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "legend.fontsize": 7.7, "axes.labelpad": 2.3, "pdf.fonttype": 3, "ps.fonttype": 3,
})

# geometry measured from the submitted figure (2126 x 1011 px at 300 dpi = 180 mm wide)
W, H = 2126, 1011
fig = plt.figure(figsize=(W / 300, H / 300))
a1 = fig.add_axes([153 / W, 1 - 861 / H, (988 - 153) / W, (861 - 84) / H])
a2 = fig.add_axes([1227 / W, 1 - 861 / H, (2061 - 1227) / W, (861 - 84) / H])
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---- panel a: share of clinician-negative CIN2+ found when re-reviewing the top 10/20/30%
W3 = [0.10, 0.20, 0.30]
others = ["xgb", "lgbm", "catboost", "rf", "ftt", "tabpfn", "tab", "tabmfm_pre030_colid1", "tabmfm_nopre_colid1"]
a1.plot([0, 30], [0, 30], color=MID, lw=1.1, ls=(0, (3.5, 3)), label="Random referral", zorder=1)
for fam in others:
    a1.plot([0] + [w * 100 for w in W3], [0] + [R["B2"][f"{fam}|{w:.2f}"]["rescue_rate"] * 100 for w in W3],
            color=LIGHT, lw=1.4, zorder=1)
a1.plot([], [], color=LIGHT, lw=1.4, label="Nine other model families")
lr = [R["B2"][f"lr|{w:.2f}"] for w in W3]
y = [s["rescue_rate"] * 100 for s in lr]
err = [[(s["rescue_rate"] - s["rescue_rate_ci"][0]) * 100 for s in lr],
       [(s["rescue_rate_ci"][1] - s["rescue_rate"]) * 100 for s in lr]]
a1.errorbar([w * 100 for w in W3], y, yerr=err, color=BLUE, lw=2.2, elinewidth=2.2, capsize=4, capthick=1.6,
            marker="o", ms=6.0, mfc=BLUE, mec="white", mew=1.0, label="Logistic regression", zorder=3)
a1.set_xlim(0, 32); a1.set_ylim(0, 62)
a1.set_xticks([0, 10, 20, 30]); a1.set_yticks(range(0, 61, 10))
a1.set_xlabel("Clinician-negative women re-reviewed (%)")
a1.set_ylabel("Clinician-missed CIN2+ found (%)")
a1.legend(frameon=False, loc="upper left", borderaxespad=0.45, handlelength=2.1)

# ---- panel b: net benefit of the referral strategies (logistic regression)
b = R["B3"]; pt = b["pt"]
a2.plot(pt, b["all"], color=MID, lw=1.3, label="Refer all")
a2.axhline(0, color=DARK, lw=1.1, label="Refer none")
a2.plot(pt, b["S1"], color=DARK, lw=1.9, ls=(0, (1.2, 1.4)), label="Clinician alone")
a2.plot(pt, b["S4"], color=ORANGE, lw=1.9, ls=(0, (4, 2.5)), label="Integrated model")
a2.plot(pt, b["S3"], color=BLUE, lw=2.4, label="Clinician, then model")
a2.set_xlim(0.05, 0.40); a2.set_ylim(-0.02, 0.26)
a2.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
a2.set_xticklabels([f"{v:.2f}" for v in a2.get_xticks()])
a2.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
a2.set_yticklabels([f"{v:.2f}" for v in a2.get_yticks()])
a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
a2.legend(frameon=False, loc="upper right", borderaxespad=0.35, handlelength=2.6)

# ---- panel titles
ty = 1 - 59 / H
fig.text(154 / W, ty, "a  Yield among clinician-negative women", fontsize=9.5, fontweight="bold", va="baseline")
fig.text(1229 / W, ty, "b  Referral strategies (logistic regression)", fontsize=9.5, fontweight="bold", va="baseline")

fig.savefig(OUT / "Figure4.pdf")
fig.savefig(OUT / "Figure4.eps")
import io
from PIL import Image
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, facecolor="white"); buf.seek(0)
Image.open(buf).convert("RGB").save(OUT / "Figure4.tif", compression="tiff_lzw", dpi=(300, 300))
fig.savefig(OUT / "Figure4_preview.png", dpi=150)
print("written to", OUT)
