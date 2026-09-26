# Article-to-code map

Numbering follows the revised manuscript (three tables, five figures, no Supplementary Material).

| Article item | Generator | Primary output |
|---|---|---|
| Figure 1 (study design and analysis workflow) | Schematic; not generated from data | — |
| Table 1 (variables, coding, a priori groups, univariable AUROC) | `data/schema.csv`; `analysis/10_main_analysis.py` | `numbers.json` (`univariable`) |
| Section 3.1 (baseline characteristics; full table) | `analysis/05_baseline_table.py` | `results/reference/baseline_characteristics.csv`, `.md` |
| Table 2 (discrimination, probability quality, paired comparison with logistic regression) | `analysis/10_main_analysis.py` | `perf_all.csv`, `contrasts_vs_lr.csv`, `numbers.json` |
| Figure 2 (effect sizes; ablations; removal of the clinician assessment) | `analysis/10_main_analysis.py`, `analysis/20_figures.py` | `contrasts_vs_lr.csv`, `contrasts_ablation.csv`, `contrasts_pathfig.csv` |
| Figure 3 (group Shapley attribution) | `analysis/12_group_shapley.py`, `analysis/20_figures.py` | `numbers_A.json` (`auc_subsets` holds the AUROC of all 31 subsets in all ten families) |
| Table 3 (re-review of clinician-negative women) | `analysis/11_complementarity.py` | `numbers_B.json` |
| Figure 4 (complementarity) | `analysis/11_complementarity.py`, `analysis/20_figures.py` | `numbers_B.json` |
| Figure 5a–b (calibration, decision curves) | `analysis/10_main_analysis.py`, `analysis/20_figures.py` | `numbers.json` |
| Figure 5c (learning curves) | `analysis/10_main_analysis.py`, `analysis/20_figures.py` | `numbers.json` (`learning_curve`); plotted as `FigureS1_learning_curve` |
| Section 2.4 (trainable parameters, selected epochs) | `scripts/01_fit_models.py` | `fit_models_metadata.json` |
| Section 3.7 (calibration within clinician strata) | `analysis/14_strata_calibration.py` | `strata_calibration.json` |
| Section 3.8 (age sensitivity analysis) | `analysis/13_sensitivity_age.py` | `sens_age.json` |

`analysis/20_figures.py` plots the data behind Figures 2–5; the layout and colours of the published figures differ.
