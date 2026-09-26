# Analysis B — clinician–model complementarity (pre-specified)

Written 2026-09-25, after the ten-family contrasts were computed and BEFORE any stratified
(clinician-positive / clinician-negative) quantity was computed. Nothing below is revised after results.

CIBA = clinician image-based assessment (`pathology_fig`), 1 = positive judgement.
Models used here never saw CIBA: the `__nopf` out-of-fold predictions of the rerun.
Primary learner: logistic regression (its pipeline is reproduced to 1e-12). Robustness: all ten families.

## B1. Discrimination within clinician strata
Raw pooled OOF AUROC of each `__nopf` model among CIBA-negative and among CIBA-positive women,
DeLong 95% CI. Between-stratum difference: z = (A_neg − A_pos) / sqrt(V_neg + V_pos) (independent groups).

## B2. Rescue yield among CIBA-negative women (workload-anchored, label-free thresholds)
Within each held-out fold, rank CIBA-negative women by model risk and refer the top w,
w ∈ {10%, 20%, 30%}. No outcome labels are used to set the rule. Pooled over folds report:
CIN2+ found among CIBA-negative CIN2+, additional referrals, referrals per CIN2+ found, and the
expected yield of referring the same number at random (w × CIBA-negative CIN2+).
Pathway sensitivity/specificity: clinician alone vs clinician + model at each w.
95% CIs by patient-level bootstrap (2,000, seed 13), folds fixed.

## B3. Net benefit of four referral strategies, pt 0.05–0.40
S1 clinician alone (refer CIBA-positive).
S2 LR without CIBA alone: nested-calibrated `lr__nopf` ≥ pt.
S3 sequential: refer CIBA-positive; among CIBA-negative refer if stratum-recalibrated risk ≥ pt.
   Stratum recalibration is nested: for held-out fold k the calibrator family is chosen by inner CV and
   fitted on CIBA-negative women of the other folds only; fold-k labels are never used.
S4 integrated LR with CIBA: nested-calibrated `lr__full` ≥ pt.
Primary contrasts: ΔNB(S3 − S1) and ΔNB(S3 − S4) at pt = 0.10, 0.20, 0.30 with bootstrap 95% CIs.
Robustness: ΔNB(S3 − S1) for all ten families at the same pt.

## B4. External cohort (descriptive only; 34 events)
If CIBA is available and row-aligned, apply B2 to external CIBA-negative women using the frozen
`lr__nopf` predictions. No inference.

## Interpretation caveat, fixed in advance
If the reporting pathologist could see referral information including the clinician's judgement,
the clinician's apparent accuracy (S1) may be inflated; S3 − S1 would then understate the model's
contribution. This will be stated whichever way the result falls.
