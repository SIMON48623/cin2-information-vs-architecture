# Pre-specified downstream analysis plan A — English translation

This is an English translation of `A_analysis_plan_prespecified.md`; the
Chinese source file remains byte-for-byte unchanged and controls if any wording
differs.

- Define the coalition value as uncalibrated pooled out-of-fold AUROC minus
  0.5, with the empty coalition set to zero.
- For each model family, calculate exact five-group Shapley values from all 31
  non-empty subsets and report their shares.
- Keep out-of-fold predictions fixed. Use 2,000 patient-level bootstrap samples
  with seed 13 and report the 2.5th and 97.5th percentiles.
- Across the ten model families, calculate rankings, Kendall's W, and a
  10,000-permutation test with seed 13. Report agreement on the top group and
  the range of shares.
- Restrict the main conclusion to consistency of the top group and confidence
  in Kendall's W. Treat all other findings as descriptive.

