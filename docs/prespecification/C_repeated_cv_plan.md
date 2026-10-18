# Analysis C — repeated cross-validation (added in revision)

Written 2026-10-18, in response to the handling editor, BEFORE any repeated-CV result on the development cohort was computed. Nothing below is revised after results.

Purpose. The primary comparisons use one stratified five-fold assignment (random_state 42) and one model seed (13), so their DeLong tests and bootstrap intervals are conditional on a single set of fitted models. This analysis estimates how the comparison of each configuration with logistic regression varies across fold assignments, refitting and random initialisation.

Design. Ten repetitions of stratified five-fold cross-validation of the development cohort (n = 879). In repetition r (r = 1, ..., 10) the fold random_state is 1000 + r; the model seed, the seed of the inner early-stopping split and the seeds of all stochastic learners are 2000 + r. The twelve full-predictor-set configurations of the primary analysis are refitted with the hyperparameters, preprocessing and early-stopping rule of the primary protocol; nothing is tuned. Classical learners run on CPU; neural and foundation models may run on GPU, and the device is recorded.

Outcomes. For each configuration versus logistic regression: the mean, standard deviation, minimum and maximum of the pooled out-of-fold AUROC difference across the ten repetitions, and the number of repetitions in which the difference was positive; and a corrected repeated k-fold t test (Bouckaert and Frank, 2004) on the 50 per-fold AUROC differences, t = mean(d) / sqrt((1/50 + n_test/n_train) * var(d)), 49 degrees of freedom, two-sided, Holm-adjusted across the eleven comparisons.

Reporting. Results are reported whatever their direction. The primary single-protocol analysis is not replaced. The clinician-assessment removal, variable-group attribution and complementarity analyses are not repeated.
