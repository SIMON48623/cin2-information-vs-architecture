# Fixed analysis protocol

The complete machine-readable protocol is in `configs/protocol.yaml`. Random
seed 13 is used for model fitting, calibration selection, bootstrap sampling,
and permutation tests. The single patient-level split is
`StratifiedKFold(5, shuffle=True, random_state=42)` and is reused by every
model and feature set.

Torch CPU fitting uses `torch.set_num_threads(8)` before model construction,
matching the original rerun script. Deterministic algorithms are enabled. The
eight-thread setting is part of the recorded protocol because changing the
CPU thread count altered later FT-Transformer early-stopping checkpoints even
when the seeds, folds, input tensors, architecture, and optimizer were fixed.

Numeric variables receive training-fold median imputation and standardization.
Categorical variables receive training-fold most-frequent imputation and
one-hot encoding for classical models. Neural models use fold-local category
vocabularies; unseen values map to the reserved category. TabPFN uses median
imputation and integer categorical encoding. No preprocessing is learned from
an outer validation fold.

Neural early stopping uses a stratified 15% split of each outer training fold,
seed 13, and never inspects the outer validation fold. External validation
refits logistic regression and the pretrained Tab-MFM on all development
records; Tab-MFM uses the rounded median of the five selected epochs. The
supervised neural loss is class-weighted binary cross entropy.

`ftt` is the official `rtdl_revisiting_models.FTTransformer` with its default
three-block, width-192, eight-head architecture and default optimizer. `tab`
is the article's feature-token/[CLS] transformer with width 64, two blocks,
four heads, and feed-forward width 128. Tab-MFM jointly represents numeric and
categorical tokens, uses per-column categorical lookup embeddings, reconstructs
masked numeric values and categorical levels, and classifies a mean-pooled
representation through LayerNorm, dropout, and a linear head. TabPFN receives
median-imputed numeric columns plus integer-encoded categorical columns and the
corresponding categorical feature indices; it never receives one-hot data.

The twelve configured model variants comprise five classical families,
FT-Transformer, TabPFN, a tabular transformer, the two primary Tab-MFM
variants, and two full-feature Tab-MFM ablations. The ten primary families are
evaluated for both `full` and `nopf`; the two ablations are full-only.

Inference uses the Sun–Xu fast DeLong formulation, a common set of 2,000
patient bootstrap resamples (seed 13), Holm adjustment, fold-nested selection
of sigmoid versus isotonic calibration, fold-nested thresholds, Wilson
intervals, decision-curve analysis, exact five-group Shapley decomposition,
Kendall's W, and 10,000 permutations. Fast mode uses the same estimator classes
and preprocessing, while limiting Tab-MFM pretraining to one epoch, all neural
supervised training to two epochs, bootstrap resampling to 50 replicates, and
the permutation test to 100 replicates. With no local TabPFN checkpoint, that
family is explicitly skipped and no substitute column is produced. Fast
outputs are never paper results.

The article's main pipeline was run on CPU. In analysis A, classical families
were run on CPU and the five neural/foundation-model families on CUDA. A family
and all its 31 subsets must stay on one device. GPU and CPU floating-point
results need not be bit-identical.
