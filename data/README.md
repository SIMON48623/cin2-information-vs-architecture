# Data

**SYNTHETIC — not patient data.** The two committed CSV files are generated
only from the published marginal summaries in Table 2. Variables are sampled
independently. The outcome coefficients are human-selected for plausible
directions and were not estimated from, fitted to, or sampled from patient
records. Synthetic results are demonstrations and will not reproduce the
article's numerical results.

Patient-level development and external-validation data cannot be distributed.
Authorized investigators can pass a local CSV or XLSX file with the columns in
`schema.csv`. Columns that look like identifiers or names are dropped
immediately after loading and are never included in any output.

The public synthetic files contain 879 development rows and 103 external rows.
The documented missingness is introduced only in the development file: age 3,
gravidity 3, parity 2, and iodine result 1.

