# Figure and metric provenance

Figures are reproduced by `.venv\Scripts\python.exe scripts\build_journal_evidence.py`. They use the saved condition report and the assay run named by `models/assay_v2/latest.json`.

Assay run used: `train-20261024-160717`.

`grouped_results` shows pooled outer-fold MAE for both grouping designs. `prediction_diagnostics` uses all 208 sequence-grouped observations and their saved predictions. `architecture` is an implementation diagram. Each is supplied as PNG and editable SVG.

`metric_tables.csv` preserves cohort labels. Never rank methods across different cohorts as if their errors were directly comparable. `within_twofold_fraction` uses the ratio of 1 + TPA; it is not raw-concentration accuracy.
