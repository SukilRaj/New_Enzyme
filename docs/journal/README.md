# Journal handover package

Prepared 26 October 2026. This folder records the installed research platform and its measured computational results. It is a manuscript basis, not a submission-ready claim of high accuracy or experimentally improved enzymes.

- `ENZAIME_HELIX_PROJECT_SUMMARY.pdf`: formatted project summary, draft abstract, methods, comparisons, results, limitations and run instructions.
- `PROJECT_AND_JOURNAL_SUMMARY.md`: editable source for that PDF.
- `BENCHMARK_COMPARISON_AND_GAPS.md`: which environmental and benchmark comparisons are implemented, their measured results, and the missing work for a direct base-paper comparison.
- `figures/`: architecture, grouped evaluation and prediction diagnostics as PNG and editable SVG.
- `metric_tables.csv`: 19 metric rows with explicit study/cohort labels.
- `FIGURES_AND_METRICS.md`: figure and metric provenance, including the definition of the shifted twofold criterion.
- `evidence_manifest.json`: SHA-256 fingerprints for the implementation, reports, predictions and selected linked PET artifacts.
- `helix_environment_snapshot.txt`: installed Python packages for the Helix environment only.
- `ENZAIME_HELIX_JOURNAL_PACKAGE.zip`: these documents plus selected saved reports, splits, predictions, verification records, screenshots and run guides. It does not include the complete application, datasets or model checkpoints.

The runnable project remains in `D:\Enzyme_ML_new` and still needs the linked PET backend/data/models in `D:\Final_Year_Project\EnzAIme_ML`. Open `EnzAIme-Helix.code-workspace` in VS Code and follow `docs/RUN_IN_VS_CODE.md`. Do not move just this journal ZIP and expect the application to run.

The saved results are exploratory. PET R² remains negative; GFP baseline correlation is not classification accuracy; mutant folding, wet-lab validation and a matched reproduction of the paper's independent evaluator are unfinished. Author details, source-data references, a target journal's formatting and any additional experiments must be supplied before submission.

## Recreate the documentation

From `D:\Enzyme_ML_new`, install optional document dependencies using `python -m pip install -r requirements-report.txt` in a compatible project environment. Use the project's Python interpreter when running:

```powershell
.\.venv\Scripts\python.exe scripts/build_journal_evidence.py
.\.venv\Scripts\python.exe scripts/build_journal_pdf.py
.\.venv\Scripts\python.exe scripts/package_journal.py
```

The PDF generator uses the Windows Arial, Times and Consolas fonts. The current PDF was generated with the bundled document runtime and visually checked page by page. `build_journal_evidence.py` reads the current assay `latest.json`; preserve the original cited run when producing revisions. Packaging does not rerun training.
