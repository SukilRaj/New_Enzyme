# Helix — local protein evolution laboratory

## Integrated EnzAIme × Helix platform

Start with the [VS Code run guide](docs/RUN_IN_VS_CODE.md), [complete project and journal summary](docs/journal/PROJECT_AND_JOURNAL_SUMMARY.md), or [FASTA and structure guide](docs/SEQUENCE_AND_STRUCTURE_GUIDE.md). The journal folder includes a formatted PDF, editable source, figures, metric tables and evidence hashes. The summary distinguishes completed implementation from unvalidated research claims.

Run `Start-Research.cmd` to start both PET discovery and GFP evolution through one shared workspace at **http://127.0.0.1:8765/**. Its persistent navigation covers PET discovery, condition analysis, protein images, sequence/structure, GFP evolution, model training, evaluation, experiment history and evidence. Existing research modules load inside the shared shell; changing sections preserves their state. Transitions honor the operating system's reduced-motion setting. In VS Code you can also use **Terminal → Run Task → Start integrated EnzAIme + Helix**. Restart the server after updating Python files; stop the previous server first if it occupies port 8765.

The hub provides shared experiment history, automatically persisted PET recommendations, dataset/checkpoint fingerprints and an evidence export. The PET module uses the existing project at `D:\Final_Year_Project\EnzAIme_ML` and its `.venv-1`; this is a linked integration and needs both folders. Read the [thorough project comparison and integration guide](docs/PROJECT_COMPARISON_AND_INTEGRATION.md) for architecture, implemented features, limitations and a final-year research plan. PET scores and GFP scores retain separate targets and units.

### Training laboratory and model limitations

The maintained PET frontend now lives in this project's `frontend/` folder, including its built assets. FASTA variants can be imported in **Sequence & structure** and their edits inspected on cached reference coordinates, with a separate **Download reference PDB** action. A FASTA file is a sequence, not a set of 3D coordinates; no mutant fold is predicted. The original PET backend/data/model folder remains required.

Open **Training laboratory** from the shared navigation to inspect the completed assay-aware PET training study, grouped-fold predictions, feature coverage, prediction plot and downloadable protocol/report artifacts. The completed `assay_v2` study compared 24 prespecified candidates and evaluated sequence-grouped and sequence-plus-study-grouped generalization before refitting its selected recipe on all 208 eligible measurements. The all-data refit is for research inspection; it has no independent holdout and is not promoted as the prediction service. The existing PET checkpoint remains the default.

Nested sequence-grouped selection for sequence + conditions had MAE **1,404.6 µM**, R² **−0.051**, and **22.1%** within a factor of two on the **1 + TPA** scale. Each fold selected its own model; ExtraTrees was chosen for the final full-data refit. The sequence-plus-study holdout had MAE **1,495.5 µM**, R² **−0.073**. These results are weak and do not support a high-accuracy claim. Twenty-five sequence groups and 22 studies are represented, with substantial assay/target heterogeneity and a heavy-tailed response. The training report explains the split design and limitations in more detail.

### Conditions, protein images and measured comparisons

Open **http://127.0.0.1:8765/analysis** after starting `Start-Research.cmd`. The studio adds temperature/pH/time response surfaces, condition-window ranking of 25 proteins, measured-assay evidence, and five downloadable structural views for each of 13 sequence-matched proteins. Every analysis is saved in the shared registry.

The comparison tab exposes reproducible fixed-test and nested protein-grouped evaluations. Environment-only MAE is 17.7% lower than the original PET checkpoint on its historical test, and sequence+environment MAE is 16.6% lower than sequence-only in grouped evaluation. These are different comparisons; R² remains negative. The experimental structure-fusion candidate performs worse than the original on the fixed test, so the original checkpoint remains the default. These results do not demonstrate superiority to the base paper.

Read [analysis methods, results and demonstration steps](docs/ANALYSIS_METHODS_AND_RESULTS.md) for the exact protocols, image provenance, limitations and reproduction commands. Images render existing AlphaFold coordinates; they do not predict new protein folds. Environmental effects are modeled for PET, whose data includes assay conditions, and are not inferred for GFP without corresponding labels.

Local Windows application built around the authors' downloaded Directed Evolution repository. Start with the **avGFP fluorescent-protein benchmark**.

## Launch

Double-click `Start-Helix.cmd`, then open **http://127.0.0.1:8765**. Keep its terminal open while an experiment is running. Close that terminal to stop the server. Restart Helix after updating Python files so the server loads the changed experiment engine. Jobs interrupted by shutdown are marked interrupted on the next start.

The project uses `.venv` (Python 3.12). To install again on another machine, install Python 3.12 and run `Setup-Helix.ps1` for the CPU application, then `Setup-Helix.ps1 -IncludeESM` for the larger deep-learning dependencies. Dependencies are in `requirements-core.txt` and `requirements-local.txt`. Model files download from Hugging Face on first ESM use and are stored in `models/huggingface`. No account or paid API is required for these public models.

## What is implemented

- A local dashboard with dataset distribution, experiment configuration, progress, live fitness curves, candidate table, exports and persistent history.
- A CPU baseline: position-specific one-hot encoding, ridge regression, random substitution, elitist selection. This is **not** the paper's method.
- An ESM adaptation: frozen ESM-2 encoder, the repository's Attention1D decoder, ESM masked-language-model proposals, random/importance masking, and elitist selection. Only canonical amino acids are proposed.
- An optional research extension: three bootstrap-trained Attention1D heads estimate prediction spread; candidate selection combines mean + 0.5 standard deviations with a 0.5-scaled normalized minimum Hamming-distance bonus. The candidate export records uncertainty and the resulting selection score.
- Deduplication before a seeded 80/20 split; held-out RMSE, MAE and Spearman correlation. Held-out labels never enter training or search.
- Persistent configurations, dataset hash, explicit sequence splits, trained predictor, held-out predictions, candidate CSV and result JSON under `runs/<id>`.
- Cancellation between operations, input validation, localhost-only binding, and request-token checks on state-changing endpoints.

## Scientific limits / remaining replication work

This is a local research foundation, **not yet a full reproduction of the paper**. The original ZIP contains no trained checkpoints and no `landscape_params/esm1b_landscape` files required by the original independent evaluator. None of the local scores should be described as validated fluorescence or paper benchmark results.

The ESM adapter intentionally uses smaller settings suitable for local testing: 128 decoder hidden units, learning rate 0.001, batch size 4, user-selected sample limit and epochs, 50/50 random/importance proposals, two proposals per population member, and a median-fitness training sequence as the starting point. Importance is computed for each selected sequence rather than the full proposal population. It supports 8M (smoke test) and 35M ESM-2 encoders. These changes must be reported in project results. The paper's settings, starting sequences, evaluator, full benchmark suite, and repeated baseline comparisons still need to be matched before claiming reproduction.

The UI initially exposes only avGFP. All eight original datasets remain in the extracted source directory. This does not claim that all original benchmarks have been executed.

A random split can yield optimistic performance for closely related variants. Mutation-distance splits and repeated seeds are needed for stronger generalization evidence. Optimizing a model can exploit its prediction errors; generated candidates require independent evaluation.

## Upstream fixes

- Removed the duplicate `--predictor_ckpt_path` argument that prevented inference argument parsing.
- Fixed Lightning CPU device configuration in `scripts/train_decoder.py`.
- Defaulted training workers to zero for Windows.
- Made external WandB logging opt-in; local CSV logging remains enabled.

Original version pins are preserved inside `Directed_Evolution-main/requirements.txt`; the separate local environment uses Python 3.12-compatible versions. These environment changes also prevent a claim of bit-for-bit reproduction. Original archive remains unmodified.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
node --check lab/static/app.js
```

Tests exercise deterministic baseline results, disjoint sequence splits, elitism, cancellation, validation, and local API access controls. They do not validate model biology or reproduce paper tables.

## Research comparator and scientific limits

The uncertainty-and-diversity option is an implemented comparator, not an established novelty claim. Prior work already uses uncertainty-guided active learning in directed evolution ([ALDE, Yang et al., 2025](https://doi.org/10.1038/s41467-025-55987-8)) and co-optimizes predicted fitness with sequence diversity ([MODIFY, Ding et al., 2024](https://doi.org/10.1038/s41467-024-50698-y)). The implementation here is a simple combination of established ideas. Its ensemble spread is an uncalibrated uncertainty proxy, and its fixed acquisition weights (0.5 for ensemble standard deviation and 0.5 for normalized minimum Hamming distance) are starting values. A possible project contribution is to study uncertainty calibration on mutation-distance splits and its interaction with ESM masked proposals, with equal budgets, multiple seeds, and controlled ablations; this needs a fuller literature review before it can be presented as novel. Candidate scores are model outputs and require independent evaluation.

## Attribution

Tran, Thanh V. T. and Hy, Truong Son. *Protein Design by Directed Evolution Guided by Large Language Models*. IEEE Transactions on Evolutionary Computation 29(2), 418–428 (2025). DOI: 10.1109/TEVC.2024.3439690.

Original repository: https://github.com/HySonLab/Directed_Evolution. Original code and this derivative integration are provided under GPL-3.0; see `Directed_Evolution-main/LICENSE`. Keep attribution when distributing the project. The local paper is not bundled in the application.
