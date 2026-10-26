# EnzAIme × Helix: project comparison and integrated final-year platform

Audit date: 24 October 2026. This report describes code and saved artifacts inspected locally, not a claim that every historical experiment was independently rerun.

**Implementation update:** the integrated application now also includes a condition-analysis and protein-image studio at `/analysis`. Response surfaces, condition-window ranking, five downloadable structure views, fixed-test ablations and nested grouped evaluation are implemented. The original neural image branch remains untrained; a separate structure-descriptor surrogate was trained and evaluated, with worse fixed-test performance than the original. See [Analysis methods and results](ANALYSIS_METHODS_AND_RESULTS.md) for the current experiment and evidence. Work packages below remain a roadmap where their acceptance criteria have not been met.

## 1. What each project actually does

**EnzAIme_ML** (`D:\Final_Year_Project\EnzAIme_ML`) is a PET-enzyme research application. It asks: *which catalog proteins does a trained model rank highest at a specified temperature, pH and reaction duration?* The target is TPA concentration in micromolar units. The currently served application is the rebuilt FastAPI/React workspace, not the older Ridge pipeline still present in its source tree.

**Enzyme_ML_new** (`D:\Enzyme_ML_new`) is a local adaptation of the downloaded directed-evolution paper repository. It asks: *how do protein-fitness predictors and mutation-selection strategies behave on the avGFP benchmark?* It trains a predictor for each experiment and searches sequences using predicted fitness. The original downloaded code and datasets are preserved under `Directed_Evolution-main`.

They share protein representations and research tooling, but their targets, datasets and scientific claims are different. GFP is a fluorescent protein benchmark, not evidence that the PET model can optimize plastic degradation. Joining their software does not establish transfer learning or biological improvement.

## 2. Feature-by-feature comparison

| Area | EnzAIme_ML | Enzyme_ML_new |
|---|---|---|
| Main task | Rank existing PET-protein catalog under conditions | Train and search avGFP sequence variants |
| Backend/UI | FastAPI, React/Vite | Flask, HTML/CSS/JavaScript |
| Current neural architecture | Frozen ESM-2 35M residue features, Attention1D, condition inputs; optional image branch | Frozen ESM-2 8M/35M, original Attention1D decoder adapted for CPU |
| Simple baseline | Historical Ridge and controlled-analysis scripts remain, but active v2 does not fall back to Ridge | Position-specific one-hot + Ridge, random proposals, elitist selection |
| Activity evidence | 221 confirmed assay rows / 25 protein IDs; checkpoint pipeline uses 208 rows after exclusions/deduplication | 51,715 valid deduplicated avGFP sequences |
| Output units | Predicted TPA concentration, µM; internal target log(1+TPA) | Original dataset fitness scale |
| Conditions | Temperature, pH, duration are model inputs | No PET assay conditions |
| Mutation tools | Validated substitution preview, reference checking, evidence browsing | Random or ESM masked-language-model proposals and iterative selection |
| Structures | PDB parsing, chain mapping, sequence coverage, molecular viewer, 13 verified cached matches | Illustrative molecule artwork; no structural prediction or validation |
| Explanations | Gradient × input residue sensitivity, condition derivatives, attention, optional constrained Ollama evidence selection | Logs, metrics, predictions and ensemble spread |
| Stability | 5,622 measured Tm records / 660 sequences; optional model/Pareto code | No Tm predictor |
| Image modality | 13 sequence-verified AlphaFold-derived distance maps; current checkpoint image_trained=false | No image branch |
| Uncertainty | No established calibrated uncertainty for current PET predictions | Three bootstrap decoder heads for optional uncertainty/diversity selection; uncalibrated spread |
| Persistence | Model manifests, held-out predictions, features and research reports; UI result downloads | Per-run state, configuration, split, metrics, candidate CSV, model and provenance |
| Main limitation | Current held-out R² approximately −2.30; mutation-effect validity unestablished | Missing independent paper oracle; adapted settings and limited benchmarks |

## 3. EnzAIme_ML implementation details

The active entry point `backend/app/main.py` imports `workspace_app.py`. Its routers cover neural status/recommendations, attribution, local explanations, sequence mutation preview, PDB inspection, measured stability evidence, cached structures, evaluation and data inventory. The earlier Ridge-related routes and services are historical code, not evidence that the active workspace serves those models.

`NeuralRecommender` checks the dataset digest, encoder provenance, feature/sequence hashes and checkpoint schema before scoring. It loads tensor weights with `weights_only=True`, standardizes conditions using checkpoint statistics, ranks the catalog by predicted TPA and exposes out-of-range conditions. The image branch is disabled by checkpoint metadata. A compatible stability checkpoint is required before joint Pareto ranking becomes available.

The saved held-out predictions allow R² to be recomputed. Negative R² means the squared error exceeds that of predicting the held-out mean. A functional neural inference endpoint is not evidence of useful prediction quality. The checkpoint's stored MAE/RMSE and recalculated R² should accompany demonstrations.

Gradient × input explains local model sensitivity. It does not estimate causal mutation effects, prove a catalytic mechanism, or establish uncertainty. Llama explanations select trusted evidence IDs and render controlled text; the LLM is not the source of scientific scores.

The structure inventory contains 13 full sequence matches and 12 excluded mappings. These are predicted AlphaFold structures. A distance map is a deterministic representation of a structure, not a second experimental label. Sparse image coverage could confound a multimodal experiment.

Stability data collection and training/import pipelines are implemented. A prepared notebook, collected dataset or importer is not a trained and validated stability model. Historical ablation/nested-analysis reports remain separate from the current checkpoint unless artifact hashes establish a link.

## 4. Enzyme_ML_new implementation details

The CPU engine validates inputs, deduplicates sequences before splitting, uses a seeded 80/20 split, trains a Ridge predictor and proposes amino-acid substitutions. Held-out labels are used for evaluation rather than candidate selection. The starting sequence is drawn from training data.

The ESM adaptation freezes the encoder and trains an Attention1D decoder. It uses local cached models when available, masked-language-model proposals and bounded search settings. The uncertainty/diversity option trains three bootstrap decoder heads, scores mean + 0.5 standard deviation, and applies a 0.5-scaled normalized minimum Hamming-distance bonus during population selection. This method changes training cost as well as acquisition; fair comparisons must report both.

Runs store dataset and engine hashes, seed/configuration, train/test sequences, model revision where available, held-out predictions, metrics, logs and candidate exports. RMSE, MAE and Spearman evaluate the predictor. Predicted candidate fitness evaluates the same surrogate being optimized, so it is not an independent measure of biological improvement.

Only the avGFP workflow is currently exposed in the local UI. The paper's evaluator/checkpoints are missing from the downloaded archive, and settings differ from the paper. This remains a partial local adaptation, not full reproduction of all paper tables.

## 5. What this integration adds

The integration is implemented in `Enzyme_ML_new`; the PET source and model artifacts remain in their existing directory. This is a **linked local integration**, not a self-contained portable copy. Preserve both directories, or set `ENZAIME_ROOT` and `ENZAIME_PYTHON` when relocating the PET project.

1. **Shared research hub:** a responsive overview with live service availability, artifact-derived evidence counts, checkpoint readiness, separate workspace entry points and research milestones.
2. **One browser origin:** the hub, PET frontend and GFP frontend are available through port 8765. The PET backend uses an internal port 8766, with a per-launch secret on gateway requests.
3. **Separate compatible Python environments:** existing PET and GFP stacks run in separate processes. This avoids silently replacing working dependency versions.
4. **Integrated PET workflow:** existing recommendations, residue attribution, mutation preview, PDB inspection, structure catalog and optional local explanations are routed through an allowlisted gateway. Existing UI capabilities are reused rather than represented by placeholder buttons.
5. **Automatic PET experiment records:** each successful ranking through the integrated gateway saves its conditions, result, units, evaluation metadata, dataset digest and checkpoint digest to `research_runs/`. Earlier standalone PET UI requests are not retroactively recovered.
6. **Cross-project experiment registry:** PET and GFP runs appear together, filterable by domain, with individual artifact downloads. Their numerical scores are not mixed.
7. **Evidence index export:** a downloadable JSON summary records model/data fingerprints and saved experiment metadata, without runtime tokens.
8. **Supervised launcher:** one command starts both children, waits for readiness and terminates only its own child processes on exit. It refuses to take over occupied ports.
9. **Explicit offline behavior:** launching the original Helix-only command still provides the hub, saved metadata and GFP engine, while PET API requests explain how to start the combined service.

### Architecture

```text
Browser: http://127.0.0.1:8765
  ├─ /                  Research hub / evidence / registry
  ├─ /evolution         Helix UI → Flask evolution engine → runs/
  ├─ /analysis          Conditions, structure views, comparison → analysis_runs/
  └─ /pet               Existing React PET UI
       └─ allowlisted API gateway → authenticated local PET worker :8766
            └─ existing FastAPI services → PET models / features / structures
                 └─ successful rankings → research_runs/ with provenance
```

## 6. How to run in VS Code

Open `D:\Enzyme_ML_new`, select Terminal → New Terminal, then run:

```powershell
.\Start-Research.cmd
```

Open `http://127.0.0.1:8765/`. Keep the terminal open. Ctrl+C stops the launcher-owned services. If a previous Helix server occupies 8765, stop its terminal first; the launcher does not kill unknown listeners. The old `Start-Helix.cmd` runs only the GFP engine and hub.

Installed runtime requirements: this project's `.venv`, PET project's `.venv-1`, and PET `frontend/dist` build. The integration does not automatically start or install Ollama; local explanation functionality is optional and remains unavailable when Ollama/model is absent. Training new models may need downloads or a GPU; cached PET recommendations use local features.

If folders change:

```powershell
$env:ENZAIME_ROOT = 'D:\path\to\EnzAIme_ML'
$env:ENZAIME_PYTHON = 'D:\path\to\EnzAIme_ML\.venv-1\Scripts\python.exe'
.\Start-Research.cmd
```

## 7. A substantial final-year scope

Suggested title: **EnzAIme–Helix: An Evidence-Tracked Platform for Condition-Aware Enzyme Discovery and Protein Directed Evolution**.

The engineering contribution is the integrated platform. A research contribution must be established through experiments. More screens or a combined title do not establish scientific novelty.

### Work package A — reproducibility and evidence

Use the implemented integration to demonstrate: catalog ranking → sequence/structure inspection → saved evidence → GFP strategy experiment → provenance export. Add stable source/version tracking and a portable installation/export workflow before distributing the system to another machine. Document upstream GPL-3.0 obligations and inspect PET component licensing before publication.

### Work package B — robust PET evaluation

Rebuild PET training with exact-sequence and, where feasible, sequence-family/study splits. Fit preprocessing within training folds. Compare training-mean/median, Ridge, and neural baselines using identical splits and target transformations. Report raw-unit and log-unit errors, R², per-protein performance and condition-range coverage. Repeated assays from one protein do not constitute independent proteins.

Select hyperparameters using training/validation evidence. Keep final test data out of selection. Do not claim success until the weak current checkpoint is improved on a defensible held-out evaluation. Small test sets require uncertainty intervals and disclosure of sample/group counts.

### Work package C — uncertainty and diversity experiments

Prespecify seeds, data budget, number of proposals, rounds and compute budget. Compare: random search; one-hot Ridge greedy; ESM greedy; ensemble mean greedy; ensemble uncertainty-only; ensemble diversity-only; uncertainty + diversity. Ensemble-mean greedy is needed to separate training an ensemble from changing acquisition. Several of these controls are planned, not yet exposed by the engine.

Use calibration data separate from final test data. Evaluate prediction-error versus uncertainty, interval coverage where a calibrated interval method is used, candidate diversity, runtime and independent fitness evidence. Add mutation-distance splits to reduce optimistic near-neighbor generalization. The implemented ensemble spread alone is not a calibrated confidence interval.

The candidate contribution is a carefully evaluated relationship between calibrated uncertainty, distance-aware selection and out-of-distribution protein search. This is a hypothesis for literature review and ablation, not an established novel algorithm.

### Work package D — stability and structure fusion

Train the prepared Tm branch and verify held-out performance and compatible encoders. Then evaluate sequence-only versus sequence+image on matched proteins and matched splits. Include shuffled-image and missingness controls so structure availability cannot masquerade as structural learning. Report stability and activity errors separately before using a Pareto frontier. No reliable mutation-effect model follows automatically from wild-type Tm training.

### Work package E — independent validation

Obtain the paper oracle or use a carefully designed measured benchmark with no train/test leakage. Novel PET mutants require variant-specific measured data or laboratory validation. Until then, mutation preview and model-suggested sequences remain proposals, and the platform must not claim improved plastic degradation.

## 8. Deliverables and acceptance criteria

| Deliverable | Current state | Completion criterion |
|---|---|---|
| Integrated local application | Implemented | Hub, both workflows, safe gateway, persistence and launcher verified |
| Comparison and architecture document | This document | Traceable to code/artifacts and updated after changes |
| Baseline replication | Partial | Matched settings, independent evaluator, all claimed benchmarks |
| PET prediction model | Existing weak checkpoint | Defensible held-out comparison against simple baselines |
| Stability prediction | Data + training/import code | Compatible trained weights and measured held-out performance |
| Multimodal benefit | Unproven | Matched-subset and shuffled-image ablations |
| Acquisition research | Comparator implemented | Calibration, fair controls, multiple seeds, independent evaluation |
| Novelty claim | Not established | Literature review plus experiments supporting a distinct contribution |
| Final report/demo | Foundation ready | Methods, limitations, evidence tables, reproducible instructions |

## 9. Local evidence sources

- PET `README.md`, `docs/rebuild_v2.md`, active `backend/app/workspace_app.py` and routed services.
- PET `models/neural/manifest.json` and `held_out_predictions.json`; `data/processed/tpa_pet_ml_confirmed.csv`.
- PET `data/stability/manifest.json`, `data/structure_images/manifest.json`.
- Helix `README.md`, `REPLICATION_STATUS.md`, `lab/engine.py`, `runs/*/manifest.json`.
- Integration `lab/integration.py`, `lab/pet_worker.py`, `launch_research.py`, and `tests/test_integration.py`.

Do not use older served-model audit metrics as if they described the rebuilt neural checkpoint. Historical Ridge artifacts and current neural artifacts are distinct systems.
