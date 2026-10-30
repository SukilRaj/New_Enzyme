# Condition analysis and protein images: methods and results

Implemented and evaluated locally on 24 October 2026. This extends the integrated EnzAIme × Helix application. It is an engineering and research extension; scientific novelty and superiority to the base paper have not been established.

## Run and demonstrate

Open `D:\Enzyme_ML_new` in VS Code, run `Start-Research.cmd`, and visit **http://127.0.0.1:8765/analysis**. Keep the terminal open. Use Ctrl+C to stop both servers. The same page is linked from the research hub, PET workspace and GFP workspace.

The application still needs the existing PET directory at `D:\Final_Year_Project\EnzAIme_ML`, its `.venv-1`, cached features, model and structures. `ENZAIME_ROOT` and `ENZAIME_PYTHON` override this location. `Start-Helix.cmd` alone does not start the original PET neural model.

1. **Environmental analysis:** select a catalog protein, choose the original neural checkpoint, and run the default 40 °C / pH 8 / 48-hour window. Inspect the temperature–pH response, one-variable sweeps and catalog ranking. Download the analysis JSON.
2. **Protein image studio:** select one of the 13 proteins with a sequence-matched structure. Generate five views and download the ZIP with its provenance manifest.
3. **Model comparison:** inspect both evaluation tables and their cohort sizes. Export predictions, fold membership, protocol and results.
4. **Experiment registry:** the hub contains saved environmental analyses alongside PET recommendations and GFP experiments, preserving each task's units.

## What has been integrated

| Capability | Before | Implemented extension |
|---|---|---|
| Environmental inputs | PET neural model already accepted temperature, pH and duration | Response surfaces, sweeps, scenario summaries, training-range checks and condition-window ranking |
| Protein images | Cached structure-derived distance images and a molecular viewer in the PET project | Five reproducible downloadable views, strict full-sequence matching and per-file provenance |
| Model comparison | Historical checkpoints and separate reports | A common fixed-test comparison, nested grouped evaluation, ablations and downloadable predictions |
| Structure fusion | Original neural image branch untrained | Separate trained experimental surrogate with structure descriptors; its poor results are disclosed |
| Persistence | GFP runs and integrated PET recommendation records | Environmental analysis records with configuration, model and dataset fingerprints |

These additions do not attach unsupported PET condition effects to GFP. The provided GFP benchmark has no corresponding temperature/pH/duration labels. The two domains share the application, registry and evaluation conventions, but have distinct predictors and targets.

## Environmental score definition

For each protein, query the selected model on three equally spaced values along each requested condition interval: normally 27 combinations. Intervals are clipped to the application's input bounds: temperature 0–100 °C, pH 0–14, duration 0.01–720 hours. A zero-width interval collapses to one value. All proteins use the same grid.

The robustness score is the **10th percentile of predicted TPA concentration across that discrete grid**. Larger values rank first. This is a descriptive scenario score, not a calibrated lower confidence bound, an experimentally established optimum or evidence of improved plastic degradation. Scenario standard deviation measures variation across conditions; it is not epistemic uncertainty.

The selected protein also receives a 7 × 7 temperature–pH surface at fixed duration and 13-point one-variable sweeps, with the other inputs fixed at the center. Actual grid sizes shrink when an interval has zero width. The best surface point is a model hypothesis.

Training-condition minima/maxima and distance to the nearest standardized training-condition observation are computed using the original training split. The ranges are 30–80 °C, pH 7–9 and 1–672 hours. Being within these marginal ranges does not establish support for a particular combination or protein. The JSON includes measured observations separately from predictions and identifies whether each was used for training.

The default engine batches queries through the original ESM-2 + Attention1D checkpoint. Its center predictions were checked against the existing recommendation endpoint. The experimental fusion engine is opt-in because its test performance is worse.

## Evaluation design

The confirmed PET source contains 221 rows. Following the original neural training pipeline's filtering and deduplication leaves **208 observations across 25 exact sequence groups**. The target is micromolar TPA concentration; models fit `log(1 + TPA)` and report both raw-unit and log-unit errors. Enzyme dosage and assay material are not included as inputs, so heterogeneous assays remain a major limitation.

### Fixed historical split

The experiment uses the exact sequence hashes from the original neural manifest: 172 training observations, 18 validation observations and 18 test observations. The test contains five protein groups. The saved original predictions are checked against protein IDs and observed values before comparison.

All scalers and PCA transformations are fitted on training data only. Sequence preprocessing fits on one representative per training sequence group. Every ablation selects among the same five prespecified configurations using validation log-RMSE. The candidate list is written to `protocol.json` before fitting:

- Ridge with alpha 10, 100 or 1000.
- ExtraTrees: 160 trees, maximum depth 6, minimum leaf size 4, seed 42.
- Histogram gradient boosting: 100 iterations, 7 leaves, minimum leaf size 10, L2 penalty 10, seed 42.

The five ablations are sequence only; environment only; sequence + environment; sequence + environment + structure-availability flag; and sequence + environment + structure descriptors. There are 25 configurations in total. The experimental serving candidate is chosen on validation log-RMSE among the last two substantive input combinations (sequence + environment and full multimodal). The selected candidate is multimodal Ridge with alpha 1000. Model selection does not overwrite the original checkpoint.

The historical test had already been inspected before this work. It therefore supports **exploratory comparison**, not fresh external validation. Scores were not used to extend the candidate search or repeatedly tune a replacement.

### Nested grouped evaluation

Five outer `GroupKFold` splits cover all 208 observations once. Three inner grouped folds select configurations within each outer training fold. All measurements of an exact sequence remain together. Preprocessing is fitted anew within each training fold. The exported `cv_splits.json` contains outer and inner membership and the selected configuration for each ablation/fold.

The broader experiment evaluates sequence, environment, sequence + environment and multimodal ablations. It does not retrain the original neural checkpoint in each outer fold. Its cohort and target distribution differ from the fixed historical test, so errors between the two tables must not be compared directly. Related protein families and publications can still cross folds; exact-sequence grouping is not family- or study-independent validation.

## Measured results

### Same 18 held-out observations / 5 proteins

| Model | MAE (µM) ↓ | RMSE (µM) ↓ | R² ↑ | Log-RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|---:|
| Original EnzAIme neural checkpoint | 46.7844 | 53.4576 | −2.2965 | 2.2077 | −0.7095 |
| Environment only | 38.4906 | 47.2158 | −1.5716 | 2.0599 | −0.6535 |
| Sequence only | 258.3140 | 270.9580 | −83.6900 | 3.4730 | −0.7241 |
| Sequence + environment | 141.4538 | 154.7579 | −26.6270 | 2.9954 | −0.5873 |
| Sequence + environment + availability | 144.9620 | 158.3680 | −27.9310 | 3.0152 | −0.6246 |
| Sequence + environment + structure images | 99.1337 | 111.3755 | −13.3089 | 2.6651 | −0.2807 |
| Training log-mean constant | 257.5119 | 259.1896 | −76.4932 | 3.4263 | undefined |

The environment-only baseline has **17.7% lower MAE** than the original neural checkpoint on this historical test. It cannot distinguish proteins at identical conditions and is not a replacement for protein ranking. Its negative R² and negative correlation indicate weak predictive performance despite lower error. Negative R² means squared error exceeds the test-mean reference used in that metric.

The selected structure-fusion candidate has **99.13 µM MAE**, compared with **46.78 µM** for the original. It remains experimental. A 1,000-resample paired protein bootstrap estimates an MAE difference of +52.35 µM with a percentile 95% interval of approximately +24.95 to +167.29 µM. Only five test groups contribute, so this interval is an exploratory diagnostic with substantial limitations.

### Nested grouped out-of-fold predictions: 208 observations / 25 sequence groups

| Ablation | MAE (µM) ↓ | RMSE (µM) ↓ | R² ↑ | Log-RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|---:|
| Sequence only | 1589.9354 | 5325.1871 | −0.0875 | 2.7824 | −0.3627 |
| Environment only | 1546.2188 | 5291.5931 | −0.0738 | 2.4493 | 0.2152 |
| Sequence + environment | 1325.3023 | 5219.2997 | −0.0447 | 2.1784 | 0.4792 |
| Sequence + environment + structure images | 1540.1842 | 5301.5145 | −0.0779 | 2.6212 | −0.0325 |

Sequence + environment has **16.6% lower MAE** than the sequence-only ablation on the same outer-fold observations. This supports further study of condition features in this dataset, not a claim of superiority to the original neural model or the paper. No reliable structural-image benefit is established. R² remains negative, and uncertainty in these improvements has not been resolved with independent data.

## Feature and image construction

Sequence features are mean-pooled cached ESM-2 35M residue embeddings, standardized and reduced to up to five PCA components using training groups only. Environmental descriptors are temperature, pH, log(1 + hours), squared distance from pH 7, reciprocal absolute temperature scaled by 1000, and a temperature–log-duration product. These are fixed numerical descriptors, not mechanistic kinetic laws.

The image descriptor is an 8 × 8 block average of the C-alpha distance image plus an availability indicator. Missing structures use a zero descriptor and flag. This uses structural geometry, not a newly trained CNN or the original neural image branch. The original checkpoint still has `image_trained=false`. Predictions from the new regressors are transformed back with `expm1`; log predictions are clipped to [0, 20] as a prespecified finite numerical bound.

For images, the parser selects the first PDB model, resolves alternate atoms by occupancy, checks finite coordinates and requires a chain matching the complete reference sequence. It does not substitute a merely similar protein. Available structures are AlphaFold predictions, not experimentally determined structures validated by this project.

Each generated bundle contains:

- `front.svg`, `side.svg`, `top.svg`: C-alpha backbone projections on principal coordinate axes, with sequence-order coloring. Apparent chain gaps over 5 Å are not joined.
- `distance.png`: a residue-by-residue C-alpha distance map, capped at 20 Å; brighter pixels mean closer residues.
- `contacts.png`: C-alpha proximity at ≤8 Å, including diagonal/near-neighbor proximity. It is not an atom-level contact predictor.
- `manifest.json`: protein and sequence identifiers, source structure hash, generator hash, representation and hashes of all five images.

The generator renders existing coordinates. It does not generate a new fold, validate a mutation, or create new experimental training labels. Thirteen structures are currently available; the other 12 proteins remain explicitly unsupported for image generation. Structure availability may be confounded with protein identity and literature coverage. Matched-subset and shuffled-structure controls would be needed before an image-benefit claim.

## Relationship to the base paper

The supplied paper reports MLDE avGFP maximum fitness **9.50 ± 0.37** in Table II and average fitness **9.33 ± 0.40** in Table V (printed pages 423–424; PDF pages 6–7). Source: Tran & Hy, *Protein Design by Directed Evolution Guided by Large Language Models*, DOI 10.1109/TEVC.2024.3439690.

Those are scores for generated sequences under the paper's separate evaluation oracle. The local GFP adaptation lacks that evaluator. PET concentration errors are another task, scale and outcome. Comparing 9.50 with a PET error or a locally optimized GFP surrogate score cannot establish superiority. A valid paper comparison requires the matching evaluator, settings, starting sequences, budgets and repeated trials.

Environmental inputs already existed in EnzAIme_ML, and condition modeling and structural representations are established techniques. The defensible project contribution at this stage is their traceable integration, visualization, scenario analysis and controlled evaluation. A stronger algorithmic novelty claim needs a literature review and independent experiments. Adding more features alone does not establish novelty.

## Reproduce and inspect

Training (recreates the same bounded experiment; does not tune against reported test scores):

```powershell
.\.venv\Scripts\python.exe scripts\train_condition_models.py
```

Verification with both local servers running:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_analysis.py
node --check lab\static\analysis.js
```

Artifacts:

| Location | Content |
|---|---|
| `models/condition_analysis/` | Model, selection protocol, fixed predictions, out-of-fold predictions, splits and report |
| `analysis_runs/` | Saved configuration, condition predictions, rankings, observed evidence and fingerprints |
| `generated_structures/<protein_id>/` | Five generated views and provenance |
| `reports/analysis_verification.json` | Live API verification, model consistency and ZIP hashes |
| `tests/test_analysis.py` | Split disjointness, score recomputation, model selection, condition bounds, deterministic geometry and API restrictions |

Local verification confirms that the implemented software and recorded calculations work. It does not establish biological activity or clinical/environmental outcomes. Future stronger evaluation should use new measurements, sequence-family/study splits, matched image controls, assay covariates and independent validation of suggested conditions.

## Assay-aware training study (assay_v2)

The integrated **Training laboratory** also exposes a separate, reproducible assay-context study. It parsed only structured, pre-assay fields from source records using an explicit whitelist; it did not use outcome text, free-form evidence, or study identifiers as predictive features. The available context variables are enzyme concentration (four unit forms), substrate concentration, reaction volume, PET crystallinity and agitation. Values retain their original unit dimensions, with categorical preparation/material metadata and missingness indicators. The report records coverage by feature. Because coverage is uneven and source studies use different protocols, missing assay conditions remain a major limitation.

The updated search compares 26 prespecified configurations across context-only, sequence-plus-environment and sequence-plus-context representations, with ridge, ExtraTrees, histogram boosting, kernel ridge and a small seeded neural ensemble. Two additional sequence-plus-environment candidates fit the target directly in µM rather than log1p(µM), to test whether that target scale improves errors dominated by extreme concentrations. Nested evaluation reports two selection objectives: mean absolute log1p error averaged equally over sequence groups (the existing primary recipe selection), and raw squared error in µM, first averaged within each sequence group and then equally across groups. Both objectives select only on inner folds; their outer-fold predictions are compared separately. Preprocessing and PCA are fit inside training folds. Nested five-fold validation is run with sequence groups and with connected sequence/study groups; the latter prevents a shared sequence or study from crossing the outer train/evaluation boundary. The primary selected recipe is refit on all 208 records for artifact inspection. This final fit has no independent evaluation set and must not be treated as a validated deployment model.

| Outer evaluation | Inner-fold selection objective | MAE (µM) | MSE (µM²) | RMSE (µM) | R² | Spearman | Within 2× on 1 + TPA |
|---|---|---:|---:|---:|---:|---:|---:|
| Sequence-grouped, sequence + environment | Group-balanced log1p MAE | 1,404.6 | 27,396,707 | 5,234.2 | −0.051 | 0.530 | 22.1% |
| Sequence-grouped, sequence + environment | Group-balanced raw MSE | 1,452.0 | 26,930,460 | 5,189.5 | −0.033 | 0.387 | 23.1% |
| Sequence-grouped training-median baseline | Training median | 1,600.5 | 27,866,892 | 5,278.9 | −0.069 | −0.611 | 10.6% |
| Sequence-and-study-grouped, sequence + environment | Group-balanced log1p MAE | 1,495.5 | 27,977,197 | 5,289.3 | −0.073 | 0.319 | 16.8% |
| Sequence-and-study-grouped, sequence + environment | Group-balanced raw MSE | 1,356.9 | 25,415,565 | 5,041.4 | 0.025 | 0.617 | 25.5% |
| Sequence-and-study-grouped training-median baseline | Training median | 1,585.4 | 27,955,678 | 5,287.3 | −0.072 | −0.494 | 9.1% |

MSE is in squared concentration units (µM²). The 2× criterion is `abs(log1p(observed) - log1p(predicted)) <= log(2)`, using numeric µM values. It compares **1 + TPA**, not raw concentration; this distinction matters near zero. Each outer fold selects a model using only its own inner folds. The final ExtraTrees refit uses the log1p-MAE selection rule and is a separate artifact from these out-of-fold predictions.

Compared with log1p-MAE selection, raw-MSE selection reduces sequence-grouped pooled MSE by 1.7% and RMSE by 0.9%, with R² moving from −0.051 to −0.033; MAE increases by 3.4%. Under the stricter sequence-and-study split, its MSE is 9.2% lower, RMSE 4.7% lower, MAE 9.3% lower, and R² moves from −0.073 to 0.025. These are exploratory nested cross-validation results on the same small dataset, not external validation; the sequence-grouped raw-MSE result still has negative R². The large RMSE/P90 errors reveal poor performance on extreme values. Adding assay context did not win selection; it is shown as an ablation rather than advertised as an accuracy gain. A separately held historical 18-row test was involved in prior iteration and is not an untouched confirmation set; do not compare its scores against another model or the paper as if the evaluation were controlled. Neither research model is promoted over the existing PET checkpoint.

The latest run is identified by `models/assay_v2/latest.json`; its `report.json`, `protocol.json`, `context_audit.json`, `splits.json`, and sequence / sequence-study out-of-fold CSVs contain the reproducible evidence. Recreate the study with:

```powershell
.\.venv\Scripts\python.exe scripts\train_assay_models.py
```

### Verification, data audit and baseline comparison

The saved out-of-fold predictions were recomputed independently from the latest run's `sequence_oof.csv` and `sequence_and_study_oof.csv`. MAE, MSE, RMSE and R² reproduce the `report.json` values to 0.0 difference for every mode on both designs. Both split designs were re-checked for disjointness: no exact sequence and no study crosses an outer train/evaluation boundary, all 208 rows are covered once, and each design has 5 folds (the connected sequence-plus-study design collapses 25 sequences and 22 studies into 19 components). The modeled cohort rebuilds reproducibly: 221 confirmed rows → 213 after finite/positive-duration and nonnegative-target filters → 208 after exact duplicate removal.

Target audit (208 rows): range 0.3–47,000 µM, median 234.5 µM, mean 1,556.0 µM, with 5 values above 10,000 µM. Units arrive in four forms and are standardized upstream: mM × 1000 (27 rows), µM × 1 (137), µmol/L × 1 (45) and µg/mL × 1000 / 166.13 using the TPA molecular weight (12). The µg/mL recomputation matched the stored standardized value to 1.1 × 10⁻¹³. Signed bias on the selected recipe is −1,316.7 µM (systematic underprediction), concentrated in the >1,000 µM group (MAE 4,962.8, bias −4,962.8) while the 0–10 µM group is overpredicted by +102.7 µM.

Protein 37 (PHL-7) is the dominant error source: 8 records, median 17,000 µM, MAE 20,070.8 µM, spanning two studies (`10.1002/cssc.202101062`, `10.1021/acscatal.2c02275`). The existing targeted audit (`reports/results/ablation_error_analysis/targeted_audit/`) found the values internally consistent across dataset, JSON and provenance records but recorded as approximate digitized values from a figure; a transcription error could not be verified and the raw replicate/figure values were not retained. These records were therefore **left unchanged** — not capped, relabeled or removed — pending the original source values.

A prespecified baseline comparison (`scripts/compare_assay_baselines.py`) reuses the frozen outer folds and evaluates training-fold mean/median constants, whole-cohort constants, and fixed-hyperparameter Huber/Ridge models on the same sequence-plus-environment features. It writes `baseline_comparison.json` / `.csv` beside the run artifacts (exposed through the training artifact allowlist).

| Design | Model | MAE (µM) | RMSE (µM) | R² |
|---|---|---:|---:|---:|
| Sequence-grouped | Selected sequence + environment (log-error) | 1,404.6 | 5,234.2 | −0.051 |
| Sequence-grouped | Raw-MSE-selected sequence + environment | 1,452.0 | 5,189.5 | −0.033 |
| Sequence-grouped | Training-median baseline | 1,600.5 | 5,278.9 | −0.069 |
| Sequence-grouped | Ridge (α=1000), fixed | 1,508.7 | 5,290.0 | −0.073 |
| Sequence-grouped | Huber (ε=1.35, α=1.0), fixed | 2,026.6 | 5,821.0 | −0.299 |
| Sequence-and-study-grouped | Raw-MSE-selected sequence + environment | 1,356.9 | 5,041.4 | 0.025 |
| Sequence-and-study-grouped | Selected sequence + environment (log-error) | 1,495.5 | 5,289.3 | −0.073 |
| Sequence-and-study-grouped | Training-median baseline | 1,585.4 | 5,287.3 | −0.072 |
| Sequence-and-study-grouped | Ridge (α=1000), fixed | 1,499.3 | 5,285.3 | −0.073 |

No candidate dominates: the log-error recipe gives the lowest sequence-grouped MAE, the raw-MSE recipe gives the lowest sequence-and-study MAE and is the only headline evaluation with nonnegative R², and the simple training-median baseline is within a few percent of both on MAE. Nothing here is promoted to serving.

### Deployment status of the PET endpoint

The live PET endpoint (`backend/app/api/neural.py` via `services/neural_recommender.py` in the EnzAIme backend) loads the original neural checkpoint at `models/neural/` (AttentionOracle, `image_trained: false`, 208 rows, `status: research_checkpoint_not_validated_for_mutation_effects`). It does **not** load this integrated assay_v2 model. The assay_v2 recipe remains a research artifact; promoting it would require a documented evaluation gate and explicit approval, which has not happened.

This work improves the audit trail and comparison of plausible feature sets. It does not demonstrate high accuracy, prove that more training alone will fix the data limitations, validate a biological recommendation, or establish a novel scientific method.


