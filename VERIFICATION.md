# Local verification

## Handover and FASTA workflow — 26 October 2026

- Full Python suite: **24 tests passed**, including independent recomputation of reported training metrics, nested sequence/study separation, source/model fingerprints, shared routes, and rejection of FASTA before coordinate inspection.
- Frontend: production build and ESLint passed; **6 sequence-file tests passed** for parsing, one-based substitution recovery, malformed input, length mismatch and renamed FASTA detection.
- Real-browser workflow: **10 assertions passed** for the exact S121E/D186H/R280A example, reference mapping at all three sites, FASTA export/import, unchanged reference PDB download, canvas controls, state retention, file guidance and navigation. No browser JavaScript errors. Evidence: `reports/sequence_workflow_verification.json`.
- The maintained React source/build is now in `Enzyme_ML_new/frontend/`; the original PET backend/data/models remain linked. This supersedes the earlier note that the frontend must be served from the original project.
- Existing training metrics are unchanged. The new workflow does not train a folding model or validate mutant function.

## Completed CPU experiment

Run ID: `20261023-162906-a42ad8`.

- Started from the browser UI and completed all ten rounds.
- Dataset: avGFP; 2,000 sampled unique sequences; seed 42.
- Training: 1,600 sequences. Held-out test: 400 sequences.
- Model: one-hot position-specific ridge regression, alpha 10.
- Test RMSE: 0.6409801154.
- Test MAE: 0.5339851002.
- Test Spearman correlation: 0.8052003450.
- Candidate CSV endpoint returned HTTP 200 with 16 rows.
- Browser refresh retained the completed experiment in history.
- Dashboard, dataset distribution, and candidate library inspected in the browser.
- Core environment: `pip check` passed.

These metrics describe one CPU baseline experiment, not the paper's LLM results or experimentally validated protein properties.

## ESM workflow and uncertainty/diversity comparator smoke run

The local 8M ESM model completed a one-epoch run on CPU with 100 sampled sequences, an 80/20 split, one evolution round, and two candidates. The single-head ESM adapter reported RMSE 1.515, MAE 1.156, and Spearman ρ -0.017 on the held-out 20 sequences (run `20261024-092703-afed29`).

The uncertainty-and-diversity implementation then completed with seed 44 and the same small budget (run `20261024-093528-uncertainty-diversity`). It trained three bootstrap heads and returned two candidates with mean ensemble spread 0.466 and mean pairwise Hamming distance 1.0; held-out RMSE was 1.397, MAE 1.186, and Spearman ρ 0.364. This single smoke run verifies that the code path executes and saves uncertainty, selection score, and diversity summaries. It is far too small to compare methods or support a novelty claim. The scores are predictions against a tiny random split, not validated protein measurements.

The engine uses the cached pinned ESM snapshot locally when all required files are present, avoiding network access during local runs. A server that was already running before the engine update must be restarted before submitting the new selection option.

## Automated checks

The core tests passed for input validation, deterministic runs, train/test sequence disjointness, elitist selection, cancellation, and local API access restrictions. Python compilation and JavaScript syntax checks passed. The optional ESM integration test uses tiny randomly initialized weights to exercise actual tensor operations without a download; it is not a pretrained-model quality evaluation.
# Integrated platform verification — 24 October 2026

- Full local suite: **10 tests passed** (`.venv\Scripts\python.exe -m unittest discover -s tests -v`). This includes the existing real tiny-tensor ESM test, deterministic baseline tests, and new gateway/persistence/size-limit tests.
- JavaScript syntax check passed for `lab/static/research.js`.
- Both engines started through `launch_research.py`: Flask gateway on 8765 and the authenticated PET worker on 8766, using their existing separate environments.
- Real API checks passed: neural checkpoint load, three PET recommendations, saved artifact round trip, exact-sequence structure mapping (coverage 1.0), ten residue attributions, mutation preview, and a registry containing both PET and avGFP records.
- Browser check passed: integrated PET UI produced five recommendations at 40 °C, pH 8 and 48 hours. The shared registry displayed both new PET records alongside existing GFP experiments.
- Evidence: `reports/integration_verification.json`; browser preview: `reports/integrated-workspace.png`.
- PET evaluation observed: R² −2.296455, MAE 46.784351 µM, RMSE 53.457619 µM, 18 held-out observations / 5 proteins. These are existing checkpoint results, not improved training results from the integration.
- Optional Ollama explanations were offline during verification. Stability weights and trained image fusion were unavailable; no new biological or predictive-validity claim is made.
- Portability limitation: PET source, frontend build, model/features and `.venv-1` remain in `D:\Final_Year_Project\EnzAIme_ML` and are required by the linked integration.

## Condition and image studio verification — 24 October 2026

- Full local suite: **19 tests passed**. New checks recompute stored scores from predictions, verify group disjointness and outer-fold coverage, confirm validation-based selection and the model fingerprint, and exercise numeric bounds, zero-width condition windows, structure matching, deterministic rendering and API restrictions.
- Live verification passed using both real services: the batched original-model center agrees with the existing recommendation endpoint; 27 condition scenarios, 49 heatmap cells and three 13-point sweeps are returned; all 25 proteins are ranked; the experimental model also runs; saved analyses round-trip through the shared registry.
- Image generation passed for IsPETase: all five images are served, the ZIP contains five views plus its manifest, and downloaded hashes match. The catalog exposes 13 available sequence-matched structures.
- Out-of-bounds conditions are rejected. The original PET checkpoint is unchanged and remains the default.
- Browser checks passed for a real condition analysis, the measured-observation panel, all five generated images, and both comparison tables. Screenshots: `reports/condition-analysis.png`, `reports/protein-images.png`, `reports/model-comparison.png`. JavaScript syntax checks passed after the final UI edits.
- Evidence: `reports/analysis_verification.json`, `models/condition_analysis/report.json`, saved predictions and split manifests. Detailed interpretation: `docs/ANALYSIS_METHODS_AND_RESULTS.md`.
- Fixed-test environment-only MAE is 38.4906 µM versus original 46.7844 µM, an exploratory 17.7% reduction. The selected fusion model is worse at 99.1337 µM. Nested grouped sequence+environment MAE is 1325.3023 µM versus sequence-only 1589.9354 µM, a 16.6% reduction on that different cohort. R² remains negative. These do not establish superiority to the paper or biological validity.

