# Environmental integration and benchmark comparison

Checked against the installed source and saved results on 26 October 2026. This note clarifies the distinction between application integration, model inputs, predictive benchmarking and directed-evolution evaluation. No new training experiment was run to prepare this note.

## What environmental integration currently means

The shared application includes both PET condition analysis and an avGFP directed-evolution adaptation of the base paper. Temperature, pH and reaction duration are actual inputs to the PET models and their scenario analysis. They are not inputs to the avGFP predictor or its mutation-selection loop. Environmental inputs were already present in the original EnzAIme PET model; the extensions add condition windows, sensitivity analysis, richer assay-context experiments and reproducible comparisons.

The local `Directed_Evolution-main/preprocessed_data/avGFP/avGFP.csv` has 51,715 rows and exactly two columns: `sequence` and `fitness`. It supplies no per-observation temperature, pH or duration labels. Its size does not make it a suitable benchmark for learning environmental response. Assigning invented condition labels, or assigning one constant condition to every row, would not establish a condition-dependent model.

The implemented project therefore supports the claim “an integrated platform for condition-dependent PET analysis and GFP directed evolution.” It does not yet support “an environmentally conditioned version of the paper's directed-evolution algorithm.” The latter requires an additional model/search implementation and suitable measured labels.

## Comparisons already measured

### PET feature ablation on identical outer-fold observations

The earlier condition study uses 208 eligible measurements, 25 exact-sequence groups, five outer folds and three inner selection folds. All measurements of an exact sequence remain together. Each feature variant uses the same outer partitions and a common candidate-selection procedure; preprocessing is fitted only inside training partitions.

| Inputs | MAE (µM TPA), lower is better | RMSE (µM TPA) | R² | Spearman |
|---|---:|---:|---:|---:|
| Sequence only | 1589.94 | 5325.19 | -0.0875 | -0.3627 |
| Environment only | 1546.22 | 5291.59 | -0.0738 | 0.2152 |
| Sequence + environment | 1325.30 | 5219.30 | -0.0447 | 0.4792 |
| Sequence + environment + structure descriptor | 1540.18 | 5301.51 | -0.0779 | -0.0325 |

Sequence + environment has 16.6% lower MAE than sequence only in this experiment. This is evidence about feature sets on the local PET cohort. It is not a comparison against the published GFP algorithm. All R² values remain negative, and the structural descriptor did not improve the sequence + environment result. These are internal grouped evaluations, not a new external benchmark dataset.

### Original EnzAIme checkpoint on its historical test

| Method | Test rows | MAE (µM TPA) | R² |
|---|---:|---:|---:|
| Original EnzAIme neural checkpoint | 18 | 46.7844 | -2.2965 |
| Environment-only ablation | 18 | 38.4906 | -1.5716 |
| Sequence + environment surrogate | 18 | 141.4538 | -26.6270 |
| Sequence + environment + structure descriptor | 18 | 99.1337 | -13.3089 |

These rows share five held-out protein groups. This historical test has already informed development and should be labelled exploratory. The environment-only method has 17.7% lower MAE but cannot distinguish proteins at identical conditions. The combined sequence/environment/image surrogate is worse than the original checkpoint. None of these results establishes overall superiority of the integrated project. Do not compare the absolute errors in this table with the 208-row table: the target distributions and evaluation designs differ.

### Later assay-context study

The 26-configuration nested study reports both log1p-MAE-selected and raw-MSE-selected sequence + environment models. On sequence-grouped folds, log1p-MAE selection yields MAE 1404.64 µM, MSE 27,396,707 µM², RMSE 5234.19 µM and R² -0.0507; raw-MSE selection yields MAE 1452.01 µM, MSE 26,930,460 µM², RMSE 5189.46 µM and R² -0.0328. Thus the raw-MSE objective slightly lowers squared error but worsens MAE. On stricter sequence-and-study folds, log1p-MAE selection yields MAE 1495.47 µM, MSE 27,977,197 µM², RMSE 5289.35 µM and R² -0.0729; raw-MSE selection yields MAE 1356.88 µM, MSE 25,415,565 µM², RMSE 5041.39 µM and R² 0.0253. Each outer fold chooses its model using only inner folds. This is exploratory validation on the same 208-record dataset, not an external confirmation or evidence that the full-data refit is independently tested; the current live PET checkpoint was not replaced.

### GFP predictive evaluation

The saved one-hot Ridge run used 2,000 sampled avGFP sequences, with 1,600 training and 400 test observations. MAE is 0.533985, RMSE is 0.640980 and Spearman correlation is 0.805200. This is a predictive evaluation on a sample of a benchmark dataset; 0.805200 is not 80.52% accuracy. Existing tiny ESM demonstrations use different sample budgets/seeds, so their scores do not constitute a fair matched method comparison.

## What is required for a direct base-paper comparison

The official paper implementation exposes population, proposal count, mutation model, wild-type starting sequence and predictor checkpoint as experimental settings. The PEX implementation referenced by the local replication notes provides separate downloadable landscape-oracle models. The required compatible evaluator is not installed in the local GFP workflow. Sources: [official Directed_Evolution implementation](https://github.com/HySonLab/Directed_Evolution), [official PEX implementation](https://github.com/HeliXonProtein/proximal-exploration).

Two different evaluation questions must be reported separately:

| Question | Required experiment | Present status |
|---|---|---|
| How well do predictors estimate measured GFP fitness? | Ridge and ESM-based predictors on identical saved train/validation/test sequences and preprocessing; matched tuning procedure; multiple seeds; MAE, RMSE and rank correlation | One baseline result and small execution demonstrations; matched comparison incomplete |
| Does the proposed search design better proteins than the paper's search? | Reproduce the paper configuration and evaluate all searches with the same independent oracle, starting sequence, budget, candidate count and repeated seeds; report fitness, diversity and novelty under matched definitions | Incomplete; compatible independent evaluator and matched runs missing |
| Do environmental inputs help PET prediction? | Sequence-only, condition-only and combined models on identical grouped splits; matched selection; report raw/log errors and uncertainty | Initial internal ablations complete; independent confirmation remains needed |
| Does environmental conditioning improve sequence design? | Train on measured sequence-condition-outcome tuples, condition the search objective, and evaluate generated candidates independently under those conditions | Not implemented or demonstrated |

For a stronger environmental study, obtain measured outcomes for multiple sequences or mutants under varying assay conditions, preserve units/protocols, hold out protein families and studies where feasible, and compare matched models with and without conditions. Environmental robustness from model-generated scenarios must remain separate from measured performance. An independent external cohort and group-level uncertainty estimates would strengthen the existing PET evidence.

## Wording supported for the manuscript

“On a 208-observation PET cohort, nested sequence-grouped evaluation found that sequence-plus-environment features reduced MAE by 16.6% relative to sequence-only features, although R² remained negative. The integrated application also implements an avGFP directed-evolution adaptation. Environmental conditioning of that adaptation and a matched comparison against the base paper's independent-oracle results remain future work.”

Primary local evidence: `models/condition_analysis/report.json`, `models/assay_v2/train-20261024-160717/report.json`, `runs/20261023-162906-a42ad8/status.json`, `lab/engine.py`, `lab/assay_model.py` and `REPLICATION_STATUS.md`. The existing journal `metric_tables.csv` provides cohort-labelled numeric results. Comparisons should report failures and limitations alongside improvements; selecting only a favourable metric would misrepresent these experiments.
