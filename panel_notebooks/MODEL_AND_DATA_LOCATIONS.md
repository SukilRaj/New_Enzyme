# Exact model and dataset locations

All notebook-package paths below are relative to the extracted panel_notebooks folder. Original project paths are included for tracing existing application artifacts. New demonstration models live under the output directories printed by each notebook.

| Model | Notebook-package file | Original project file | Evaluation scope |
|---|---|---|---|
| Original PET neural | `pet/models/neural/weights.pt` | `D:/Final_Year_Project/EnzAIme_ML/models/neural/weights.pt` | Historical 18 rows / 5 sequence groups |
| Condition/image surrogate | `models/condition_analysis/model.pkl` | `D:/Enzyme_ML_new/models/condition_analysis/model.pkl` | Historical holdout; grouped results in separate CSV |
| Assay full-data refit | `models/assay_v2/train-20261024-160717/model.pkl` | `D:/Enzyme_ML_new/models/assay_v2/train-20261024-160717/model.pkl` | Use saved out-of-fold CSVs; refit has no independent holdout |
| GFP Ridge | `runs/20261023-162906-a42ad8/baseline.pkl` | `D:/Enzyme_ML_new/runs/20261023-162906-a42ad8/baseline.pkl` | Original 400-sequence holdout |
| GFP ESM head | `runs/20261024-092703-afed29/decoder.pt` | `D:/Enzyme_ML_new/runs/20261024-092703-afed29/decoder.pt` | Original 20-sequence execution demonstration |
| Assay historical split model | `models/assay_v2/train-20261024-160717/historical_model.pkl` | `D:/Enzyme_ML_new/models/assay_v2/train-20261024-160717/historical_model.pkl` | Historical training-only fit; evaluate with historical_predictions.csv |
| GFP three-head ESM checkpoint | `runs/20261024-093528-uncertainty-diversity/decoder.pt` | `D:/Enzyme_ML_new/runs/20261024-093528-uncertainty-diversity/decoder.pt` | Original saved execution-study split; full engine code in notebook 03 |
| Frozen ESM-2 8M encoder / masked LM | `models/esm2_8m/model.safetensors` | `D:/Enzyme_ML_new/models/huggingface/hub/models--facebook--esm2_t6_8M_UR50D/snapshots/c731040fcd8d73dceaa04b0a8e6329b345b0f5df/model.safetensors` | Pretrained weights; GFP task predictions require a fitted head |
| Frozen ESM-2 35M encoder / masked LM | `models/esm2_35m/model.safetensors` | `D:/Enzyme_ML_new/models/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8/model.safetensors` | Pretrained weights used for PET features; not a fitted PET activity predictor |

## Datasets

- PET confirmed assay data: `pet/data/processed/tpa_pet_ml_confirmed.csv` (221 source records; 208 eligible modeling rows).
- PET raw collected records: `pet/master_dataset.csv`; `pet/data/tpa_pet_ml_dataset.csv`; recorded exclusion review under `pet/reports/results/`.
- PET frozen residue features: `pet/data/neural_features/index.json` and its 25 NPZ files.
- Reference coordinates/images: `pet/data/structure_images/` (13 sequence-matched proteins).
- avGFP benchmark: `Directed_Evolution-main/preprocessed_data/avGFP/avGFP.csv` (51,715 sequences).
- Original PET dataset location: `D:/Final_Year_Project/EnzAIme_ML/data/processed/tpa_pet_ml_confirmed.csv`.

Load each checkpoint with its accompanying preprocessing objects, architecture/normalization metadata and matching encoder. An encoder file is not the separately trained activity/fitness head. Notebook 01 evaluates PET artifacts; notebook 02 evaluates GFP artifacts.
