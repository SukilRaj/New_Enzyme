# Model and dataset locations

These paths distinguish the original application artifacts from their portable notebook copies. The portable copies are complete for the default demonstrations.

| Model | Notebook-package file | Original project file | Evaluation scope |
|---|---|---|---|
| Original PET neural | pet/models/neural/weights.pt | D:/Final_Year_Project/EnzAIme_ML/models/neural/weights.pt | Historical 18 rows / 5 sequence groups |
| Condition/image surrogate | models/condition_analysis/model.pkl | D:/Enzyme_ML_new/models/condition_analysis/model.pkl | Historical holdout; grouped results in separate CSV |
| Assay full-data refit | models/assay_v2/train-20261024-160717/model.pkl | D:/Enzyme_ML_new/models/assay_v2/train-20261024-160717/model.pkl | Use saved out-of-fold CSVs; refit has no independent holdout |
| GFP Ridge | runs/20261023-162906-a42ad8/baseline.pkl | D:/Enzyme_ML_new/runs/20261023-162906-a42ad8/baseline.pkl | Original 400-sequence holdout |
| GFP ESM head | runs/20261024-092703-afed29/decoder.pt | D:/Enzyme_ML_new/runs/20261024-092703-afed29/decoder.pt | Original 20-sequence execution demonstration |

The original PET CSV is D:/Final_Year_Project/EnzAIme_ML/data/processed/tpa_pet_ml_confirmed.csv. Its notebook copy is pet/data/processed/tpa_pet_ml_confirmed.csv. The avGFP dataset is Directed_Evolution-main/preprocessed_data/avGFP/avGFP.csv in both the main project and the notebook bundle.

The pretrained ESM model folders models/esm2_8m and models/esm2_35m contain actual local safetensors weights, tokenizer/configuration files and SOURCE.json with the recorded revision. The GFP task decoder is a separate artifact. New notebook models are saved under the output path printed at the beginning of each notebook.
