# Panel demonstration guide

## One notebook to start

Open `00_Complete_Panel_Demonstration.ipynb` in VS Code, select the installed Python environment and choose Run All. Its PET and GFP sections are the full contents of notebooks 01 and 02. It creates two separate timestamped output directories and prints their paths. The default demonstration uses small training budgets; the complete source and opt-in longer studies are also supplied.

The notebook first loads existing checkpoints to reproduce recorded results, then trains new demonstration checkpoints. Re-running it creates another output directory. It does not overwrite the original models.

## Where each requested operation is implemented

| Requested operation | Notebook and visible code |
|---|---|
| Raw data provenance, analyte/unit checks, ambiguous exclusions | 04, complete `build_tpa_provenance_audit.py` and `build_confirmed_tpa_dataset.py` source cells |
| PET dataset loading, duplicates, eligibility, environmental variables | 00/01, `load_data`, `environmental_features`, `ConditionRegressor` |
| Assay context extraction, missingness, encoding/scaling | 00/01, complete `AssayRegressor` implementation and `dataset` |
| ESM features from amino-acid sequences | 04, `encode_pet_sequence`; 00/02, `encode_sequences` |
| Original PET checkpoint model class and loading | 00/01, `AttentionOracle`, `batch_features`, `load_pet_neural_model` |
| Manual prediction on catalog PET proteins | 00/01, `predict_pet` and `call_pet_neural` |
| Fresh PET regression training and validation selection | 00/01, `train_condition_model` |
| Fresh PET neural-head training and learning curves | 00/01, `train_pet_attention` |
| Full condition/assay grouped experiments | 03, complete study scripts; enable their two run flags |
| GFP dataset preprocessing | 00/02, `load_gfp_dataset` and a saved shared three-way split |
| GFP baseline training | 00/02, `position_features`, `train_gfp_ridge` |
| GFP ESM architecture, training, saving and reloading | 00/02, `Decoder`, `train_gfp_attention`, `load_gfp_heads` |
| GFP inference | 00/02, `predict_gfp`, `predict_head_features` |
| Evaluation | 00/01/02, `evaluate_predictions` for MAE, RMSE, R² and Spearman; test predictions/metrics exported |
| Environmental response and sensitivity plots | 00/01, temperature/pH surface, three condition sweeps, scenario ranking |
| Protein and residue visualization | 00/01, `coordinates`, `distance_image`, front/side/top, distance/contact images, 3D reference trace and residue neighbors |
| Directed evolution | 00/02, `propose_mutation`, `evolve_sequences`; original application engine in 03 |
| Original base-paper source | `Directed_Evolution-main/`, with upstream README and LICENSE |

## Manual model calls during the demonstration

After the PET model-loading and function-definition cells have run:

```python
predict_pet('1', temperature_c=40, ph=8, duration_hours=48, engine='original')
predict_pet('1', temperature_c=45, ph=8.5, duration_hours=72, engine='condition')
predict_pet('1', temperature_c=40, ph=8, duration_hours=48, engine='assay')
```

The result is predicted TPA concentration in micromolar for the chosen catalog protein. Supported engines load actual packaged checkpoints. Catalog IDs are printed in the notebook. A mutated sequence cannot borrow the original protein's cached embedding as if it were unchanged.

After the GFP training/function cells have run:

```python
sequence = sample.sequence.iloc[0]  # or another validated equal-length GFP sequence
predicted_fitness, ensemble_sd = predict_gfp([sequence])
print(predicted_fitness, ensemble_sd)
```

This calls the newly trained demonstration ensemble. `ensemble_sd` is disagreement between heads, not a calibrated confidence interval. Existing checkpoint evaluations appear in separate cells with their original test data.

After notebook 04 has loaded its local ESM model:

```python
embedding, sequence_sha256 = encode_pet_sequence(catalog_rows.loc['1', 'sequence'])
print(embedding.shape)  # number of residues × 480
```

## Change the training budget manually

- PET demonstration: set `DEMO_EPOCHS` before the neural-training cell; default 12. The selected epoch comes from validation loss.
- GFP demonstration: set `SAMPLES`, `HEAD_EPOCHS`, `ENSEMBLE_SIZE`, `SEED` and `ESM_FOLDER` in the dataset configuration cell; defaults 200, 18, 3, 42 and `esm2_8m`. Both predictors share the same 60/20/20 split. Use a fresh kernel and rerun from the beginning after changing these settings.
- Complete PET studies: in clean notebook 03 set `RUN_FULL_CONDITION_STUDY=True` and/or `RUN_FULL_ASSAY_STUDY=True`. These run all specified nested folds/configurations and write new results below the notebook's output directory. They take longer than the panel demonstration.
- Original application GFP engine: notebook 03, `RUN_APPLICATION_GFP=True`; choose `APPLICATION_ENGINE='baseline'` or `'esm'`, then edit its configuration. The initial 100-sequence/one-epoch setting checks execution only.
- All PET features: notebook 04, `REBUILD_ALL_PROTEINS=True` before the feature-generation cell.

More training does not guarantee better unseen-protein performance. Specify a study before looking at its test results. These are regression tasks, so an accuracy percentage is not the reported metric.

## Explain these results accurately

1. PET temperature, pH and duration are measured dataset columns used by the PET models. The GFP benchmark has sequence/fitness only; no environmental labels have been added to it.
2. Original checkpoint reproduction, new small demonstration fits and nested grouped studies are separate evaluations. Their different sample sizes/splits should remain visible.
3. The assay-v2 serving checkpoint was fitted to all 208 eligible records. Its saved outer-fold predictions, rather than its training-set fit, support generalization reporting.
4. The five protein images and 3D trace render existing reference coordinates. FASTA contains sequence characters; PDB contains coordinates. Exporting a mutant FASTA does not compute its structure.
5. Candidate search scores are the search model's predictions. The base paper's independent oracle is not included, so these are not a matched reproduction of the paper's design scores.

## Files to take to the panel

Take the entire extracted `panel_notebooks` folder, including `bundle_manifest.json`, datasets, `lab`, `models`, `pet`, `runs` and the upstream source. The notebooks alone are insufficient. Keep `executed/` for viewing previously checked results immediately, and use the clean root notebooks for live execution. An executed notebook may have optional study flags enabled from verification; inspect those flags before rerunning it.

Start the Python environment and run the installation/version checks before the presentation. No web server is needed. `README.md` contains Windows, other-machine and Kaggle instructions; `MODEL_AND_DATA_LOCATIONS.md` identifies the saved models.
