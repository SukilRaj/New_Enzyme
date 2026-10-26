# EnzAIme × Helix: complete notebook demonstration package

This folder is a self-contained scientific handover: one combined panel notebook and four focused notebooks, equivalent Python files, complete training functions/source, the PET and avGFP datasets, cached ESM features, saved checkpoints, offline pretrained ESM models, and saved evaluation evidence. No frontend, Flask/FastAPI server or original linked PET folder is needed for the default notebook workflows.

## Open these notebooks

**Start with `00_Complete_Panel_Demonstration.ipynb`.** It combines notebooks 01 and 02 into one file and runs both workflows in one kernel. Use the focused notebooks below when you want to demonstrate only one part. Do not run all five notebooks in succession unless you intend to repeat the combined demonstration.

1. **01_PET_Environment_Training_Evaluation.ipynb** — dataset loading, cleaning/deduplication, environmental data, sequence-group splits, original neural architecture and inference, historical test reproduction, fresh regression and neural-head training, saved assay-model calls, response surfaces, sensitivity, scenario ranking, grouped comparisons, residuals, five structure images and a 3D reference view.
2. **02_GFP_Benchmark_and_Directed_Evolution.ipynb** — full benchmark loading, original Ridge checkpoint evaluation, a shared train/validation/test split, fresh Ridge and three-head ESM Attention1D training, matched predictive evaluation, model save/reload, original ESM-head reproduction, masked-language-model substitutions, greedy/uncertainty-diversity search, plots and candidate FASTA export.
3. **03_Full_Training_and_Source_Code.ipynb** — the complete condition-study and assay-study training scripts plus the full GFP application engine, visible as executable code. Long study calls are opt-in. A source inventory points to the downloaded paper implementation and original PET preprocessing/feature-generation scripts.

4. **04_Data_Provenance_and_Feature_Preparation.ipynb** — rebuilds the 227-record provenance audit and the 221-row confirmed dataset from the supplied source records, verifies every parsed column against the packaged CSV, and regenerates ESM-2 35M per-residue features. The default encodes one protein; a flag enables all 25.

Each major operation is implemented in visible notebook cells. Model classes imported by historical pickle files are also supplied under `lab/`. The `.py` equivalents use VS Code `# %%` cells and can be inspected without Jupyter. `PANEL_GUIDE.md` maps each requested operation to its notebook/function. `executed/` holds tested notebook copies with tables and plots already visible; `verification.json` records execution results.

## Run in the installed local system

Open `D:\Enzyme_ML_new\panel_notebooks` in VS Code. Install/enable the Python and Jupyter extensions if needed. Open notebook 00 and select **Python Environments → D:\Enzyme_ML_new\.venv\Scripts\python.exe** as its kernel. Choose **Run All**, or run one cell at a time with Shift+Enter. Use a fresh kernel when opening a different focused notebook.

Alternatively, in a PowerShell terminal:

```powershell
Set-Location D:\Enzyme_ML_new\panel_notebooks
.\Start-Notebooks.cmd
```

Equivalent command from the project root:

```powershell
Set-Location D:\Enzyme_ML_new
.\.venv\Scripts\python.exe -m notebook --notebook-dir .\panel_notebooks
```

Jupyter support has been installed in the existing project environment. The original application is not required to be running. Stop Jupyter with Ctrl+C when finished.

## Run on another computer

Extract the full ZIP, preserving the `panel_notebooks` folder and its subfolders. Use Python 3.10–3.12. From inside the extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m notebook
```

On Linux/macOS, use `python3 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements-notebook.txt` and `.venv/bin/python -m notebook`. PyTorch can be CPU-only; a compatible GPU build is optional. Dependency installation needs Internet access. The supplied model weights/data allow the default workflows to run without network access after setup.

## Run on Kaggle

1. Extract the ZIP locally. Create a **private Kaggle dataset** from the extracted `panel_notebooks` folder, including data/models/source and `bundle_manifest.json`. Do not upload just the `.ipynb` files: they contain code, while the companion files contain the actual data and weights. Retain provenance and attribution; public redistribution needs the relevant upstream terms reviewed.
2. Create a Kaggle notebook and import the desired `.ipynb`. Attach your private dataset using the notebook's input-data controls. The first code cell searches `/kaggle/input` for `bundle_manifest.json`. If there are multiple copies, set `BUNDLE_OVERRIDE` to the intended `panel_notebooks` directory.
3. Use a Python runtime. GPU is optional; notebook 02 detects CUDA. The default is the packaged ESM-2 8M model; a 35M model is also included.
4. If the dependency check requests versions, enable Internet, set `INSTALL_DEPENDENCIES=True` in that cell, and run it once. After installation, restart the kernel/session, set the flag back to False, then Run All. The pinned scikit-learn/NumPy versions are required to reproduce serialized estimators. Do not silence version mismatch errors.
5. Kaggle input folders remain read-only. All new predictions, models, plots, candidate sequences and summaries are written below `/kaggle/working/enzaime_outputs/`. Download the output folder from the Kaggle session when finished.

The notebook code is tested locally. Kaggle path handling and output separation are implemented, but execution inside your Kaggle account is not claimed until you run it there. The first environment setup may need a session restart; prepare it before the panel session.

## Exactly where the models are

All paths below are relative to this `panel_notebooks` folder; the notebooks print their absolute paths at runtime. See `MODEL_AND_DATA_LOCATIONS.md` for the original project locations too.

| Artifact | Bundled path | How it is evaluated |
|---|---|---|
| Original PET Attention1D checkpoint | `pet/models/neural/weights.pt` | Notebook 01 loads its architecture and reproduces the historical 18-row test |
| Original PET normalization/split | `pet/models/neural/manifest.json` | Training mean/std and exact sequence hashes must accompany weights |
| Experimental condition/image model | `models/condition_analysis/model.pkl` | Notebook 01 loads its class and checks its historical held-out predictions |
| Assay-v2 all-data model | `models/assay_v2/train-20261024-160717/model.pkl` | Inference only; use its separately saved out-of-fold CSVs for generalization |
| Assay-v2 historical model | `models/assay_v2/train-20261024-160717/historical_model.pkl` | Trained on the historical training split; report under that split only |
| Original GFP Ridge + vectorizer | `runs/20261023-162906-a42ad8/baseline.pkl` | Notebook 02 recomputes predictions on its original 400-row test |
| Existing GFP ESM head | `runs/20261024-092703-afed29/decoder.pt` | Notebook 02 uses its matching 8M encoder and original 20-row test |
| Three-head GFP checkpoint | `runs/20261024-093528-uncertainty-diversity/decoder.pt` | Original application engine implementation/source and saved evidence included |
| Frozen ESM-2 8M / 35M | `models/esm2_8m/model.safetensors`, `models/esm2_35m/model.safetensors` | Pretrained representation/mutation models, not PET/GFP task heads |

Saved pickle files are trusted local project artifacts and are checked against the bundle manifest before loading. Keep model classes, preprocessing objects, feature metadata and data hashes together; a single weight file does not reproduce the complete pipeline.

## Datasets, environment and preprocessing

- PET source: `pet/data/processed/tpa_pet_ml_confirmed.csv` (221 rows); preprocessing reproduces 208 eligible measurements, 25 sequence groups and 22 study IDs. Temperature, pH, duration and measured TPA are present. Original source-record JSON and study provenance are retained.
- Earlier collected records and audit: `pet/master_dataset.csv`, `pet/data/tpa_pet_ml_dataset.csv`, `pet/data/processed/tpa_activity_provenance_audit.csv`, and `pet/reports/results/tpa_ambiguous_review.md`. Notebook 04 rebuilds the confirmed dataset from these inputs.
- Cached frozen ESM representations: `pet/data/neural_features/*.npz`, with `index.json` mapping protein IDs to sequence hashes. No expensive re-encoding is necessary for the PET demonstration.
- Structure evidence: `pet/data/structure_images/` contains 13 sequence-matched AlphaFold reference structures and metadata. Five visualizations derive from these coordinates; they are not newly predicted mutant folds.
- GFP: `Directed_Evolution-main/preprocessed_data/avGFP/avGFP.csv` (51,715 sequence/fitness rows). It has no per-observation environmental labels.
- Exact grouped evaluation evidence: condition-model report/splits and the assay-v2 `sequence_oof.csv`, `sequence_and_study_oof.csv`, `splits.json` and `protocol.json`.

## Recommended panel sequence

Start with notebook 00: show the actual files, dataset rows and environmental columns; show cleaning and group separation; load the original checkpoint and reproduce its test; train a fresh model; change the input conditions in `predict_pet`; then display the surfaces, residuals and structure images. Continue to the GFP section for pretrained ESM loading, Attention1D training and same-split comparison. Use notebook 03 for the full research training code and opt-in experiment controls, and notebook 04 for original data provenance and feature generation.

Default budgets are modest so code can run during a demonstration. The full studies are opt-in, and every new run has its own timestamped output directory. Run everything once on the panel machine beforehand; do not rely on first-time package installation during the demonstration.

## Scientific status

The notebook package makes the implementation runnable and inspectable. It does not change existing weak PET evaluation, establish image benefit, supply missing independent-oracle assets, or turn the project into a fully matched paper reproduction. Show measured benchmark predictions separately from scores of generated sequences. Notebook 02's new matched demonstration is a small single-seed experiment, not a publication-scale comparison. Temperature/pH/time are PET features; the GFP search remains sequence-based. Consult `documentation/BENCHMARK_COMPARISON_AND_GAPS.md` for supported comparison claims.

The copied upstream directed-evolution source retains its `LICENSE` and README; its decoder attributes adopted components. `SOURCE.json` beside each ESM model records model ID and revision. Dataset/structure source provenance remains in the shipped manifests and records.
