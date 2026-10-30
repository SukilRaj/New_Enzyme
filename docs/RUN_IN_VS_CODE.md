# Run EnzAIme and Helix in VS Code

This is the complete launch guide for the installed local integration. Both project folders are required: `D:\Enzyme_ML_new` and `D:\Final_Year_Project\EnzAIme_ML`. The shared interface runs at http://127.0.0.1:8765/.

## Start the project

1. Open VS Code and choose **File → Open Folder → D:\Enzyme_ML_new**. Alternatively, open `D:\Enzyme_ML_new\EnzAIme-Helix.code-workspace` to see both projects in one VS Code window.
2. Open **Terminal → New Terminal** and select PowerShell.
3. Run:

```powershell
Set-Location D:\Enzyme_ML_new
.\Check-Research.cmd
.\Start-Research.cmd
```

Wait for **EnzAIme × Helix is ready** and open http://127.0.0.1:8765/. Keep the terminal open. Stop the application with **Ctrl+C** in that terminal.

The launcher uses the installed virtual environments directly. You do not need to activate an environment, start npm, or run Flask and FastAPI separately. The PET frontend has already been built. `Start-Helix.cmd` starts only the Helix service; use `Start-Research.cmd` for the integrated application.

The equivalent direct command is:

```powershell
.\.venv\Scripts\python.exe .\launch_research.py
```

For the VS Code menu workflow, choose **Terminal → Run Task → Start integrated EnzAIme + Helix**. Additional tasks check the installation and train each PET study.

## What to demonstrate

| Workspace section | Suggested action | Result to inspect |
|---|---|---|
| Overview | Confirm both engines are connected | Live service status and project evidence |
| Enzyme discovery | Set 40 °C, pH 8 and 48 hours; request catalog ranking | Predicted TPA and recorded model evaluation |
| Environmental analysis | Choose a protein; use spans of 5 °C, 0.5 pH and 12 hours | 27 scenarios, 49-point response surface, three sensitivity curves |
| Protein images | Choose a protein with an available structure and generate views | Front, side, top, distance map, proximity map and ZIP |
| Sequence and structure | Import a variant FASTA or preview substitutions, then inspect edits on the reference | Residue mapping, rotatable coordinates, coverage and reference PDB download |
| Directed evolution | Run a CPU baseline with a modest sample budget | Training progress, held-out errors, search history and candidates |
| Training laboratory | Inspect the completed assay study | 26-configuration protocol, grouped errors and downloadable evidence |
| Model comparison | Read the comparison tables with their cohort labels | Separate fixed-test and grouped results |
| Experiment registry | Reopen a run and export its record | Configuration, units, hashes and saved outputs |

PET condition effects are model predictions. The images render cached structures. GFP search candidates are hypotheses until independently evaluated. Keep these distinctions in the demonstration and paper.

## FASTA and structure inspection

FASTA contains amino-acid letters; PDB contains atom coordinates. Use **Sequence & structure → Import a protein FASTA** to reload a downloaded variant. Select the same reference protein first. Then choose **Inspect edits on reference structure** and use the mutation buttons to highlight individual positions. **Download reference PDB** exports coordinates that can be loaded back into the PDB field. These are reference coordinates, not a predicted mutant fold. See [the illustrated file-format guide](SEQUENCE_AND_STRUCTURE_GUIDE.md).

## Editing the integrated frontend

The maintained React source is `D:\Enzyme_ML_new\frontend\src`; the launcher serves `frontend\dist`. Its initial source came from the linked PET frontend. To rebuild after editing:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix frontend run lint
node --test frontend/tests/sequenceFiles.test.js
```

Refresh the integrated browser with Ctrl+F5. Use port 8765 for the complete application; the standalone Vite development server does not include the integration's request-token bootstrap. The linked PET backend/data/models are still required.

## Retraining

Run training in a **second terminal** so the interface remains available. These commands rerun bounded studies and consume CPU time; running them is unnecessary just to launch the application.

```powershell
Set-Location D:\Enzyme_ML_new
.\.venv\Scripts\python.exe .\scripts\train_assay_models.py
```

This creates a new folder under `models/assay_v2/` and updates `latest.json`. The Training laboratory reads the latest run. It does not automatically replace the prediction service checkpoint.

```powershell
.\.venv\Scripts\python.exe .\scripts\train_condition_models.py
```

This rebuilds the earlier condition/image study and overwrites its study artifacts under `models/condition_analysis/`. Preserve a copy of those artifacts before rerunning if a paper cites them. GFP training is started through **Directed evolution**, with a separate model and saved run for each experiment.

## Verification commands

The first command checks installed files and packages without starting services. The live check requires `Start-Research.cmd` to be running.

```powershell
.\Check-Research.cmd
.\Check-Research.cmd --live
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\verify_integration.py
.\.venv\Scripts\python.exe .\scripts\verify_analysis.py
```

The two `verify_*.py` scripts perform real local predictions and create traceable demonstration records. Software checks do not establish model accuracy or biological efficacy.

## Common startup problems

| Message or symptom | Action |
|---|---|
| Port 8765 or 8766 occupied | Stop the previous project terminal with Ctrl+C, then relaunch. Do not run two copies. |
| Connection unavailable or failed to fetch | Confirm the launcher terminal is still open and both services reached readiness. Use the integrated launcher. |
| Old interface after an update | Stop and restart the launcher, then refresh the browser with Ctrl+F5. |
| Missing PET folder or interpreter | Restore the linked project or set the two environment variables below. |
| Missing frontend build | Run `npm --prefix frontend ci` and `npm --prefix frontend run build` from Enzyme_ML_new. |
| ESM weights unavailable | Use the cached model supported by the installed environment. A first-time model download needs network access. |
| LLM explanation or stability feature unavailable | These require their own compatible local model artifacts. Other workflows remain usable. |

For relocated folders:

```powershell
$env:ENZAIME_ROOT = 'D:\path\to\EnzAIme_ML'
$env:ENZAIME_PYTHON = 'D:\path\to\EnzAIme_ML\.venv-1\Scripts\python.exe'
.\Check-Research.cmd
.\Start-Research.cmd
```

The setup scripts in this repository rebuild the Helix environment. They do not reconstruct the entire PET project, its feature cache or trained artifacts. Copying only `Enzyme_ML_new` to another computer is insufficient. Follow the PET README when recreating its environment, preserve its data/model/structure folders, and rebuild virtual environments instead of treating them as portable binaries.

## Files to preserve for the project and paper

- Application and upstream source: `lab/`, `frontend/` (source, package lock and build), `scripts/`, `tests/`, `Directed_Evolution-main/`, launch/setup files and requirements.
- Models and data pointers: `models/`, the complete linked PET project and its data/model manifests.
- Research outputs: `runs/`, `research_runs/`, `analysis_runs/`, `generated_structures/`, `reports/`.
- Paper package: `docs/journal/`, `docs/ANALYSIS_METHODS_AND_RESULTS.md`, `docs/PROJECT_COMPARISON_AND_INTEGRATION.md` and `REPLICATION_STATUS.md`.

The base paper PDF is a separately supplied reference, not part of the software release. Preserve upstream attribution and applicable licenses when sharing code or data.
