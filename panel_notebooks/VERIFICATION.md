# Local notebook execution verification

Executed in fresh local Jupyter kernels on 2026-10-28.
The combined panel demonstration uses modest training budgets. The full PET study flags were also enabled during notebook 03 verification.

| Notebook | Code cells completed | Plots | Seconds |
|---|---:|---:|---:|
| 00_Complete_Panel_Demonstration.ipynb | 31/31 | 13 | 77.04 |
| 01_PET_Environment_Training_Evaluation.ipynb | 19/19 | 9 | 56.13 |
| 02_GFP_Benchmark_and_Directed_Evolution.ipynb | 12/12 | 4 | 78.04 |
| 03_Full_Training_and_Source_Code.ipynb | 8/8 | 0 | 368.23 |
| 04_Data_Provenance_and_Feature_Preparation.ipynb | 8/8 | 1 | 13.96 |

The clean root notebook 03 keeps long-study flags off for a quick walkthrough. Its executed copy records the completed full run with flags enabled.
Kaggle input discovery and writable output paths are implemented; these results were produced locally, not in a Kaggle account.
Execution success verifies that the code runs; it does not establish high predictive performance or reproduction of the base paper.

## Included completed output folders

- `outputs/pet-20261026-112832-f3dd13`
- `outputs/gfp-20261026-113037-6e0f85`
- `outputs/full_studies-20261026-113202-268a1b`
- `outputs/data_preparation-20261026-100222-04512b`
