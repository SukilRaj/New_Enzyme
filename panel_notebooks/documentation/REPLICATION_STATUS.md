# Replication status

## Source evidence

Base paper: Tran & Hy, IEEE TEVC 29(2), 418–428 (2025), DOI 10.1109/TEVC.2024.3439690. Read from the user-provided local PDF. Source archive: `Directed_Evolution-main.zip`.

Section V-B (printed page 422) specifies:

| Component | Paper | Local initial adaptation |
|---|---|---|
| Mutation model | ESM-2, 35M | ESM-2, 35M or 8M test option |
| Optimization encoder | 33-layer ESM-2 | Same smaller encoder selected for local run |
| Evaluation oracle | Separate trained oracle from reference [24] | Missing from archive; not substituted |
| Population | 128 | Configurable 2–64 |
| Proposals per member | 4 | 2 |
| Random masking fraction | 0.6 | 0.5 |
| k-mer size | 1 | 1 |
| Repeats | 5, mean and standard deviation | One seed per saved run |
| Starting point | Wild type | Median-fitness sequence from training subset |
| Benchmarks | Eight | avGFP local workflow |

Sections V-C/D also require external baseline implementations and independent-oracle measures (maximum/average fitness, diversity, novelty, and distance to wild type). The ZIP does not bundle those baseline implementations. Local RMSE/MAE/Spearman on measured held-out labels assess predictor performance; they are not substitutes for the paper's design benchmark metrics.

## Completion gates

1. Local application starts and baseline runs end to end.
2. Real ESM weights download and a small local training/search run succeeds.
3. Resolve missing independent oracle artifacts and supplementary training settings.
4. Implement and validate a matched avGFP paper configuration.
5. Run five seeds and common-oracle comparisons, then assess the remaining benchmark scope.
6. Select a novelty hypothesis after literature review; add ablations and independent evaluation.

The UI and local adaptation address gate 1 and provide the code path for gate 2. Later gates are not claimed complete.

## Independent-oracle source located

The PEX authors' repository is https://github.com/HeliXonProtein/proximal-exploration. Its `download_landscape.sh` references a Google Drive archive (file ID `1uy9zgtJ60Z83LCbm7Z_AoksAkmK4CcsC`). The archive's availability and contents have not yet been verified locally. Review the downloaded artifacts and licensing before using them, and record hashes. Do not run the old shell downloader unchanged on Windows; it also disables certificate checking. This source discovery does not mean the oracle has been installed.
