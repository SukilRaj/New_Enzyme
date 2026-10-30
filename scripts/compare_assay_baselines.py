"""Prespecified baseline comparison on the frozen assay_v2 outer folds.

Read-only with respect to training: it reuses the split definitions saved by the
latest `scripts/train_assay_models.py` run and evaluates the requested baseline
family on those identical outer folds. No model is (re)selected against these
scores and nothing here is promoted to serving.

Baselines / candidates (all prespecified, none tuned on the outer folds):
- training_mean            constant = mean target of the outer training fold
- training_median          constant = median target of the outer training fold
- global_mean              constant = mean target of all 208 observations
- global_median            constant = median target of all 208 observations
- sequence_environment_huber  HuberRegressor (epsilon=1.35, alpha=1.0) on the
                           same sequence+environment features, log1p target
- sequence_environment_ridge  Ridge(alpha=1000), fixed, as a reference point

Writes baseline_comparison.json/.csv next to the run artifacts.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from assay_model import dataset, AssayRegressor  # noqa: E402


def scores(y, pred):
    err = np.abs(y - pred)
    signed = pred - y
    mse = float(np.mean((y - pred) ** 2))
    return {
        "mae_um": float(np.mean(err)),
        "mse_um2": mse,
        "rmse_um": float(np.sqrt(mse)),
        "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
        "log_rmse": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(pred)) ** 2))),
        "spearman": float(spearmanr(y, pred).statistic) if np.ptp(pred) else None,
        "median_absolute_error_um": float(np.median(err)),
        "p90_absolute_error_um": float(np.quantile(err, 0.9)),
        "signed_bias_um": float(np.mean(signed)),
        "rows": int(len(y)),
    }


def transform_fit(data, train):
    """Fit the exact AssayRegressor sequence+environment pipeline on `train`."""
    return AssayRegressor(mode="sequence_environment", kind="ridge", strength=1000, seed=42).fit(
        *(data[k][train] for k in ("sequence", "env", "context", "target", "groups"))
    )


def features(model, data, index):
    return model.scaler.transform(model.imputer.transform(
        model.features(data["sequence"][index], data["env"][index], data["context"][index])))


def run_estimator(make, data, train, test, transform="log1p"):
    model = transform_fit(data, train)
    x_train = features(model, data, train)
    y_train = data["target"][train]
    target = np.log1p(y_train) if transform == "log1p" else y_train
    est = make().fit(x_train, target)
    raw = est.predict(features(model, data, test))
    if transform == "log1p":
        return np.expm1(np.clip(raw, 0, 20))
    return np.maximum(raw, 0)


def main():
    data = dataset()
    run_id = json.loads((ROOT / "models/assay_v2/latest.json").read_text())["run_id"]
    folder = ROOT / "models/assay_v2" / run_id
    splits = json.loads((folder / "splits.json").read_text())
    y_all = data["target"]
    results = {}
    for design, folds in splits.items():
        predictions = {
            "training_mean": np.full(len(y_all), np.nan),
            "training_median": np.full(len(y_all), np.nan),
            "sequence_environment_huber": np.full(len(y_all), np.nan),
            "sequence_environment_ridge": np.full(len(y_all), np.nan),
        }
        for fold in folds:
            train = np.array(fold["train"])
            test = np.array(fold["test"])
            predictions["training_mean"][test] = float(np.mean(y_all[train]))
            predictions["training_median"][test] = float(np.median(y_all[train]))
            predictions["sequence_environment_huber"][test] = run_estimator(
                lambda: HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500), data, train, test)
            predictions["sequence_environment_ridge"][test] = run_estimator(
                lambda: Ridge(alpha=1000), data, train, test)
        # constant references computed once on the whole cohort
        predictions["global_mean"] = np.full(len(y_all), float(np.mean(y_all)))
        predictions["global_median"] = np.full(len(y_all), float(np.median(y_all)))
        design_scores = {name: scores(y_all, p) for name, p in predictions.items()}
        results[design] = design_scores
        with (folder / f"{design}_baseline_oof.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_row", "observed_um", *predictions])
            for i in range(len(y_all)):
                writer.writerow([i, y_all[i], *[p[i] for p in predictions.values()]])
    out = {
        "run_id": run_id,
        "note": ("Prespecified baseline comparison recomputed on the frozen outer folds. "
                 "Constants are fold-specific (mean/median of the outer training fold); global "
                 "constants use all 208 rows as an absolute reference floor. Huber and Ridge use "
                 "fixed hyperparameters and are not selected on these scores."),
        "designs": results,
    }
    (folder / "baseline_comparison.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    rows = [{"design": d, "model": m, **s} for d, modes in results.items() for m, s in modes.items()]
    with (folder / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
