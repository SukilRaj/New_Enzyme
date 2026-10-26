"""Train a research oracle on Kaggle; preserve held-out groups and provenance."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from backend.app.services.neural_oracle import AttentionOracle, batch_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/tpa_pet_ml_confirmed.csv")
    p.add_argument("--features", default="data/neural_features")
    p.add_argument("--output", default="models/neural")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sequence-only", action="store_true", help="Disable images for the ablation")
    args = p.parse_args()
    if args.epochs < 1:
        p.error("epochs must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    csv.field_size_limit(10_000_000)
    raw = Path(args.dataset).read_bytes()
    with open(args.dataset, encoding="utf-8-sig", newline="") as f:
        source = list(csv.DictReader(f))
    rows, seen = [], set()
    for row in source:
        values = [float(row[k]) for k in ("temperature_c", "ph", "duration_hours", "target_value_standardized")]
        if not np.isfinite(values).all() or values[2] <= 0 or values[3] < 0:
            continue
        seq = row["sequence"].strip().upper()
        key = (seq, *values, row.get("literature_doi", ""), row.get("assay_index", ""), row.get("measurement_index", ""))
        if key not in seen:
            rows.append(row)
            seen.add(key)
    index = json.loads((Path(args.features) / "index.json").read_text())
    features = []
    groups = []
    for row in rows:
        meta = index["proteins"][row["protein_id"]]
        digest = hashlib.sha256(row["sequence"].strip().upper().encode()).hexdigest()
        if digest != meta["sequence_sha256"]:
            raise ValueError("Feature/sequence mismatch")
        groups.append(digest)
        with np.load(Path(args.features) / meta["file"], allow_pickle=False) as f:
            features.append({"sequence": f["sequence"].astype(np.float32), "image": f["image"].astype(np.float32)})
    if len(set(groups)) < 5:
        raise ValueError("At least five unique sequence groups are required")
    groups = np.array(groups)
    remaining, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(rows, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed+1).split(remaining, groups=groups[remaining]))
    train, val = remaining[a], remaining[b]
    env = np.array([[float(r[k]) for k in ("temperature_c", "ph", "duration_hours")] for r in rows], dtype=np.float32)
    target = np.log1p(np.array([float(r["target_value_standardized"]) for r in rows], dtype=np.float32))
    mean, std = env[train].mean(0), env[train].std(0).clip(1e-6)
    present = np.array([float(index["proteins"][r["protein_id"]]["image_present"] and not args.sequence_only) for r in rows], dtype=np.float32)
    if present.any() and not present[train].any():
        raise ValueError("Image branch has no training images; use --sequence-only or add paired training images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"sequence_dim": features[0]["sequence"].shape[-1], "image_dim": 512, "hidden": 64}
    model = AttentionOracle(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def predict(ids, training=False):
        x, mask, im = batch_features([features[i] for i in ids], device)
        ip = torch.as_tensor(present[ids], device=device)
        if training:
            ip = ip * (torch.rand_like(ip) > 0.2)  # learn explicit missing-image behavior
        e = torch.as_tensor((env[ids]-mean)/std, device=device)
        return model(x, mask, im, ip, e)[0]

    best, state, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        for ids in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train)/8)))):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predict(ids, True), torch.as_tensor(target[ids], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            error = float(torch.mean((predict(val)-torch.as_tensor(target[val], device=device))**2))
        if error < best:
            best, state, wait = error, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= 20:
            break
    if state is None:
        raise ValueError("Training failed to produce finite validation loss")
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        pred_log = predict(test).cpu().numpy()
    pred = np.maximum(0, np.expm1(pred_log))
    observed = np.expm1(target[test])
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite held-out predictions; checkpoint will not be exported")
    baseline = np.expm1(np.median(target[train]))
    metrics = {"mae_um": float(np.abs(pred-observed).mean()),
               "r2": float(1-np.sum((pred-observed)**2)/np.sum((observed-observed.mean())**2)) if np.var(observed)>0 else None,
               "rmse_um": float(np.sqrt(np.mean((pred-observed)**2))),
               "training_median_baseline_mae_um": float(np.abs(observed-baseline).mean()),
               "log_rmse": float(np.sqrt(np.mean((pred_log-target[test])**2)))}
    manifest = {"schema_version": 1, "status": "research_checkpoint_not_validated_for_mutation_effects",
        "architecture": config, "encoder": index["encoder"], "image_encoder": index["image_encoder"],
        "image_trained": bool(present[train].any()), "target": "log1p_tpa_um",
        "environment_order": ["temperature_c", "ph", "duration_hours"],
        "environment_mean": mean.tolist(), "environment_std": std.tolist(),
        "environment_min": env[train].min(0).tolist(), "environment_max": env[train].max(0).tolist(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed,
        "rows_used": len(rows), "rows_excluded_or_deduplicated": len(source)-len(rows),
        "split_sequence_hashes": {name: sorted(set(groups[ids])) for name, ids in [("train", train), ("validation", val), ("test", test)]},
        "test_metrics": metrics,
        "validation_metrics": {"log_mse": best},
        "limitations": ["Exact-sequence groups; homologous sequences and papers may cross splits.",
                        "Small heterogeneous assay dataset; no causal or wet-lab validation.",
                        "Image branch requires sequence-only and shuffled-image ablations."]}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in state.items()}, out / "weights.pt")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "held_out_predictions.json").write_text(json.dumps([
        {"protein_id": rows[i]["protein_id"], "observed_um": float(o), "predicted_um": float(v),
         "temperature_c": float(rows[i]['temperature_c']), "ph": float(rows[i]['ph']),
         "duration_hours": float(rows[i]['duration_hours']), "publication_doi": rows[i].get('literature_doi'),
         "condition_group_id": rows[i].get('condition_group_id')}
        for i, o, v in zip(test, observed, pred)], indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
