"""Checkpoint-backed research recommendation; missing artifacts fail explicitly."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from backend.app.services.neural_oracle import AttentionOracle, batch_features

ROOT = Path(__file__).resolve().parents[3]


class NeuralRecommender:
    def __init__(self, model_dir=None, features_dir=None, dataset=None, stability_dir=None):
        model_dir = Path(model_dir or ROOT / "models/neural")
        self.features_dir = Path(features_dir or ROOT / "data/neural_features")
        dataset = Path(dataset or ROOT / "data/processed/tpa_pet_ml_confirmed.csv")
        for path in (model_dir / "manifest.json", model_dir / "weights.pt", self.features_dir / "index.json"):
            if not path.is_file():
                raise FileNotFoundError("Neural training artifacts are missing. Follow docs/multimodal_training.md; no Ridge fallback is used.")
        self.meta = json.loads((model_dir / "manifest.json").read_text())
        self.index = json.loads((self.features_dir / "index.json").read_text())
        evaluation_path = model_dir / "held_out_predictions.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation:
                observed = np.array([r["observed_um"] for r in evaluation], dtype=np.float64)
                predicted = np.array([r["predicted_um"] for r in evaluation], dtype=np.float64)
                if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
                    raise ValueError("Nonfinite held-out evaluation values")
                variance = float(np.sum((observed-observed.mean())**2))
                self.meta["test_metrics"].update({
                    "r2": float(1-np.sum((observed-predicted)**2)/variance) if variance > 0 else None,
                    "test_rows": len(evaluation),
                    "test_proteins": len({r["protein_id"] for r in evaluation}),
                })
        if self.meta["schema_version"] != 1 or self.meta["target"] != "log1p_tpa_um":
            raise ValueError("Unsupported neural checkpoint contract")
        if any(self.meta[k] != self.index[k] for k in ("encoder", "image_encoder")):
            raise ValueError("Encoder provenance does not match the checkpoint")
        if self.meta["dataset_sha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
            raise ValueError("Dataset changed since training; re-export matching artifacts")
        self.model = AttentionOracle(**self.meta["architecture"])
        self.model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.stability = None
        stability_dir = Path(stability_dir or ROOT / 'models/stability')
        if (stability_dir / 'manifest.json').is_file() and (stability_dir / 'weights.pt').is_file():
            from backend.app.services.stability_oracle import StabilityOracle
            self.stability = StabilityOracle(stability_dir)
            if self.stability.meta['encoder'] != self.meta['encoder']:
                raise ValueError('Stability and activity encoders are incompatible')
        self.mean = np.array(self.meta["environment_mean"], dtype=np.float32)
        self.std = np.array(self.meta["environment_std"], dtype=np.float32)
        if not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid checkpoint normalization")
        csv.field_size_limit(10_000_000)
        with dataset.open(encoding="utf-8-sig", newline="") as f:
            self.catalog = {r["protein_id"]: r for r in csv.DictReader(f)}

    def recommend(self, temperature, ph, duration, top_n=5, objective='activity'):
        env = np.array([temperature, ph, duration], dtype=np.float32)
        if not np.isfinite(env).all() or not 0 <= ph <= 14 or duration <= 0:
            raise ValueError("Invalid assay conditions")
        if objective not in ('activity', 'joint'):
            raise ValueError('Unknown objective')
        if objective == 'joint' and self.stability is None:
            raise FileNotFoundError('Joint ranking requires a trained stability checkpoint. Complete the v2 Kaggle notebook and import its results.')
        outside = bool(np.any(env < self.meta["environment_min"]) or np.any(env > self.meta["environment_max"]))
        results = []
        for pid, row in self.catalog.items():
            meta = self.index["proteins"][pid]
            if hashlib.sha256(row["sequence"].strip().upper().encode()).hexdigest() != meta["sequence_sha256"]:
                raise ValueError("Protein sequence and features disagree")
            path = (self.features_dir / meta["file"]).resolve()
            if not path.is_relative_to(self.features_dir.resolve()):
                raise ValueError("Feature path is outside the feature directory")
            with np.load(path, allow_pickle=False) as f:
                item = {"sequence": f["sequence"].astype(np.float32), "image": f["image"].astype(np.float32)}
            if not all(np.isfinite(x).all() for x in item.values()):
                raise ValueError("Nonfinite feature values")
            seq, mask, image = batch_features([item])
            ip = torch.tensor([float(meta["image_present"] and self.meta["image_trained"])])
            conditions = torch.tensor(((env-self.mean)/self.std)[None])
            with torch.inference_mode():
                score, attention = self.model(seq, mask, image, ip, conditions)
                no_image, _ = self.model(seq, mask, image, torch.zeros_like(ip), conditions)
            value = float(np.expm1(float(score[0])))
            if not np.isfinite(value):
                raise ValueError("Model produced a nonfinite prediction")
            positions = torch.topk(attention[0], min(5, len(item["sequence"]))).indices.tolist()
            results.append({"protein_id": pid, "protein_name": row["protein_name"],
                "predicted_activity_um": max(0., value), "predicted_tm_c": self.stability.predict(item, ph) if self.stability else None,
                "image_used": bool(ip.item()), "outside_training_conditions": outside,
                "explanation": {"attention_positions": [i+1 for i in positions],
                    "image_ablation_delta_log1p": float(score[0]-no_image[0]),
                    "note": "Attention indicates model focus, not causal residue effects. Image ablation compares the same model with its image input disabled."}})
        results.sort(key=lambda r: r["predicted_activity_um"], reverse=True)
        if objective == 'joint':
            from backend.app.services.directed_evolution import pareto_front
            front = pareto_front([{'activity': r['predicted_activity_um'], 'tm': r['predicted_tm_c'], 'protein_id': r['protein_id']} for r in results])
            ids = {r['protein_id'] for r in front}
            results = [{**r, 'pareto_optimal': True} for r in results if r['protein_id'] in ids]
        return {"model": "ESM-2 Attention1D with optional image fusion", "status": self.meta["status"],
            "objective": objective, "ranking_note": "Pareto front, displayed by activity; no single joint optimum is implied." if objective == 'joint' else "Descending predicted TPA concentration.",
            "conditions": {"temperature": temperature, "ph": ph, "duration": duration},
            "objectives_available": ["tpa_activity"] + (["thermostability"] if self.stability else []), "objectives_pending": [] if self.stability else ["thermostability"],
            "recommendations": [{"rank": i+1, **r} for i, r in enumerate(results[:top_n])],
            "test_metrics": self.meta["test_metrics"], "limitations": self.meta["limitations"]}

    def attribution(self, protein_id, temperature, ph, duration):
        if protein_id not in self.catalog:
            raise ValueError('Protein is not in the model catalog')
        env = np.array([temperature, ph, duration], dtype=np.float32)
        if not np.isfinite(env).all() or not 0 <= ph <= 14 or duration <= 0:
            raise ValueError('Invalid assay conditions')
        meta = self.index['proteins'][protein_id]
        path = (self.features_dir / meta['file']).resolve()
        if not path.is_relative_to(self.features_dir.resolve()):
            raise ValueError('Invalid feature path')
        with np.load(path, allow_pickle=False) as f:
            item = {'sequence':f['sequence'].astype(np.float32), 'image':f['image'].astype(np.float32)}
        sequence, mask, image = batch_features([item])
        sequence.requires_grad_(True)
        present = torch.tensor([float(meta['image_present'] and self.meta['image_trained'])])
        environment = torch.tensor(((env-self.mean)/self.std)[None], requires_grad=True)
        score, _ = self.model(sequence, mask, image, present, environment)
        residue_grad, condition_grad = torch.autograd.grad(score[0], (sequence, environment))
        raw = (sequence.detach()*residue_grad).sum(-1)[0].detach().numpy()
        normalized = raw / max(float(np.abs(raw).sum()), 1e-12)
        positions = np.argsort(-np.abs(normalized))[:10]
        reference = self.catalog[protein_id]['sequence']
        return {'protein_id':protein_id, 'method':'Gradient × input on frozen ESM residue representations',
            'target':'log(1 + predicted TPA)',
            'residues':[{'position':int(i)+1,'residue':reference[i],'attribution':float(normalized[i])} for i in positions],
            'condition_gradients':dict(zip(['temperature_c','ph','duration_hours'], (condition_grad[0].detach().numpy()/self.std).astype(float).tolist())),
            'note':'Local model sensitivity, not mutation effects or a causal mechanism. Signed residue contributions are normalized by total absolute contribution; zero embeddings are an artificial reference.'}


def explain_with_ollama(evidence):
    from backend.app.services.local_llm import explain
    return explain(evidence)
