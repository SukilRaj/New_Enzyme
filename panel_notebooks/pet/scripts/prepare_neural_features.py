"""Run from project root: python -m scripts.prepare_neural_features [--images manifest.json]."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "facebook/esm2_t12_35M_UR50D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tpa_pet_ml_confirmed.csv")
    parser.add_argument("--output", default="data/neural_features")
    parser.add_argument("--images", help="JSON mapping protein_id to a list of image paths; relative to manifest")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    with open(args.dataset, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    proteins = {}
    for row in rows:
        seq = row["sequence"].strip().upper()
        if not seq or len(seq) > 1022 or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Invalid or too-long sequence: {row['protein_id']}")
        if row["protein_id"] in proteins and proteins[row["protein_id"]] != seq:
            raise ValueError("Conflicting sequences for one protein ID")
        proteins[row["protein_id"]] = seq
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL).to(device).eval()
    images = json.loads(Path(args.images).read_text()) if args.images else {}
    if images:
        from PIL import Image
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        vision = resnet18(weights=weights)
        vision.fc = torch.nn.Identity()
        vision = vision.to(device).eval()
        transform = weights.transforms()
    index = {}
    for pid, seq in proteins.items():
        tokens = tokenizer(seq, return_tensors="pt").to(device)
        with torch.inference_mode():
            states = encoder(**tokens).last_hidden_state[0, 1:len(seq)+1].cpu().numpy()
        visual = np.zeros(512, dtype=np.float32)
        paths = images.get(pid, [])
        if not isinstance(paths, list):
            raise ValueError("Each image manifest entry must be a list of paths")
        if paths:
            vectors = []
            for path in paths:
                source = Path(args.images).parent / path
                with Image.open(source) as im:
                    tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
                with torch.inference_mode():
                    vectors.append(vision(tensor)[0].cpu().numpy())
            visual = np.mean(vectors, axis=0).astype(np.float32)
        digest = hashlib.sha256(seq.encode()).hexdigest()
        filename = digest + ".npz"
        # Include ID so identical sequences with different view sets cannot overwrite.
        filename = hashlib.sha256((pid + digest).encode()).hexdigest() + ".npz"
        np.savez_compressed(out / filename, sequence=states, image=visual)
        index[pid] = {"file": filename, "sequence_sha256": digest,
                      "image_present": bool(paths), "image_count": len(paths)}
        print(f"Prepared {pid}: {len(seq)} residues, {len(paths)} views")
    (out / "index.json").write_text(json.dumps({"encoder": MODEL,
        "image_encoder": "ResNet18_Weights.IMAGENET1K_V1", "proteins": index}, indent=2))


if __name__ == "__main__":
    main()
