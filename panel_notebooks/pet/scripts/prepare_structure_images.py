"""Build sequence-verified C-alpha distance-map images from AlphaFold DB structures.

Images are deterministic structural representations, not experimental labels.
Unmatched/mutated/truncated structures are excluded rather than silently assigned.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import numpy as np
from PIL import Image
from backend.app.services.mutation import protein_catalog
from backend.app.services.structure import parse_pdb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/structure_images")
    args = p.parse_args()
    out = Path(args.output)
    (out / "pdb").mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(exist_ok=True)
    matched, excluded, view_map = [], [], {}
    for protein in protein_catalog().values():
        pid = protein["protein_id"]
        accepted = False
        reasons = []
        for accession in protein["uniprot_accessions"]:
            try:
                meta_path = out / "pdb" / f"{accession}.json"
                if meta_path.is_file():
                    entries = json.loads(meta_path.read_text())
                else:
                    with urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=25) as response:
                        entries = json.load(response)
                    meta_path.write_text(json.dumps(entries))
                for entry in entries:
                    if entry.get("uniprotSequence") != protein["sequence"]:
                        continue
                    url = entry["pdbUrl"]
                    if urlparse(url).scheme != "https" or urlparse(url).hostname != "alphafold.ebi.ac.uk":
                        raise ValueError("Unexpected structure host")
                    pdb_path = out / "pdb" / f"{pid}.pdb"
                    if not pdb_path.is_file():
                        with urlopen(url, timeout=25) as response:
                            pdb_path.write_text(response.read().decode())
                    chains, _ = parse_pdb(pdb_path.read_text())
                    residues = next((r for r in chains.values() if ''.join(x['residue'] for x in r) == protein['sequence']), None)
                    if residues is None:
                        continue
                    xyz = np.array([r["xyz"] for r in residues], dtype=np.float32)
                    distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)
                    # Near residues bright, distant residues dark. No sequence labels/text in image.
                    pixels = np.uint8(np.clip(1-distances/20., 0, 1)*255)
                    filename = f"views/{pid}_distance.png"
                    Image.fromarray(pixels).convert("RGB").save(out / filename)
                    view_map[pid] = [filename]
                    matched.append({"protein_id": pid, "accession": accession, "structure_url": url,
                        "structure_kind": "AlphaFold predicted", "sequence_identity": 1.0,
                        "coverage": 1.0, "view": filename, "residues": len(residues)})
                    accepted = True
                    break
                if accepted:
                    break
                reasons.append(f"{accession}: no full exact-sequence structure")
            except (OSError, ValueError, KeyError) as error:
                reasons.append(f"{accession}: {error}")
        if not accepted:
            excluded.append({"protein_id": pid, "reason": reasons or ["No UniProt accession"]})
        print(f"{pid}: {'matched' if accepted else 'excluded'}", flush=True)
    (out / "protein_views.json").write_text(json.dumps(view_map, indent=2))
    manifest = {"matched_proteins": len(matched), "excluded_proteins": len(excluded),
        "representation": "C-alpha distance map, 20 Angstrom cutoff, full exact sequence only",
        "source": "https://alphafold.ebi.ac.uk/", "matched": matched, "excluded": excluded,
        "limitations": "Predicted structures; no experimental structural validation. Benefit of images must be tested with held-out ablations."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(matched)} verified structural images", flush=True)


if __name__ == "__main__":
    main()
