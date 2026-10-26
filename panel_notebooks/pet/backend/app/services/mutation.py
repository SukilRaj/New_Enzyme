"""Evidence browsing and sequence edits; no learned mutation-effect claims."""
import csv
from functools import lru_cache
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SUBSTITUTION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")


def read_csv(path):
    # Master rows contain nested assay JSON larger than csv's default field limit.
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@lru_cache(maxsize=1)
def protein_catalog():
    master = {r["protein_id"]: r for r in read_csv(ROOT / "master_dataset.csv")}
    catalog = {}
    for row in read_csv(ROOT / "data/processed/tpa_pet_ml_confirmed.csv"):
        pid = row["protein_id"]
        if pid in catalog:
            if catalog[pid]["sequence"] != row["sequence"].strip().upper():
                raise ValueError(f"Conflicting confirmed sequences for protein {pid}.")
            continue
        metadata = master.get(pid, {})
        catalog[pid] = {
            "protein_id": pid, "protein_name": row["protein_name"],
            "sequence": row["sequence"].strip().upper(),
            "uniprot_accessions": [x.strip() for x in metadata.get("uniprot_accessions", "").split(";") if x.strip()],
            "pdb_ids": [x.strip() for x in metadata.get("pdb_ids", "").split(";") if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", x.strip())],
        }
    return catalog


def get_protein(protein_id):
    protein = protein_catalog().get(protein_id)
    if protein is None:
        raise LookupError("Protein is not in the confirmed TPA/PET catalog.")
    return protein


def preview_mutations(protein_id, substitutions):
    protein = get_protein(protein_id)
    sequence = protein["sequence"]
    tokens = [x for x in re.split(r"[,;\s]+", substitutions.strip().upper()) if x]
    if not tokens or len(tokens) > 20:
        raise ValueError("Enter 1 to 20 substitutions, for example S160A, using one-based sequence positions.")
    changes, used = [], set()
    mutant = list(sequence)
    for token in tokens:
        match = SUBSTITUTION.fullmatch(token)
        if not match:
            raise ValueError(f"Invalid substitution {token!r}. Use a standard code such as S160A.")
        original, number, replacement = match.groups()
        position = int(number)
        if position > len(sequence):
            raise ValueError(f"{token}: position exceeds sequence length {len(sequence)}.")
        if position in used:
            raise ValueError(f"Position {position} occurs more than once; enter each position only once.")
        if original != sequence[position - 1]:
            raise ValueError(f"{token}: confirmed sequence has {sequence[position - 1]} at position {position}, not {original}.")
        if original == replacement:
            raise ValueError(f"{token} does not change the residue.")
        used.add(position)
        mutant[position - 1] = replacement
        changes.append({"notation": token, "position": position, "from_residue": original, "to_residue": replacement})
    return {
        "protein_id": protein_id, "protein_name": protein["protein_name"],
        "reference_sequence": sequence, "mutant_sequence": "".join(mutant),
        "changes": changes, "sequence_length": len(sequence),
        "numbering": "One-based positions in the confirmed project sequence; publication and PDB numbering may differ.",
        "status": "Sequence edit validated; biological effect unvalidated.",
        "activity_prediction": None, "stability_prediction": None,
    }


def mutation_evidence(protein_id):
    protein = get_protein(protein_id)
    source = ROOT / "data/fireprotdb/model_protein_mutations.csv"
    if not source.is_file():
        return {"protein_id": protein_id, "status": "unavailable", "records": [], "note": "The local FireProtDB extract is unavailable."}
    records = []
    for row in read_csv(source):
        if row.get("UNIPROTKB", "").strip() not in protein["uniprot_accessions"]:
            continue
        substitutions = row.get("SUBSTITUTION", "").strip()
        compatible, reason = False, "Reference measurement without a substitution."
        if substitutions and not row.get("INSERTION") and not row.get("DELETION"):
            try:
                preview_mutations(protein_id, substitutions)
                compatible, reason = True, "Residue letters match the project sequence at the reported positions."
            except ValueError as error:
                reason = str(error)
        elif row.get("INSERTION") or row.get("DELETION"):
            reason = "Insertions/deletions are not supported for sequence preview."
        records.append({
            "experiment_id": row["EXPERIMENT_ID"], "uniprot_accession": row["UNIPROTKB"],
            "substitutions": substitutions or None,
            "insertions": row.get("INSERTION") or None, "deletions": row.get("DELETION") or None,
            "reported_sequence_length": row.get("SEQUENCE_LENGTH") or None,
            "sequence_identity_status": "unresolved: source sequence is absent from the extract",
            "residue_check_passed": compatible, "residue_check_note": reason,
            # Keep raw values. This extract contains no measurement units or sign definitions.
            "measurements": {k: row[k] for k in ("TM", "DTM", "DDG", "DG", "PH", "EXP_TEMPERATURE") if row.get(k)},
            "measurement_note": "Raw source fields; units and sign conventions require checking the publication. Stability measurements do not establish TPA activity.",
            "method": row.get("METHOD") or None, "measure": row.get("MEASURE") or None,
            "buffer": row.get("BUFFER") or None, "buffer_concentration": row.get("BUFFER_CONC") or None,
            "doi": row.get("PUBLICATION_DOI") or None, "pmid": row.get("PUBLICATION_PMID") or None,
            "source_dataset": row.get("SOURCE_DATASET") or None,
        })
    return {"protein_id": protein_id, "status": "available" if records else "no_records", "records": records,
            "source": "data/fireprotdb/model_protein_mutations.csv",
            "note": "Accession-matched evidence only. Exact source sequence and numbering remain unverified. Multi-substitution results describe the whole variant, not individual substitutions."}
