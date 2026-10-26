"""Conservative PDB C-alpha inspection without folding or energy estimates."""
import hashlib
import math

from backend.app.services.mutation import get_protein

RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))


def parse_pdb(text):
    chains = {}
    first_model = None
    for line_number, line in enumerate(text.splitlines(), 1):
        record = line[:6].strip()
        if record == "MODEL":
            if first_model is not None:
                break
            first_model = line[10:14].strip() or "1"
        if record == "ENDMDL":
            break
        if record != "ATOM" or line[12:16].strip() != "CA":
            continue
        try:
            residue_name = line[17:20].strip()
            residue = RESIDUES.get(residue_name, "X")
            chain = line[21:22].strip()
            number = int(line[22:26])
            insertion = line[26:27].strip()
            coords = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            occupancy = float(line[54:60].strip() or "1")
            b_factor = float(line[60:66]) if line[60:66].strip() else None
            if not all(math.isfinite(x) for x in coords + [occupancy] + ([] if b_factor is None else [b_factor])) or occupancy < 0:
                raise ValueError()
        except (ValueError, IndexError):
            raise ValueError(f"Malformed C-alpha coordinate at PDB line {line_number}.") from None
        entries = chains.setdefault(chain, {})
        key = (number, insertion)
        candidate = {"residue": residue, "residue_name": residue_name, "residue_number": number,
                     "insertion_code": insertion, "chain": chain, "xyz": coords,
                     "occupancy": occupancy, "b_factor": b_factor, "altloc": line[16:17].strip()}
        old = entries.get(key)
        if old and old["residue_name"] != residue_name:
            raise ValueError("Alternate residue identities are unsupported; supply one conformation.")
        # Highest occupancy; on ties prefer blank, then lexical alternate location.
        order = lambda item: (-item["occupancy"], item["altloc"])
        if old is None or order(candidate) < order(old):
            entries[key] = candidate
    if not chains:
        raise ValueError("No ATOM C-alpha coordinates found. Upload a PDB coordinate file (mmCIF is not supported).")
    if sum(map(len, chains.values())) > 5000:
        raise ValueError("At most 5,000 C-alpha residues are supported; upload a smaller structure.")
    return {chain: list(entries.values()) for chain, entries in chains.items()}, first_model or "1"


def inspect_structure(protein_id, pdb_text, chain_id=None, position=None):
    protein = get_protein(protein_id)
    sequence = protein["sequence"]
    if position is not None and position > len(sequence):
        raise ValueError(f"Position exceeds the reference sequence length {len(sequence)}.")
    chains, model = parse_pdb(pdb_text)
    choices = [{"chain_id": key, "residue_count": len(value)} for key, value in chains.items()]
    if chain_id is None and len(chains) > 1:
        return {"status": "select_chain", "chains": choices, "note": "Select a chain before mapping sequence positions."}
    chosen = next(iter(chains)) if chain_id is None else chain_id
    if chosen not in chains:
        raise ValueError("Selected chain is absent from the first PDB model.")
    residues = chains[chosen]
    observed = "".join(r["residue"] for r in residues)
    start = sequence.find(observed)
    unique = start >= 0 and sequence.find(observed, start + 1) < 0 and len(observed) >= 10
    for index, residue in enumerate(residues):
        residue["sequence_position"] = start + index + 1 if unique else None
    site = next((r for r in residues if unique and r["sequence_position"] == position), None) if position is not None else None
    neighbors = []
    if site:
        for residue in residues:
            if residue is site:
                continue
            distance = math.dist(site["xyz"], residue["xyz"])
            if distance <= 8:
                neighbors.append({"sequence_position": residue["sequence_position"], "residue": residue["residue"],
                                  "residue_number": residue["residue_number"], "insertion_code": residue["insertion_code"],
                                  "ca_distance_angstrom": round(distance, 3)})
        neighbors.sort(key=lambda r: r["ca_distance_angstrom"])
    return {
        "status": "mapped" if unique else "unresolved", "protein_id": protein_id,
        "chains": choices, "chain_id": chosen, "model": model, "residues": residues,
        "sha256": hashlib.sha256(pdb_text.encode("utf-8")).hexdigest(),
        "mapping_method": "Unique exact contiguous match of observed C-alpha sequence (minimum 10 residues).",
        "note": ("Observed chain matches a unique reference segment. Coordinate numbering is preserved separately."
                 if unique else "No unique exact segment match. Gaps, substitutions, unknown residues or repeated segments require a reviewed alignment; no residue mapping is inferred."),
        "coverage": len(residues) / len(sequence) if unique else None,
        "reference_start": start + 1 if unique else None, "reference_end": start + len(residues) if unique else None,
        "requested_position": position, "site": site, "neighbors": neighbors,
        "site_status": "not_requested" if position is None else ("mapped" if site else "unmapped"),
        "limitations": "First model, ATOM C-alpha trace only. Highest-occupancy alternate location per residue. B-factors are retained as raw values, not interpreted as confidence. C-alpha distances are geometric context, not atom contacts, stability, docking or activity predictions.",
    }
