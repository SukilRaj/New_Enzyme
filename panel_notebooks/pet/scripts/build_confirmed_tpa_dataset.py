import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "tpa_activity_provenance_audit.csv"
REVIEW_PATH = PROJECT_ROOT / "reports" / "results" / "tpa_ambiguous_review.md"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "results"
CONFIRMED_PATH = OUTPUT_DIR / "tpa_pet_ml_confirmed.csv"
AMBIGUOUS_PATH = OUTPUT_DIR / "tpa_pet_ambiguous_excluded.csv"
SUMMARY_PATH = REPORT_DIR / "tpa_confirmed_dataset_summary.json"
PROTEIN_COUNTS_PATH = REPORT_DIR / "tpa_confirmed_protein_counts.csv"


def non_empty(series):
    return series.fillna("").astype(str).str.strip().ne("")


def numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce")
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std()),
    }


def main():
    audit = pd.read_csv(
        AUDIT_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    review = REVIEW_PATH.read_text(encoding="utf-8")
    reasons = re.findall(r"\*\*Reason:\*\* (.+)", review)

    ambiguous = audit[audit["classification"] == "TPA_AMBIGUOUS"].copy()
    confirmed = audit[audit["classification"] == "TPA_CONFIRMED"].copy()

    assert len(audit) == 227
    assert len(confirmed) == 221
    assert len(ambiguous) == 6
    assert len(reasons) == len(ambiguous)

    ambiguous.insert(
        len(ambiguous.columns),
        "exclusion_reason",
        reasons,
    )

    required_columns = [
        "protein_id",
        "protein_name",
        "sequence",
        "uniprot_accession",
        "uniprot_entry",
        "ec_number",
        "substrate",
        "target_name",
        "target_value",
        "target_unit",
        "target_value_standardized",
        "standardized_unit",
        "analyte",
        "metric_type",
        "temperature_c",
        "ph",
        "duration_value",
        "duration_unit",
        "duration_hours",
        "assay_method",
        "material_description",
        "source",
        "source_file",
        "source_record_id",
        "assay_index",
        "measurement_index",
        "replicate_n",
        "literature_doi",
        "conversion_formula",
        "molecular_weight_used",
        "conversion_status",
        "classification",
        "possible_duplicate",
        "condition_group_id",
        "repeated_condition",
        "zero_duration_flag",
        "original_record_json",
    ]
    missing_columns = [column for column in required_columns if column not in audit]
    assert not missing_columns, missing_columns

    required_confirmed_fields = [
        "sequence",
        "target_value",
        "temperature_c",
        "ph",
        "duration_value",
        "duration_unit",
    ]
    missing_confirmed_values = {
        column: int((~non_empty(confirmed[column])).sum())
        for column in required_confirmed_fields
    }
    missing_confirmed_values = {
        key: value for key, value in missing_confirmed_values.items() if value
    }
    assert not missing_confirmed_values, missing_confirmed_values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    confirmed.to_csv(CONFIRMED_PATH, index=False, encoding="utf-8-sig")
    ambiguous.to_csv(AMBIGUOUS_PATH, index=False, encoding="utf-8-sig")

    protein_counts = (
        confirmed.groupby(["protein_name", "protein_id"], as_index=False)
        .size()
        .rename(columns={"size": "record_count"})
        .sort_values(
            ["record_count", "protein_name", "protein_id"],
            ascending=[False, True, True],
        )
    )
    protein_counts.to_csv(PROTEIN_COUNTS_PATH, index=False, encoding="utf-8-sig")

    standardized_target = pd.to_numeric(
        confirmed["target_value_standardized"],
        errors="coerce",
    )
    confirmed_condition_counts = confirmed.groupby(
        "condition_group_id",
        dropna=False,
    ).size()
    confirmed_duplicate_counts = confirmed.groupby(
        list(audit.columns),
        dropna=False,
    ).size()
    confirmed_duplicate_groups = int((confirmed_duplicate_counts > 1).sum())
    confirmed_duplicate_rows = int(
        confirmed["possible_duplicate"].astype(str).str.lower().eq("true").sum()
    )
    available_uniprot = non_empty(confirmed.drop_duplicates("protein_id")["uniprot_accession"])
    zero_duration = confirmed["zero_duration_flag"].astype(str).str.lower().eq("true")

    summary = {
        "total_original_records": len(audit),
        "confirmed_records": len(confirmed),
        "excluded_ambiguous_records": len(ambiguous),
        "protein_count": int(confirmed["protein_id"].nunique()),
        "unique_proteins": int(confirmed["protein_id"].nunique()),
        "target_statistics": {
            "field": "target_value_standardized",
            **numeric_summary(standardized_target),
        },
        "condition_statistics": {
            "ph": {
                "min": float(pd.to_numeric(confirmed["ph"]).min()),
                "max": float(pd.to_numeric(confirmed["ph"]).max()),
            },
            "temperature_c": {
                "min": float(pd.to_numeric(confirmed["temperature_c"]).min()),
                "max": float(pd.to_numeric(confirmed["temperature_c"]).max()),
            },
            "duration_hours": {
                "min": float(pd.to_numeric(confirmed["duration_hours"]).min()),
                "max": float(pd.to_numeric(confirmed["duration_hours"]).max()),
            },
        },
        "unit_counts": confirmed["target_unit"].value_counts().to_dict(),
        "protein_record_counts": dict(
            zip(
                protein_counts["protein_id"],
                protein_counts["record_count"],
            )
        ),
        "zero_duration_count": int(zero_duration.sum()),
        "duplicate_group_count": confirmed_duplicate_groups,
        "duplicate_rows_involved": confirmed_duplicate_rows,
        "repeated_condition_group_count": int(
            (confirmed_condition_counts > 1).sum()
        ),
        "uniprot_available_count": int(available_uniprot.sum()),
        "uniprot_missing_count": int((~available_uniprot).sum()),
        "missing_confirmed_values": missing_confirmed_values,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert len(confirmed) == 221
    assert len(ambiguous) == 6
    assert set(confirmed["classification"]) == {"TPA_CONFIRMED"}
    assert not set(ambiguous.index).intersection(confirmed.index)
    assert not missing_confirmed_values

    print("========================================")
    print("ENZAIme CONFIRMED DATASET")
    print("========================================")
    print("Original records:", len(audit))
    print("Confirmed TPA records:", len(confirmed))
    print("Excluded ambiguous records:", len(ambiguous))
    print("Unique proteins:", summary["unique_proteins"])
    print()
    print("Target:")
    print("Min:", summary["target_statistics"]["min"])
    print("Median:", summary["target_statistics"]["median"])
    print("Mean:", summary["target_statistics"]["mean"])
    print("Max:", summary["target_statistics"]["max"])
    print()
    print("pH:")
    print("Min:", summary["condition_statistics"]["ph"]["min"])
    print("Max:", summary["condition_statistics"]["ph"]["max"])
    print()
    print("Temperature:")
    print("Min:", summary["condition_statistics"]["temperature_c"]["min"])
    print("Max:", summary["condition_statistics"]["temperature_c"]["max"])
    print()
    print("Duration:")
    print("Min:", summary["condition_statistics"]["duration_hours"]["min"])
    print("Max:", summary["condition_statistics"]["duration_hours"]["max"])
    print()
    print("Exact duplicate groups:", summary["duplicate_group_count"])
    print("Repeated condition groups:", summary["repeated_condition_group_count"])
    print("Zero-duration rows:", summary["zero_duration_count"])
    print()
    print("UniProt available:", summary["uniprot_available_count"])
    print("UniProt missing:", summary["uniprot_missing_count"])
    print()
    print("Top proteins by record count:")
    print(protein_counts.to_string(index=False))
    print()
    print("Low-count proteins:")
    print(protein_counts[protein_counts["record_count"] <= 5].to_string(index=False))
    print()
    print("========================================")
    print("Confirmed dataset row count:", len(confirmed))
    print("Ambiguous dataset row count:", len(ambiguous))
    print("Missing confirmed required fields:", missing_confirmed_values)
    print("Original ML/master/model files modified: no")
    print("Model training performed: no")
    print("ESM-2 embeddings regenerated: no")
    print("CONFIRMED DATASET READY")


if __name__ == "__main__":
    main()