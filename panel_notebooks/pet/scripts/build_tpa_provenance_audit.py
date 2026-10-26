import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "master_dataset.csv"
ML_PATH = PROJECT_ROOT / "data" / "tpa_pet_ml_dataset.csv"
AUDIT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "results"
AUDIT_PATH = AUDIT_DIR / "tpa_activity_provenance_audit.csv"
SUMMARY_PATH = REPORT_DIR / "tpa_provenance_audit_summary.json"
AMBIGUOUS_PATH = REPORT_DIR / "tpa_provenance_ambiguous.csv"

MW_TPA = 166.13
CONDITION_COLUMNS = [
    "protein_id",
    "temperature_c",
    "ph",
    "duration_value",
    "duration_unit",
    "substrate",
]


def parse_json(value, default):
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def as_text(value):
    if value is None:
        return ""
    return str(value).strip()


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_equal(left, right):
    left_number = as_number(left)
    right_number = as_number(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= max(
            1e-9,
            abs(right_number) * 1e-9,
        )
    return as_text(left) == as_text(right)


def standardize(value, unit, analyte_is_tpa):
    numeric_value = as_number(value)
    normalized_unit = as_text(unit).replace("μ", "µ")
    if numeric_value is None:
        return "", "", "", "invalid_original_value"
    if not analyte_is_tpa:
        return "", "", "", "not_converted_analyte_not_confirmed_tpa"
    if normalized_unit in {"µM", "um", "uM", "µmol/L", "umol/L"}:
        return numeric_value, "µM", "value", "converted"
    if normalized_unit == "mM":
        return numeric_value * 1000, "µM", "value * 1000", "converted"
    if normalized_unit == "µg/mL":
        return (
            numeric_value * 1000 / MW_TPA,
            "µM",
            "value * 1000 / 166.13",
            "converted",
        )
    return "", "", "", "unsupported_unit"


def source_record_matches(row, record):
    conditions = parse_json(record.get("conditions", ""), {})
    if not isinstance(conditions, dict):
        return []
    for row_column, source_column in [
        ("temperature_c", "temperature_c"),
        ("ph", "ph"),
        ("duration_value", "duration_value"),
    ]:
        if source_column not in conditions or not values_equal(
            row[row_column], conditions[source_column]
        ):
            return []
    if as_text(row["duration_unit"]) != as_text(
        conditions.get("duration_unit", "")
    ):
        return []
    if as_text(row["substrate"]) != as_text(record.get("substrate", "")):
        return []

    matches = []
    measurements = parse_json(record.get("measurements", ""), [])
    if not isinstance(measurements, list):
        return matches
    for measurement in measurements:
        if not isinstance(measurement, dict):
            continue
        standardized, _, _, status = standardize(
            measurement.get("value"),
            measurement.get("unit"),
            True,
        )
        target_standardized = standardize(
            row["target_activity"],
            row["unit"],
            True,
        )[0]
        if (
            status == "converted"
            and standardized != ""
            and target_standardized != ""
            and values_equal(standardized, target_standardized)
        ):
            matches.append(measurement)
    return matches


def record_json(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def source_fields(records, field):
    values = []
    for record in records:
        value = record.get(field, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        value = as_text(value)
        if value and value not in values:
            values.append(value)
    return "; ".join(values)


def matched_measurements(records, matches):
    output = []
    for record, measurements in zip(records, matches):
        for measurement in measurements:
            output.append((record, measurement))
    return output


def classification(matched):
    if not matched:
        return "NO_TPA_MATCH"
    analytes = [as_text(measurement.get("analyte")) for _, measurement in matched]
    if any(not analyte for analyte in analytes):
        return "UNRESOLVED"
    tpa_matches = [
        analyte for analyte in analytes if analyte.upper() == "TPA"
    ]
    if not tpa_matches:
        return "NO_TPA_MATCH"
    if any(analyte and analyte.upper() != "TPA" for analyte in analytes):
        return "TPA_AMBIGUOUS"
    return "TPA_CONFIRMED"


def condition_group_id(row):
    values = [as_text(row[column]) for column in CONDITION_COLUMNS]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"condition_{digest}"


def build_source_index(master):
    source_index = {}
    for _, row in master.iterrows():
        records = parse_json(row["activity_records_json"], [])
        source_index[as_text(row["protein_id"])] = records
    return source_index


def main():
    master = pd.read_csv(
        MASTER_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    ml = pd.read_csv(
        ML_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    master_by_id = master.set_index("protein_id", drop=False)
    source_index = build_source_index(master)

    rows = []
    candidate_cache = {}
    for row_index, row in ml.iterrows():
        protein_id = as_text(row["protein_id"])
        candidate_records = []
        candidate_measurements = []
        for record in source_index.get(protein_id, []):
            measurements = source_record_matches(row, record)
            if measurements:
                candidate_records.append(record)
                candidate_measurements.append(measurements)
        matched = matched_measurements(
            candidate_records,
            candidate_measurements,
        )
        candidate_cache[row_index] = (
            candidate_records,
            candidate_measurements,
            matched,
        )

    duplicate_key_counts = ml.groupby(
        list(ml.columns),
        dropna=False,
    ).size()
    duplicate_keys = {
        key for key, count in duplicate_key_counts.items() if count > 1
    }
    condition_key_counts = ml.groupby(
        CONDITION_COLUMNS,
        dropna=False,
    ).size()
    condition_ids = {
        tuple(as_text(value) for value in key): condition_group_id(
            dict(zip(CONDITION_COLUMNS, key))
        )
        for key in condition_key_counts.index
    }

    for row_index, row in ml.iterrows():
        candidate_records, _, matched = candidate_cache[row_index]
        master_row = master_by_id.loc[as_text(row["protein_id"])]
        record_classification = classification(matched)
        analytes = sorted(
            {
                as_text(measurement.get("analyte"))
                for _, measurement in matched
                if as_text(measurement.get("analyte"))
            }
        )
        metric_types = sorted(
            {
                as_text(measurement.get("metric_type"))
                for _, measurement in matched
                if as_text(measurement.get("metric_type"))
            }
        )
        tpa_confirmed = record_classification == "TPA_CONFIRMED"
        standardized_value, standardized_unit, formula, conversion_status = (
            standardize(
                row["target_activity"],
                row["unit"],
                tpa_confirmed,
            )
        )
        if record_classification == "TPA_AMBIGUOUS":
            conversion_status = "not_converted_ambiguous_analyte"
            standardized_value = ""
            standardized_unit = ""
            formula = ""
        key = tuple(as_text(row[column]) for column in CONDITION_COLUMNS)
        full_key = tuple(row[column] for column in ml.columns)
        output = {
            "protein_id": row["protein_id"],
            "protein_name": row["protein_name"],
            "sequence": row["sequence"],
            "uniprot_accession": master_row.get("uniprot_accessions", ""),
            "uniprot_entry": master_row.get("uniprot_entry", ""),
            "ec_number": master_row.get("ec_number", ""),
            "substrate": row["substrate"],
            "target_name": "; ".join(analytes),
            "target_value": row["target_activity"],
            "target_unit": row["unit"],
            "target_value_standardized": standardized_value,
            "standardized_unit": standardized_unit,
            "analyte": "; ".join(analytes),
            "metric_type": "; ".join(metric_types),
            "temperature_c": row["temperature_c"],
            "ph": row["ph"],
            "duration_value": row["duration_value"],
            "duration_unit": row["duration_unit"],
            "duration_hours": (
                as_number(row["duration_value"])
                * (24 if row["duration_unit"] == "d" else 1)
            ),
            "assay_method": source_fields(candidate_records, "detection_methods"),
            "material_description": source_fields(candidate_records, "material"),
            "source": source_fields(candidate_records, "evidence"),
            "source_file": source_fields(candidate_records, "source_file"),
            "source_record_id": "",
            "assay_index": source_fields(candidate_records, "assay_index"),
            "measurement_index": "",
            "replicate_n": "; ".join(
                sorted(
                    {
                        as_text(measurement.get("replicate_n"))
                        for _, measurement in matched
                        if as_text(measurement.get("replicate_n"))
                    }
                )
            ),
            "literature_doi": source_fields(
                candidate_records,
                "literature_doi",
            ),
            "conversion_formula": formula,
            "molecular_weight_used": MW_TPA if row["unit"] == "µg/mL" and tpa_confirmed else "",
            "conversion_status": conversion_status,
            "classification": record_classification,
            "possible_duplicate": full_key in duplicate_keys,
            "condition_group_id": condition_ids[key],
            "repeated_condition": condition_key_counts.loc[key] > 1,
            "zero_duration_flag": as_number(row["duration_value"]) == 0,
            "original_record_json": json.dumps(
                candidate_records,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        rows.append(output)

    audit = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")

    ambiguous = audit[audit.classification.isin(["TPA_AMBIGUOUS", "UNRESOLVED"])]
    ambiguous.to_csv(AMBIGUOUS_PATH, index=False, encoding="utf-8-sig")

    duplicate_groups = ml[ml.duplicated(list(ml.columns), keep=False)].drop_duplicates(list(ml.columns))
    repeated_rows = audit[audit.repeated_condition]
    matched_count = int(audit.original_record_json.ne("[]").sum())
    analyte_counts = Counter()
    for analyte_value in audit["analyte"]:
        analyte_counts.update(
            analyte
            for analyte in analyte_value.split("; ")
            if analyte
        )
    summary = {
        "total_ml_records": len(ml),
        "matched_records": matched_count,
        "unmatched_records": len(ml) - matched_count,
        "classification_counts": {
            key: int(audit.classification.eq(key).sum())
            for key in [
                "TPA_CONFIRMED",
                "TPA_AMBIGUOUS",
                "NO_TPA_MATCH",
                "UNRESOLVED",
            ]
        },
        "unit_counts": ml["unit"].value_counts().to_dict(),
        "analyte_counts": dict(analyte_counts),
        "protein_counts": ml["protein_name"].value_counts().to_dict(),
        "source_counts": audit["source_file"].replace("", "(blank)").value_counts().to_dict(),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_rows": int(audit.possible_duplicate.sum()),
        "repeated_condition_groups": int(condition_key_counts.gt(1).sum()),
        "repeated_condition_rows": len(repeated_rows),
        "zero_duration_rows": int(audit.zero_duration_flag.sum()),
        "conversion_status_counts": audit["conversion_status"].value_counts().to_dict(),
        "output_row_count": len(audit),
        "source_record_id_available": False,
        "measurement_index_available": False,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("1. Total ML rows:", len(ml))
    print("2. Matched rows:", matched_count)
    print("3. Unmatched rows:", len(ml) - matched_count)
    for key, value in summary["classification_counts"].items():
        print(f"{key} count:", value)
    print("8. Exact duplicate groups:", len(duplicate_groups))
    print("9. Repeated-condition groups:", summary["repeated_condition_groups"])
    print("10. Zero-duration rows:", summary["zero_duration_rows"])
    print("11. Unit counts:", summary["unit_counts"])
    print("12. Conversion status counts:", summary["conversion_status_counts"])
    print("13. First 10 ambiguous/unresolved records:")
    print(
        ambiguous[
            [
                "protein_id",
                "protein_name",
                "target_value",
                "target_unit",
                "temperature_c",
                "ph",
                "duration_value",
                "duration_unit",
                "analyte",
                "classification",
            ]
        ].head(10).to_string(index=False)
    )
    print("Output row count:", len(audit))
    print("Raw/master/model files modified: no")
    print("Model training performed: no")
    print("READY FOR REVIEW")


if __name__ == "__main__":
    main()