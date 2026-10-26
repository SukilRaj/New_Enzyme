# TPA Ambiguous Record Review

## Summary

Total ambiguous records: 6

All six rows have source evidence containing an explicit TPA measurement. None can be promoted to `TPA_CONFIRMED` because the ML row does not retain enough assay/measurement provenance to prove that its numeric value came from the TPA measurement rather than another source measurement with the same value. No row is classified as `TPA_NOT_CONFIRMED` because every reviewed source candidate contains TPA.

The provenance audit contains no native `source_record_id` or `measurement_index`. The review therefore uses the available `assay_index`, DOI, source evidence, conditions, and complete nested measurement records.

## Record-by-record review

### Record 1

| Field | Value |
|---|---|
| ML row identifier | Audit row 1; original ML row index 0 |
| `protein_id` | 41 |
| `protein_name` | PET46 |
| Substrate | PET |
| Temperature | 30.0 °C |
| pH | 8.0 |
| Duration | 72 h |
| Original target | 1.91 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | TPA; MHET; BHET; TPA + MHET + BHET (total aromatic products) |
| `metric_type` | `product_concentration`; also a `conversion_percent` record is present |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.1038/s42004-023-00998-z` |
| Assay index | 1 |

**Source evidence:** The assay uses RP-UHPLC with TPA, MHET, and BHET standards. At the matching conditions, the source reports:

- TPA: 1.91 µM, directly reported, triplicate, SD 0.77.
- MHET: 0 µM, triplicate.
- BHET: 0 µM, triplicate.
- Total aromatic products `(TPA + MHET + BHET)`: 1.91 µM, directly reported, triplicate, SD 0.77.
- PET conversion: 0%, a separate percentage metric.

**Interpretation:** TPA is explicitly measured, but the exact same numeric value is also explicitly reported as total aromatic products. The ML row does not preserve a measurement index identifying which of those two measurements became the target.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** The source proves that TPA was measured, but it also reports the same value for a combined product metric. Numeric equality does not establish the selected analyte.

### Record 2

| Field | Value |
|---|---|
| ML row identifier | Audit row 2; original ML row index 1 |
| `protein_id` | 42 |
| `protein_name` | CalB |
| Substrate | PET |
| Temperature | 37.0 °C |
| pH | 7.0 |
| Duration | 28 d |
| Original target | 3.4 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | BHET; MHET; TPA |
| `metric_type` | `product_concentration` |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.1016/j.procbio.2016.07.023` |
| Assay index | 6; 12 |

**Source evidence:** Two source assays match the reduced ML condition key:

- Assay 6, pretreatment PT-A: BHET 3.6 µM, MHET 5.5 µM, TPA 3.4 µM.
- Assay 12, pretreatment PT-B: BHET 3.4 µM, MHET 6.9 µM, TPA 3.4 µM.

Both assays used HPLC-UV at 254 nm, commercial CalB preparation, duplicate observations, and 20 mg protein/g PET. The assays differ in PET pretreatment, but that distinction is absent from the ML dataset.

**Interpretation:** The target can be TPA from either assay, but in assay 12 the same value, 3.4 µM, is also BHET. The ML row does not retain the assay or measurement index.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** TPA is explicit, but the numeric value is also an explicit BHET value in one matching assay and the assay variant is not retained.

### Record 3

| Field | Value |
|---|---|
| ML row identifier | Audit row 3; original ML row index 2 |
| `protein_id` | 42 |
| `protein_name` | CalB |
| Substrate | PET |
| Temperature | 37.0 °C |
| pH | 7.0 |
| Duration | 28 d |
| Original target | 3.4 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | BHET; MHET; TPA |
| `metric_type` | `product_concentration` |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.1016/j.procbio.2016.07.023` |
| Assay index | 6; 12 |

**Source evidence:** This row has the same source candidates as Record 2:

- Assay 6: TPA 3.4 µM; BHET 3.6 µM; MHET 5.5 µM.
- Assay 12: TPA 3.4 µM; BHET 3.4 µM; MHET 6.9 µM.

The source reports HPLC-UV product concentrations and duplicate observations. The two assay variants use different PET pretreatments, which are not represented in the ML condition key.

**Interpretation:** The row is an exact duplicate of Record 2 at the ML level. TPA is explicitly present, but the same value is also BHET in assay 12, and the ML record cannot identify the selected assay or measurement.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** The duplicate ML rows do not resolve whether the source target was the TPA measurement or the coincident BHET measurement.

### Record 4

| Field | Value |
|---|---|
| ML row identifier | Audit row 4; original ML row index 3 |
| `protein_id` | 44 |
| `protein_name` | HiC |
| Substrate | PET |
| Temperature | 50.0 °C |
| pH | 7.0 |
| Duration | 0 d |
| Original target | 0.0 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | BHET; MHET; TPA |
| `metric_type` | `product_concentration` |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.1016/j.procbio.2016.07.023` |
| Assay index | 23; 28 |

**Source evidence:** Both matching assays report all three products as zero:

- Assay 23: BHET 0 µM, MHET 0 µM, TPA 0 µM.
- Assay 28: BHET 0 µM, MHET 0 µM, TPA 0 µM.

Both use HPLC-UV at 254 nm, duplicate observations, and PET at 50 °C and pH 7. Assay 23 uses a single 40 mg/g enzyme loading. Assay 28 uses a make-up strategy with 20 mg/g at day 0 and again at day 7.

**Interpretation:** The source explicitly reports TPA, but zero is simultaneously reported for BHET and MHET. The zero-duration condition also requires separate interpretation; the source does not establish that this is a valid activity endpoint rather than a baseline or time-zero observation.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** A zero numeric value cannot identify which of the three product measurements was selected, and the two assay strategies are not preserved in the ML row.

### Record 5

| Field | Value |
|---|---|
| ML row identifier | Audit row 5; original ML row index 4 |
| `protein_id` | 44 |
| `protein_name` | HiC |
| Substrate | PET |
| Temperature | 50.0 °C |
| pH | 7.0 |
| Duration | 0 d |
| Original target | 0.0 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | BHET; MHET; TPA |
| `metric_type` | `product_concentration` |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.1016/j.procbio.2016.07.023` |
| Assay index | 23; 28 |

**Source evidence:** This is an exact ML duplicate of Record 4. Assay 23 and assay 28 each explicitly report BHET, MHET, and TPA at 0 µM, using the same HPLC-UV method and duplicate observations. The assays differ in enzyme-loading strategy.

**Interpretation:** TPA is present in the source evidence, but the zero value is shared by all three reported products. The ML row does not identify the analyte or assay variant.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** The source contains TPA but cannot distinguish it from the coincident zero BHET or MHET measurements. The zero-duration interpretation is also unresolved.

### Record 6

| Field | Value |
|---|---|
| ML row identifier | Audit row 6; original ML row index 5 |
| `protein_id` | 185 |
| `protein_name` | SM14est |
| Substrate | PET |
| Temperature | 45.0 °C |
| pH | 8.0 |
| Duration | 7 h |
| Original target | 2.3 µM |
| Standardized value | Not available; intentionally not converted because the target mapping is ambiguous |
| Analytes | TPA; di-aromatic oligomers |
| `metric_type` | `product_concentration`; additional categorical and kinetic metrics are also present |
| Source file | `pazy_polymer_assays.json.gz` |
| DOI | `10.3389/fmicb.2023.1170880` |
| Assay index | 22; 26 |

**Source evidence:** Two source assays match the reduced ML conditions:

- Assay 22, 500 mM NaCl: TPA 9.4 µM; di-aromatic oligomers 2.3 µM; MHET 146.9 µM; BHET 19.3 µM; pooled aromatic products 270 µM.
- Assay 26, 0 mM NaCl: TPA 2.3 µM; di-aromatic oligomers 1.6 µM; MHET 17.3 µM; BHET 4.3 µM; pooled aromatic products 56 µM.

The source uses RP-HPLC with individual standards for TPA, MHET, and BHET, plus UV spectrophotometry for pooled products. Both assays report duplicate observations and also include a separate initial PET hydrolysis-rate metric.

**Interpretation:** The ML value 2.3 µM is explicitly TPA in assay 26. However, the same numeric value is explicitly the di-aromatic oligomer concentration in assay 22. The ML dataset does not retain the NaCl condition or assay index needed to distinguish them.

**Decision:** `STILL_AMBIGUOUS`

**Confidence:** High

**Reason:** The source explicitly identifies TPA, but the identical numeric value belongs to a different analyte under another matching assay condition.

## Final Decision Table

| Record | Protein | Decision | Confidence | Reason |
|---:|---|---|---|---|
| 1 | PET46 | STILL_AMBIGUOUS | High | 1.91 µM is reported both as TPA and total aromatic products. |
| 2 | CalB | STILL_AMBIGUOUS | High | 3.4 µM is TPA in both candidate assays and BHET in assay 12. |
| 3 | CalB | STILL_AMBIGUOUS | High | Exact duplicate of Record 2; assay and measurement identity are absent. |
| 4 | HiC | STILL_AMBIGUOUS | High | BHET, MHET, and TPA are all reported as 0 µM. |
| 5 | HiC | STILL_AMBIGUOUS | High | Exact duplicate of Record 4; zero cannot identify the analyte. |
| 6 | SM14est | STILL_AMBIGUOUS | High | 2.3 µM is TPA in assay 26 and di-aromatic oligomers in assay 22. |

No records are promoted to `TPA_CONFIRMED`. No records are classified as `TPA_NOT_CONFIRMED`, because every reviewed source candidate explicitly contains a TPA measurement. No CSV files were modified, no records were removed, and no model training was performed.

AMBIGUOUS REVIEW COMPLETE