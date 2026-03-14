"""MIMIC-IV patient data extractor (PostgreSQL or DuckDB/Parquet)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from trial_matcher.schemas.patient import (
    ICD10Diagnosis,
    LabResult,
    PatientProfile,
    RxNormMedication,
)

logger = logging.getLogger(__name__)

# ICD-9 → ICD-10 GEM crosswalk (simplified sample)
# Full crosswalk available from CMS: https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs
ICD9_TO_ICD10_SAMPLE: dict[str, str] = {
    "250.00": "E11.9",    # T2DM without complications
    "250.02": "E11.9",    # T2DM uncontrolled
    "401.9": "I10",       # Essential hypertension
    "410.90": "I21.9",   # Acute MI
    "414.01": "I25.10",  # CAD
    "585.3": "N18.3",    # CKD stage 3
}

# Standard LOINC codes for common labs
LOINC_MAP = {
    "HEMOGLOBIN A1C": "4548-4",
    "HBA1C": "4548-4",
    "CREATININE": "2160-0",
    "GLUCOSE": "1558-6",
    "POTASSIUM": "2823-3",
    "SODIUM": "2951-2",
    "BUN": "3094-0",
    "ALT": "1742-6",
    "AST": "1920-8",
    "TOTAL BILIRUBIN": "1975-2",
}


class MIMICExtractor:
    """
    Extracts PatientProfile from MIMIC-IV database.

    Supports two backends:
    - PostgreSQL (via psycopg2): for full MIMIC-IV setup
    - DuckDB (via duckdb): for Parquet-based setup (simpler local dev)
    """

    def __init__(self, db_url: str | None = None, parquet_dir: str | None = None) -> None:
        self.db_url = db_url
        self.parquet_dir = parquet_dir
        self._conn: Any = None
        self._backend: str | None = None

    def connect(self) -> None:
        if self.db_url:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self.db_url)
                self._backend = "postgresql"
                logger.info("Connected to MIMIC-IV PostgreSQL")
            except ImportError:
                raise RuntimeError("psycopg2 required: pip install psycopg2-binary")
        elif self.parquet_dir:
            try:
                import duckdb
                self._conn = duckdb.connect()
                self._backend = "duckdb"
                logger.info("Connected to MIMIC-IV via DuckDB/Parquet")
            except ImportError:
                raise RuntimeError("duckdb required: pip install duckdb")
        else:
            raise ValueError("Either db_url or parquet_dir must be provided")

    def extract_patient(
        self, subject_id: int, hadm_id: int | None = None,
        enrollment_date: date | None = None,
    ) -> PatientProfile:
        """Extract full PatientProfile for a MIMIC-IV patient."""
        if self._conn is None:
            self.connect()

        if enrollment_date is None:
            enrollment_date = date.today()

        diagnoses = self._extract_diagnoses(subject_id, hadm_id)
        medications = self._extract_medications(subject_id, hadm_id)
        labs = self._extract_labs(subject_id)

        return PatientProfile(
            patient_id=f"MIMIC-{subject_id}",
            source="mimic_iv",
            enrollment_date=enrollment_date,
            diagnoses=diagnoses,
            medications=medications,
            labs=labs,
        )

    def _extract_diagnoses(
        self, subject_id: int, hadm_id: int | None
    ) -> list[ICD10Diagnosis]:
        query = """
            SELECT d.icd_code, d.icd_version, p.admittime, p.dischtime
            FROM diagnoses_icd d
            JOIN admissions p ON d.subject_id = p.subject_id AND d.hadm_id = p.hadm_id
            WHERE d.subject_id = %(subject_id)s
            ORDER BY p.admittime DESC
        """
        if hadm_id:
            query = query.replace(
                "WHERE d.subject_id = %(subject_id)s",
                "WHERE d.subject_id = %(subject_id)s AND d.hadm_id = %(hadm_id)s",
            )

        rows = self._execute(query, {"subject_id": subject_id, "hadm_id": hadm_id})
        diagnoses = []
        for icd_code, icd_version, admittime, dischtime in rows:
            # Convert ICD-9 to ICD-10 if needed
            if icd_version == 9:
                normalized = ICD9_TO_ICD10_SAMPLE.get(icd_code, f"ICD9-{icd_code}")
            else:
                normalized = icd_code

            onset = admittime.date() if admittime else None
            resolved = dischtime.date() if dischtime else None
            diagnoses.append(ICD10Diagnosis(
                icd10_code=normalized,
                description="",
                onset_date=onset,
                resolved_date=resolved,
                is_active=resolved is None,
                source="structured",
            ))
        return diagnoses

    def _extract_medications(
        self, subject_id: int, hadm_id: int | None
    ) -> list[RxNormMedication]:
        query = """
            SELECT drug, ndc, starttime, stoptime, dose_val_rx, dose_unit_rx, route
            FROM prescriptions
            WHERE subject_id = %(subject_id)s
        """
        rows = self._execute(query, {"subject_id": subject_id})
        medications = []
        for drug, ndc, starttime, stoptime, dose_val, dose_unit, route in rows:
            try:
                dose_mg = float(dose_val) if dose_val else None
            except (ValueError, TypeError):
                dose_mg = None

            medications.append(RxNormMedication(
                drug_name=drug or "",
                dose_mg=dose_mg if dose_unit and "mg" in str(dose_unit).lower() else None,
                route=route or "",
                start_date=starttime.date() if starttime else None,
                stop_date=stoptime.date() if stoptime else None,
                source="structured",
            ))
        return medications

    def _extract_labs(self, subject_id: int) -> list[LabResult]:
        query = """
            SELECT d.label, l.valuenum, l.valueuom, l.charttime,
                   l.ref_range_lower, l.ref_range_upper, l.flag
            FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            WHERE l.subject_id = %(subject_id)s
              AND l.valuenum IS NOT NULL
            ORDER BY l.charttime DESC
        """
        rows = self._execute(query, {"subject_id": subject_id})
        labs = []
        for label, valuenum, valueuom, charttime, ref_low, ref_high, flag in rows:
            loinc = LOINC_MAP.get((label or "").upper())
            labs.append(LabResult(
                lab_name=label or "",
                loinc_code=loinc,
                value=float(valuenum),
                unit=valueuom or "",
                reference_low=float(ref_low) if ref_low else None,
                reference_high=float(ref_high) if ref_high else None,
                result_datetime=charttime if isinstance(charttime, datetime) else datetime.combine(charttime, datetime.min.time()),
                is_abnormal=flag in ("abnormal", "critical") if flag else False,
                source="structured",
            ))
        return labs

    def _execute(self, query: str, params: dict) -> list:
        if self._backend == "postgresql":
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
        elif self._backend == "duckdb":
            result = self._conn.execute(
                query.replace("%(", "?").replace(")s", ""),
                list(params.values()),
            )
            return result.fetchall()
        return []

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MIMICExtractor":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
