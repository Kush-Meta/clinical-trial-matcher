"""Patient Profile page — interactive form entry or demo patient selection."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings
from trial_matcher.schemas.patient import (
    ICD10Diagnosis,
    LabResult,
    PatientProfile,
    RxNormMedication,
    VitalSign,
)

st.set_page_config(page_title="Patient Profile — Clinical Trial Matcher", layout="wide")
st.title("👤 Patient Profile")

settings = get_settings()


@st.cache_data
def load_demo_patients() -> list[PatientProfile]:
    path = Path("data/precomputed/demo_patients.json")
    if path.exists():
        return [PatientProfile.model_validate(p) for p in json.loads(path.read_text())]
    return []


# ── Mode selector ─────────────────────────────────────────────────────────────

if settings.demo_mode:
    mode = "demo"
else:
    mode = st.radio(
        "Patient source",
        ["Enter patient manually", "Load demo patient"],
        horizontal=True,
        label_visibility="collapsed",
    )
    mode = "form" if "manually" in mode else "demo"

# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE
# ─────────────────────────────────────────────────────────────────────────────

if mode == "demo":
    patients = load_demo_patients()
    if not patients:
        st.error("No demo patients found. Run `scripts/precompute_demo.py` first.")
        st.stop()

    labels = [
        f"{p.patient_id} | Age {p.age or '?'} | {p.sex} | "
        f"{len(p.diagnoses)} dx | {len(p.medications)} meds"
        for p in patients
    ]
    idx = st.selectbox("Select demo patient", range(len(patients)), format_func=lambda i: labels[i])
    patient = patients[idx]
    st.session_state["selected_patient"] = patient.model_dump(mode="json")
    _show_patient(patient)

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE FORM
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.markdown("Fill in what you know — leave fields blank if data is unavailable.")

    with st.form("patient_form"):
        st.subheader("Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=60)
        with c2:
            sex = st.selectbox("Sex", ["M", "F", "Unknown"])
        with c3:
            enrollment_date = st.date_input("Enrollment date", value=date.today())
        with c4:
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=80.0, value=0.0,
                                   help="Leave 0 if unknown")

        st.subheader("Diagnoses")
        st.caption("Add up to 10 diagnoses. ICD-10 code is optional — the engine will fuzzy-match condition names.")
        dx_rows = []
        for i in range(5):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                name = st.text_input(f"Condition {i+1}", key=f"dx_name_{i}",
                                      placeholder="e.g. type 2 diabetes mellitus")
            with c2:
                code = st.text_input(f"ICD-10 {i+1}", key=f"dx_code_{i}",
                                      placeholder="e.g. E11.9")
            with c3:
                onset = st.date_input(f"Onset {i+1}", key=f"dx_onset_{i}",
                                       value=None, min_value=date(1940, 1, 1))
            if name:
                dx_rows.append((name, code, onset))

        st.subheader("Medications")
        st.caption("Active medications at enrollment.")
        med_rows = []
        for i in range(5):
            c1, c2, c3 = st.columns([3, 1, 2])
            with c1:
                mname = st.text_input(f"Drug {i+1}", key=f"med_name_{i}",
                                       placeholder="e.g. aspirin")
            with c2:
                dose = st.number_input(f"Dose (mg) {i+1}", key=f"med_dose_{i}",
                                        min_value=0.0, value=0.0)
            with c3:
                indication = st.text_input(f"Indication {i+1}", key=f"med_ind_{i}",
                                            placeholder="e.g. MI prevention")
            if mname:
                med_rows.append((mname, dose if dose > 0 else None, indication))

        st.subheader("Lab Results")
        st.caption("Most recent values. LOINC code optional.")
        lab_rows = []
        # Common labs as quick-entry
        common_labs = [
            ("HbA1c", "4548-4", "%"),
            ("Creatinine", "2160-0", "mg/dL"),
            ("Fasting Glucose", "1558-6", "mg/dL"),
            ("LDL Cholesterol", "2089-1", "mg/dL"),
            ("eGFR", "62238-1", "mL/min/1.73m²"),
        ]
        for lab_name, loinc, unit in common_labs:
            c1, c2 = st.columns([1, 2])
            with c1:
                val = st.number_input(f"{lab_name} ({unit})", key=f"lab_{loinc}",
                                       min_value=0.0, value=0.0,
                                       help=f"LOINC: {loinc}")
            with c2:
                days_ago = st.slider(f"{lab_name} — days ago", 1, 365, 30,
                                      key=f"lab_days_{loinc}")
            if val > 0:
                lab_rows.append((lab_name, loinc, val, unit, days_ago))

        st.subheader("Behavioral Flags")
        c1, c2, c3 = st.columns(3)
        with c1:
            drug_abuse = st.selectbox("Drug abuse", ["Unknown", "Yes", "No"])
        with c2:
            alcohol_abuse = st.selectbox("Alcohol abuse", ["Unknown", "Yes", "No"])
        with c3:
            supplements_raw = st.text_input("Dietary supplements",
                                             placeholder="fish oil, vitamin D, …")

        submitted = st.form_submit_button("💾 Save Patient", type="primary")

    if submitted:
        diagnoses = [
            ICD10Diagnosis(
                icd10_code=code or None,
                description=name,
                onset_date=onset,
                is_active=True,
            )
            for name, code, onset in dx_rows
        ]
        medications = [
            RxNormMedication(
                drug_name=mname,
                dose_mg=dose,
                indication=indication or "",
                start_date=enrollment_date - timedelta(days=365),
            )
            for mname, dose, indication in med_rows
        ]
        labs = [
            LabResult(
                lab_name=lab_name,
                loinc_code=loinc,
                value=val,
                unit=unit,
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=days_ago),
                    datetime.min.time(),
                ),
            )
            for lab_name, loinc, val, unit, days_ago in lab_rows
        ]
        vitals = []
        if bmi > 0:
            vitals.append(VitalSign(
                vital_name="bmi", value=bmi, unit="kg/m2",
                measured_datetime=datetime.combine(enrollment_date, datetime.min.time()),
            ))

        flag_map = {"Yes": True, "No": False, "Unknown": None}
        supplements = [s.strip() for s in supplements_raw.split(",") if s.strip()]

        patient = PatientProfile(
            patient_id=f"USER-{uuid.uuid4().hex[:6].upper()}",
            source="uploaded",
            age=int(age),
            sex=sex,
            enrollment_date=enrollment_date,
            diagnoses=diagnoses,
            medications=medications,
            labs=labs,
            vitals=vitals,
            has_drug_abuse=flag_map[drug_abuse],
            has_alcohol_abuse=flag_map[alcohol_abuse],
            dietary_supplements=supplements,
        )

        st.session_state["selected_patient"] = patient.model_dump(mode="json")
        st.success(f"Patient **{patient.patient_id}** saved. Go to Trial Search → Match Results.")
        _show_patient(patient)
    elif st.session_state.get("selected_patient"):
        patient = PatientProfile.model_validate(st.session_state["selected_patient"])
        _show_patient(patient)


# ─────────────────────────────────────────────────────────────────────────────
# Shared display helper
# ─────────────────────────────────────────────────────────────────────────────

def _show_patient(patient: PatientProfile) -> None:
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", patient.patient_id)
    c2.metric("Age", patient.age or "—")
    c3.metric("Sex", patient.sex)
    c4.metric("Completeness", f"{int(patient.data_completeness_score * 100)}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🩺 Diagnoses", "💊 Medications", "🔬 Labs", "🔴 Behavioral"])

    with tab1:
        if patient.diagnoses:
            st.dataframe(pd.DataFrame([
                {"ICD-10": d.icd10_code or "—", "Condition": d.description,
                 "Onset": str(d.onset_date or "—"), "Active": d.is_active}
                for d in patient.diagnoses
            ]), use_container_width=True)
        else:
            st.info("No diagnoses recorded.")

    with tab2:
        if patient.medications:
            st.dataframe(pd.DataFrame([
                {"Drug": m.drug_name, "Dose": f"{m.dose_mg}mg" if m.dose_mg else "—",
                 "Indication": m.indication or "—",
                 "Start": str(m.start_date or "—"), "Stop": str(m.stop_date or "Ongoing")}
                for m in patient.medications
            ]), use_container_width=True)
        else:
            st.info("No medications recorded.")

    with tab3:
        if patient.labs:
            st.dataframe(pd.DataFrame([
                {"Test": l.lab_name, "Value": f"{l.value} {l.unit}",
                 "Date": str(l.result_date), "Abnormal": "⚠️" if l.is_abnormal else "✓"}
                for l in sorted(patient.labs, key=lambda x: x.result_datetime, reverse=True)
            ]), use_container_width=True)
        else:
            st.info("No lab results recorded.")

    with tab4:
        flag_icon = {True: "🔴 Yes", False: "✅ No", None: "❓ Unknown"}
        c1, c2 = st.columns(2)
        c1.metric("Drug Abuse", flag_icon.get(patient.has_drug_abuse, "❓"))
        c2.metric("Alcohol Abuse", flag_icon.get(patient.has_alcohol_abuse, "❓"))
        if patient.dietary_supplements:
            st.info(f"Supplements: {', '.join(patient.dietary_supplements)}")
