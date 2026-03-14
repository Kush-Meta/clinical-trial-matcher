"""Patient Profile page — display and select patient."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings
from trial_matcher.schemas.patient import PatientProfile

st.set_page_config(page_title="Patient Profile — Clinical Trial Matcher", layout="wide")
st.title("👤 Patient Profile")

settings = get_settings()


@st.cache_data
def load_demo_patients() -> list[PatientProfile]:
    path = Path("data/precomputed/demo_patients.json")
    if path.exists():
        data = json.loads(path.read_text())
        return [PatientProfile.model_validate(p) for p in data]
    return []


patients = load_demo_patients()

if not patients:
    st.error("No patient data found. Run `scripts/precompute_demo.py` first.")
    st.stop()

# Patient selector
patient_labels = [
    f"{p.patient_id} | Age {p.age or '?'} | {p.sex} | {len(p.diagnoses)} dx | {len(p.medications)} meds"
    for p in patients
]
selected_idx = st.selectbox("Select Patient", range(len(patients)), format_func=lambda i: patient_labels[i])
patient = patients[selected_idx]

# Store in session state for other pages
st.session_state["selected_patient"] = patient.model_dump(mode="json")

st.divider()

# Header row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patient ID", patient.patient_id)
with col2:
    st.metric("Age", patient.age or "Unknown")
with col3:
    st.metric("Sex", patient.sex)
with col4:
    completeness_pct = int(patient.data_completeness_score * 100)
    st.metric("Data Completeness", f"{completeness_pct}%")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🩺 Diagnoses", "💊 Medications", "🔬 Labs", "📊 Vitals", "🔴 Behavioral"])

with tab1:
    if patient.diagnoses:
        dx_data = [
            {
                "ICD-10": dx.icd10_code,
                "Description": dx.description or "(no description)",
                "Onset": str(dx.onset_date) if dx.onset_date else "Unknown",
                "Status": "Active" if dx.is_active else "Resolved",
                "Source": dx.source,
            }
            for dx in patient.diagnoses
        ]
        st.dataframe(pd.DataFrame(dx_data), use_container_width=True)
    else:
        st.info("No diagnoses recorded.")

with tab2:
    if patient.medications:
        med_data = [
            {
                "Drug": m.drug_name,
                "RxNorm": m.rxnorm_code or "—",
                "Dose": f"{m.dose_mg}mg" if m.dose_mg else "—",
                "Route": m.route or "—",
                "Indication": m.indication or "—",
                "Start": str(m.start_date) if m.start_date else "Unknown",
                "Stop": str(m.stop_date) if m.stop_date else "Ongoing",
            }
            for m in patient.medications
        ]
        st.dataframe(pd.DataFrame(med_data), use_container_width=True)
    else:
        st.info("No medications recorded.")

with tab3:
    if patient.labs:
        lab_data = [
            {
                "Test": l.lab_name,
                "LOINC": l.loinc_code or "—",
                "Value": f"{l.value} {l.unit}",
                "Reference": (
                    f"{l.reference_low}–{l.reference_high} {l.unit}"
                    if l.reference_low and l.reference_high else "—"
                ),
                "Date": str(l.result_date),
                "Abnormal": "⚠️" if l.is_abnormal else "✓",
                "Source": l.source,
            }
            for l in sorted(patient.labs, key=lambda x: x.result_datetime, reverse=True)
        ]
        st.dataframe(pd.DataFrame(lab_data), use_container_width=True)
    else:
        st.info("No lab results recorded.")

with tab4:
    if patient.vitals:
        vital_data = [
            {
                "Vital": v.vital_name,
                "Value": f"{v.value} {v.unit}",
                "Date": str(v.measured_datetime.date()),
            }
            for v in patient.vitals
        ]
        st.dataframe(pd.DataFrame(vital_data), use_container_width=True)
    else:
        st.info("No vitals recorded.")

with tab5:
    col1, col2, col3 = st.columns(3)
    with col1:
        flag_icon = {True: "🔴 Yes", False: "✅ No", None: "❓ Unknown"}
        st.metric("Drug Abuse", flag_icon.get(patient.has_drug_abuse, "❓ Unknown"))
    with col2:
        st.metric("Alcohol Abuse", flag_icon.get(patient.has_alcohol_abuse, "❓ Unknown"))
    with col3:
        st.metric("Smoking History", flag_icon.get(patient.has_smoking_history, "❓ Unknown"))

    if patient.dietary_supplements:
        st.info(f"**Dietary Supplements**: {', '.join(patient.dietary_supplements)}")
    else:
        st.info("No dietary supplements recorded.")
