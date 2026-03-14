"""Clinical Trial Matcher — Streamlit app entry point."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from trial_matcher.config import get_settings

settings = get_settings()


def load_precomputed() -> dict:
    """Load pre-computed demo data."""
    precomputed_path = Path("data/precomputed/demo_match_results.json")
    if precomputed_path.exists():
        return json.loads(precomputed_path.read_text())
    return {}


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/Clinical%20Trial-Matcher-blue", width=200)
    st.title("Clinical Trial Matcher")

    if settings.demo_mode:
        st.warning("**Demo Mode** — Pre-computed results (no LLM required)", icon="ℹ️")
    else:
        st.success("**Full Mode** — Live Ollama inference", icon="✅")
        st.caption(f"Model: `{settings.ollama_model}`")

    st.divider()
    st.markdown("""
    **Navigation**
    - 👤 Patient Profile
    - 🔍 Trial Search
    - ✅ Match Results
    - 📊 Evaluation Dashboard
    """)

    st.divider()
    st.caption("Built with Streamlit · Pydantic · Ollama")
    st.caption("[GitHub](https://github.com/yourusername/clinical-trial-matcher)")

# ── Home page ────────────────────────────────────────────────────────────────

st.title("🏥 Clinical Trial Eligibility Matching Engine")
st.markdown("""
A production-quality system that matches patient charts to clinical trial eligibility criteria —
with **typed criterion parsing**, **calibrated confidence**, and **abstention** when evidence is insufficient.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Criterion Types", "8", help="LAB_VALUE, DIAGNOSIS, MEDICATION, BEHAVIORAL, DEMOGRAPHIC, FUNCTIONAL, PROCEDURE, COMPOSITE")

with col2:
    st.metric("n2c2 Criteria", "13", help="n2c2 2018 shared task benchmark criteria")

with col3:
    abstain_pct = int(settings.abstention_confidence_threshold * 100)
    st.metric("Abstention Threshold", f"{abstain_pct}%", help="Predictions below this confidence are abstained")

with col4:
    st.metric("Demo Patients", "5", help="Synthetic patients with n2c2 prevalence distributions")

st.divider()

st.markdown("""
## How It Works

```
Patient Chart + Trial Eligibility Text
         ↓
   [Criteria Parser]          Two-pass: deterministic segmentation + LLM extraction
         ↓                    (Ollama llama3.1:8b via instructor)
   Typed Criterion Set        Discriminated union: LabValueCriterion, DiagnosisCriterion, …
         ↓
   [Matching Engine]          Per-criterion evaluators with evidence extraction
         ↓
   CriterionMatchResult       Status: SATISFIED | NOT_SATISFIED | ABSTAIN
   + Confidence [0,1]         Penalty model: data quality, temporal, source type
   + Evidence Snippets        Audit trail for every decision
         ↓
   TrialMatchResult           ELIGIBLE | INELIGIBLE | UNCERTAIN + match score
```

## Architecture Highlights

| Component | Detail |
|---|---|
| **Typed criteria** | Discriminated union drives all evaluation logic |
| **Abstention** | Below 35% confidence → ABSTAIN with reason code |
| **Temporal** | Date arithmetic with uncertainty bands for year-only dates |
| **ICD-10 matching** | Prefix match captures all specificity variants |
| **Calibration** | ECE metric + reliability diagram |
| **Benchmark** | n2c2 2018: 288 patients × 13 criteria |

→ Use the sidebar to navigate to **Patient Profile**, **Trial Search**, or **Match Results**.
""")
