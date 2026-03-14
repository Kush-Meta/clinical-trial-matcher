"""Trial Search page — ClinicalTrials.gov browser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings
from trial_matcher.trials.ctgov_client import CTGovClient, ClinicalTrial

st.set_page_config(page_title="Trial Search — Clinical Trial Matcher", layout="wide")
st.title("🔍 Clinical Trial Search")

settings = get_settings()


@st.cache_data(ttl=3600)
def load_demo_trials() -> list[dict]:
    path = Path("data/precomputed/demo_trials.json")
    if path.exists():
        return json.loads(path.read_text())
    return []


@st.cache_data(ttl=300, show_spinner="Searching ClinicalTrials.gov...")
def search_trials(query: str, max_results: int = 20) -> list[dict]:
    with CTGovClient() as client:
        trials = client.search(query, max_results=max_results)
    return [
        {
            "nct_id": t.nct_id,
            "brief_title": t.brief_title,
            "phase": t.phase,
            "overall_status": t.overall_status,
            "conditions": t.conditions,
            "eligibility_criteria": t.eligibility_criteria,
            "minimum_age": t.minimum_age,
            "maximum_age": t.maximum_age,
            "sex": t.sex,
            "lead_sponsor": t.lead_sponsor,
            "enrollment": t.enrollment,
        }
        for t in trials
    ]


# ── Search UI ────────────────────────────────────────────────────────────────

if settings.demo_mode:
    st.info("**Demo Mode**: Showing 10 pre-fetched trials. Search disabled to avoid API calls.", icon="ℹ️")
    trial_dicts = load_demo_trials()
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search ClinicalTrials.gov",
            value="type 2 diabetes mellitus",
            placeholder="e.g., type 2 diabetes, coronary artery disease...",
        )
    with col2:
        max_results = st.number_input("Max results", min_value=5, max_value=100, value=20, step=5)

    if st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            trial_dicts = search_trials(query, max_results)
        st.success(f"Found {len(trial_dicts)} trials")
    else:
        trial_dicts = load_demo_trials()

if not trial_dicts:
    st.warning("No trials loaded. Check your connection or run `scripts/precompute_demo.py`.")
    st.stop()

# Store selected trial in session state
st.divider()
st.markdown(f"**{len(trial_dicts)} trials found**")

for i, trial in enumerate(trial_dicts):
    with st.expander(
        f"**{trial['nct_id']}** — {trial['brief_title'][:100]}",
        expanded=(i == 0),
    ):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("Status")
            status_color = "🟢" if trial["overall_status"] == "RECRUITING" else "🟡"
            st.write(f"{status_color} {trial['overall_status']}")
        with col2:
            st.caption("Phase")
            st.write(trial.get("phase") or "—")
        with col3:
            st.caption("Age Range")
            min_age = trial.get("minimum_age", "")
            max_age = trial.get("maximum_age", "")
            age_range = f"{min_age} – {max_age}" if min_age or max_age else "Any"
            st.write(age_range)
        with col4:
            st.caption("Enrollment")
            st.write(trial.get("enrollment") or "—")

        if trial.get("conditions"):
            st.caption("**Conditions**")
            st.write(", ".join(trial["conditions"][:5]))

        if trial.get("lead_sponsor"):
            st.caption("**Sponsor**")
            st.write(trial["lead_sponsor"])

        if trial.get("eligibility_criteria"):
            with st.expander("📋 View Eligibility Criteria"):
                st.text(trial["eligibility_criteria"][:3000])

        if st.button(f"Select for Matching", key=f"select_{trial['nct_id']}"):
            st.session_state["selected_trial"] = trial
            st.success(f"Selected **{trial['nct_id']}** for matching → go to Match Results")
