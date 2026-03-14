"""Trial Search page — live ClinicalTrials.gov search."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings
from trial_matcher.trials.ctgov_client import CTGovClient

st.set_page_config(page_title="Trial Search — Clinical Trial Matcher", layout="wide")
st.title("🔍 Clinical Trial Search")

settings = get_settings()


@st.cache_data(ttl=3600)
def load_demo_trials() -> list[dict]:
    path = Path("data/precomputed/demo_trials.json")
    if path.exists():
        return json.loads(path.read_text())
    return []


@st.cache_data(ttl=600, show_spinner="Searching ClinicalTrials.gov…")
def search_trials(query: str, max_results: int) -> list[dict]:
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
        if t.eligibility_criteria  # only trials with parseable criteria
    ]


# ── Search bar (always live) ──────────────────────────────────────────────────

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    query = st.text_input(
        "Search",
        value="type 2 diabetes mellitus coronary artery disease",
        placeholder="Disease, drug, NCT ID, condition…",
        label_visibility="collapsed",
    )
with col2:
    max_results = st.selectbox("Max results", [10, 20, 50], index=0)
with col3:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# Run search or show defaults
if search_clicked and query:
    trial_dicts = search_trials(query, max_results)
    if not trial_dicts:
        st.warning("No recruiting trials found with eligibility criteria. Try a broader query.")
else:
    trial_dicts = load_demo_trials()
    if not search_clicked:
        st.info(
            "Showing 5 pre-loaded trials. Enter a search term and click Search to query "
            "ClinicalTrials.gov live.",
            icon="ℹ️",
        )

if not trial_dicts:
    st.stop()

st.caption(f"**{len(trial_dicts)} trials**")
st.divider()

# ── Trial cards ───────────────────────────────────────────────────────────────

for i, trial in enumerate(trial_dicts):
    status_icon = "🟢" if trial["overall_status"] == "RECRUITING" else "🟡"
    has_criteria = bool(trial.get("eligibility_criteria"))

    with st.expander(
        f"{status_icon} **{trial['nct_id']}** — {trial['brief_title'][:90]}",
        expanded=(i == 0),
    ):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", trial["overall_status"])
        c2.metric("Phase", trial.get("phase") or "—")
        age_str = " – ".join(filter(None, [trial.get("minimum_age"), trial.get("maximum_age")])) or "Any"
        c3.metric("Age range", age_str)
        c4.metric("Enrollment", trial.get("enrollment") or "—")

        if trial.get("conditions"):
            st.caption("**Conditions**")
            st.write(", ".join(trial["conditions"][:6]))
        if trial.get("lead_sponsor"):
            st.caption("**Sponsor**")
            st.write(trial["lead_sponsor"])

        if has_criteria:
            with st.expander("📋 Eligibility Criteria (raw)"):
                st.text(trial["eligibility_criteria"][:3000])
        else:
            st.warning("No eligibility criteria text available for this trial.")

        col_btn, col_status = st.columns([2, 3])
        with col_btn:
            if st.button(
                "✅ Select for Matching",
                key=f"select_{trial['nct_id']}",
                disabled=not has_criteria,
                type="primary",
            ):
                st.session_state["selected_trial"] = trial
                st.session_state.pop("match_result", None)  # clear stale result
                st.success(f"Selected **{trial['nct_id']}** → go to Match Results")

        with col_status:
            if st.session_state.get("selected_trial", {}).get("nct_id") == trial["nct_id"]:
                st.success("Currently selected for matching")
