"""Match Results page — per-criterion audit trail (live or demo)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings
from trial_matcher.schemas.match import CriterionMatchResult, MatchStatus, TrialMatchResult

st.set_page_config(page_title="Match Results — Clinical Trial Matcher", layout="wide")
st.title("✅ Match Results")

settings = get_settings()

STATUS_ICON = {"SATISFIED": "🟢", "NOT_SATISFIED": "🔴", "ABSTAIN": "🟡"}
STATUS_BADGE = {
    "SATISFIED": "✅ SATISFIED",
    "NOT_SATISFIED": "❌ NOT SATISFIED",
    "ABSTAIN": "⚠️ ABSTAIN",
}


@st.cache_data
def load_precomputed() -> list[dict]:
    path = Path("data/precomputed/demo_match_results.json")
    return json.loads(path.read_text()) if path.exists() else []


def render_result(result: TrialMatchResult) -> None:
    elig_icon = {"ELIGIBLE": "🟢", "INELIGIBLE": "🔴", "UNCERTAIN": "🟡"}
    st.markdown(
        f"## {elig_icon.get(result.overall_eligibility, '❓')} "
        f"**{result.overall_eligibility}**"
    )
    st.caption(f"{result.nct_id} — {result.trial_title}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", f"{result.overall_confidence:.0%}")
    c2.metric("Inclusion met", f"{result.n_inclusion_satisfied}/{result.n_inclusion_total}")
    c3.metric("Abstained", result.n_inclusion_abstain)
    c4.metric("Exclusions triggered", result.n_exclusion_triggered)
    c5.metric("Match score", f"{result.match_score:.2f}")

    if result.blocking_criterion_summary:
        st.error(f"**Blocking**: {result.blocking_criterion_summary}")

    st.divider()

    def render_criteria_section(results: list[CriterionMatchResult], label: str) -> None:
        if not results:
            return
        st.subheader(f"{label} ({len(results)})")
        for r in results:
            is_excl = label.startswith("Excl")
            icon = STATUS_ICON.get(r.status.value, "❓")
            prefix = "EXCL" if is_excl else "INCL"
            with st.expander(
                f"{icon} [{prefix}] {r.criterion_summary[:100]}",
                expanded=(r.status == MatchStatus.NOT_SATISFIED),
            ):
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.caption("Status")
                    st.markdown(f"**{STATUS_BADGE.get(r.status.value, r.status.value)}**")
                    if r.abstention_reason:
                        st.caption(f"`{r.abstention_reason.value}`")
                with col2:
                    st.caption("Confidence")
                    st.progress(r.confidence, text=f"{r.confidence:.0%}")
                with col3:
                    st.caption("Type")
                    st.code(r.criterion_type.value)

                if r.raw_criterion_text:
                    st.caption("**Raw text**")
                    st.text(r.raw_criterion_text[:400])

                for ev in r.evidence:
                    date_str = f" ({ev.source_date})" if ev.source_date else ""
                    st.success(f"[{ev.source_type.upper()}{date_str}] {ev.text}")

                for sub in r.sub_results:
                    sub_icon = STATUS_ICON.get(sub.status.value, "❓")
                    st.write(f"  {sub_icon} {sub.criterion_summary[:80]} — {sub.confidence:.0%}")

    render_criteria_section(result.inclusion_results, "Inclusion Criteria")
    if result.exclusion_results:
        st.divider()
        render_criteria_section(result.exclusion_results, "Exclusion Criteria")

    # Ambiguity panel
    abstained = [r for r in result.get_all_results() if r.status == MatchStatus.ABSTAIN]
    if abstained:
        st.divider()
        st.subheader("⚠️ Abstention Panel")
        st.caption(
            "Criteria where evidence was insufficient to make a confident call. "
            "Abstention prevents false positives."
        )
        st.dataframe(pd.DataFrame([
            {
                "Criterion": r.criterion_summary[:80],
                "Type": r.criterion_type.value,
                "Reason": r.abstention_reason.value if r.abstention_reason else "—",
                "Confidence": f"{r.confidence:.0%}",
            }
            for r in abstained
        ]), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE — show pre-computed results
# ─────────────────────────────────────────────────────────────────────────────

if settings.demo_mode:
    precomputed = load_precomputed()
    if not precomputed:
        st.error("No pre-computed results. Run `scripts/precompute_demo.py`.")
        st.stop()
    st.info("**Demo Mode** — pre-computed results.", icon="ℹ️")
    patient_ids = list({r["patient_id"] for r in precomputed})
    pid = st.selectbox("Patient", patient_ids)
    patient_results = [r for r in precomputed if r["patient_id"] == pid]
    trial_labels = [f"{r['nct_id']} — {r['trial_title'][:60]}" for r in patient_results]
    tidx = st.selectbox("Trial", range(len(patient_results)), format_func=lambda i: trial_labels[i])
    render_result(TrialMatchResult.model_validate(patient_results[tidx]))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

patient_data = st.session_state.get("selected_patient")
trial_data = st.session_state.get("selected_trial")

# Status sidebar
with st.sidebar:
    st.markdown("### Session State")
    if patient_data:
        st.success(f"Patient: **{patient_data.get('patient_id', '?')}**")
    else:
        st.warning("No patient selected")
        st.markdown("← Go to **Patient Profile**")
    if trial_data:
        st.success(f"Trial: **{trial_data.get('nct_id', '?')}**")
    else:
        st.warning("No trial selected")
        st.markdown("← Go to **Trial Search**")

    if not settings.llm_configured:
        st.error("No LLM configured — add `TRIAL_MATCHER_GROQ_API_KEY` to secrets.")

if not patient_data:
    st.info("👈 First select a patient on the **Patient Profile** page.")
    st.stop()

if not trial_data:
    st.info("👈 Then select a trial on the **Trial Search** page.")
    st.stop()

if not settings.llm_configured:
    st.error(
        "No LLM API key configured. Add your Groq API key to Streamlit secrets:\n\n"
        "```toml\nTRIAL_MATCHER_GROQ_API_KEY = \"gsk_...\"\n```"
    )
    st.stop()

# Show current selection
col1, col2 = st.columns(2)
col1.info(f"**Patient**: {patient_data.get('patient_id')} | Age {patient_data.get('age', '?')} | {patient_data.get('sex', '?')}")
col2.info(f"**Trial**: {trial_data.get('nct_id')} — {trial_data.get('brief_title', '')[:60]}")

# Run matching
if st.button("🚀 Run Matching Engine", type="primary", use_container_width=False):
    from trial_matcher.criteria_parser.extractor import CriteriaExtractor
    from trial_matcher.matcher.engine import MatchingEngine
    from trial_matcher.schemas.patient import PatientProfile

    patient = PatientProfile.model_validate(patient_data)

    progress = st.progress(0, text="Parsing eligibility criteria…")

    with st.spinner(""):
        extractor = CriteriaExtractor(settings)
        criteria_set = extractor.extract(
            trial_data["eligibility_criteria"],
            trial_data["nct_id"],
        )

    progress.progress(60, text=f"Parsed {len(criteria_set.inclusion)} inclusion + {len(criteria_set.exclusion)} exclusion criteria. Running matching engine…")

    engine = MatchingEngine(settings)
    result = engine.match(patient, criteria_set, trial_data.get("brief_title", ""))

    progress.progress(100, text="Done!")
    st.session_state["match_result"] = result.model_dump(mode="json")

elif st.session_state.get("match_result"):
    result = TrialMatchResult.model_validate(st.session_state["match_result"])
    st.divider()
    render_result(result)
    st.stop()

if st.session_state.get("match_result"):
    result = TrialMatchResult.model_validate(st.session_state["match_result"])
    st.divider()
    render_result(result)
