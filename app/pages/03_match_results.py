"""Match Results page — per-criterion audit trail."""

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
st.title("✅ Match Results — Per-Criterion Audit Trail")

settings = get_settings()

STATUS_COLOR = {
    "SATISFIED": "🟢",
    "NOT_SATISFIED": "🔴",
    "ABSTAIN": "🟡",
}

STATUS_BADGE = {
    "SATISFIED": "✅ SATISFIED",
    "NOT_SATISFIED": "❌ NOT SATISFIED",
    "ABSTAIN": "⚠️ ABSTAIN",
}


@st.cache_data
def load_precomputed_results() -> list[dict]:
    path = Path("data/precomputed/demo_match_results.json")
    if path.exists():
        return json.loads(path.read_text())
    return []


def confidence_bar(confidence: float) -> str:
    filled = int(confidence * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"`{bar}` {confidence:.0%}"


def render_criterion_row(result: CriterionMatchResult, is_exclusion: bool = False) -> None:
    """Render a single criterion result as an expander row."""
    status_icon = STATUS_COLOR.get(result.status.value, "❓")
    label_prefix = "EXCLUDE" if is_exclusion else "INCLUDE"

    with st.expander(
        f"{status_icon} [{label_prefix}] {result.criterion_summary[:100]}",
        expanded=(result.status == MatchStatus.NOT_SATISFIED),
    ):
        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            st.caption("Status")
            badge = STATUS_BADGE.get(result.status.value, result.status.value)
            st.markdown(f"**{badge}**")
            if result.abstention_reason:
                st.caption(f"Reason: `{result.abstention_reason.value}`")

        with col2:
            st.caption("Confidence")
            st.progress(result.confidence, text=f"{result.confidence:.0%} ({result.confidence_label})")

        with col3:
            st.caption("Criterion Type")
            st.code(result.criterion_type.value)

        if result.raw_criterion_text:
            st.caption("**Raw Criterion Text**")
            st.text(result.raw_criterion_text[:500])

        if result.evidence:
            st.caption("**Evidence**")
            for ev in result.evidence:
                date_str = f" ({ev.source_date})" if ev.source_date else ""
                st.success(f"[{ev.source_type.upper()}{date_str}] {ev.text}")

        if result.sub_results:
            st.caption("**Sub-criteria**")
            for sub in result.sub_results:
                sub_icon = STATUS_COLOR.get(sub.status.value, "❓")
                st.write(f"  {sub_icon} {sub.criterion_summary[:80]} — {sub.confidence:.0%}")


def render_trial_result(result: TrialMatchResult) -> None:
    """Render a complete trial match result."""
    # Overall status header
    eligibility_color = {"ELIGIBLE": "🟢", "INELIGIBLE": "🔴", "UNCERTAIN": "🟡"}
    st.markdown(
        f"## {eligibility_color.get(result.overall_eligibility, '❓')} "
        f"**{result.overall_eligibility}** for {result.nct_id}"
    )
    st.markdown(f"*{result.trial_title}*")

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Overall Confidence", f"{result.overall_confidence:.0%}")
    with col2:
        st.metric("Inclusion Satisfied", f"{result.n_inclusion_satisfied}/{result.n_inclusion_total}")
    with col3:
        st.metric("Abstained", result.n_inclusion_abstain, help="Criteria with insufficient evidence")
    with col4:
        st.metric("Exclusions Triggered", result.n_exclusion_triggered)
    with col5:
        st.metric("Match Score", f"{result.match_score:.2f}")

    if result.blocking_criterion_summary:
        st.error(f"**Blocking Criterion**: {result.blocking_criterion_summary}")

    st.divider()

    # Inclusion criteria
    if result.inclusion_results:
        st.subheader(f"Inclusion Criteria ({result.n_inclusion_total} total)")
        for r in result.inclusion_results:
            render_criterion_row(r, is_exclusion=False)

    # Exclusion criteria
    if result.exclusion_results:
        st.divider()
        st.subheader(f"Exclusion Criteria ({result.n_exclusion_total} total)")
        for r in result.exclusion_results:
            render_criterion_row(r, is_exclusion=True)

    # Ambiguity panel
    ambiguous_results = [
        r for r in result.get_all_results()
        if r.abstention_reason is not None
    ]
    if ambiguous_results:
        st.divider()
        st.subheader("⚠️ Ambiguity Panel — Show Your Work")
        st.caption(
            "These criteria had insufficient evidence or parsing uncertainty. "
            "Abstention prevents false positives in clinical contexts."
        )
        ambig_data = [
            {
                "Criterion": r.criterion_summary[:80],
                "Type": r.criterion_type.value,
                "Abstention Reason": r.abstention_reason.value if r.abstention_reason else "—",
                "Confidence": f"{r.confidence:.0%}",
                "Impact": "Blocked outcome" if r.is_blocking else "Uncertain eligibility",
            }
            for r in ambiguous_results
        ]
        st.dataframe(pd.DataFrame(ambig_data), use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────

precomputed = load_precomputed_results()

if settings.demo_mode or not st.session_state.get("selected_patient"):
    # Demo mode: show pre-computed results
    if not precomputed:
        st.error("No pre-computed results found. Run `scripts/precompute_demo.py`.")
        st.stop()

    st.info("**Demo Mode**: Showing pre-computed match results.", icon="ℹ️")

    # Patient selector
    result_dicts = precomputed
    patient_ids = list({r["patient_id"] for r in result_dicts})
    selected_patient_id = st.selectbox("Select Patient", patient_ids)

    patient_results = [r for r in result_dicts if r["patient_id"] == selected_patient_id]

    if not patient_results:
        st.warning("No results for this patient.")
        st.stop()

    trial_labels = [f"{r['nct_id']} — {r['trial_title'][:60]}" for r in patient_results]
    selected_trial_idx = st.selectbox("Select Trial", range(len(patient_results)),
                                       format_func=lambda i: trial_labels[i])

    result_dict = patient_results[selected_trial_idx]
    result = TrialMatchResult.model_validate(result_dict)
    st.divider()
    render_trial_result(result)

else:
    # Full mode: run live matching
    patient_data = st.session_state.get("selected_patient")
    trial_data = st.session_state.get("selected_trial")

    if not patient_data:
        st.warning("Please select a patient on the Patient Profile page first.")
        st.stop()
    if not trial_data:
        st.warning("Please select a trial on the Trial Search page first.")
        st.stop()

    if st.button("🚀 Run Matching", type="primary"):
        from trial_matcher.criteria_parser.extractor import CriteriaExtractor
        from trial_matcher.matcher.engine import MatchingEngine
        from trial_matcher.schemas.patient import PatientProfile

        patient = PatientProfile.model_validate(patient_data)

        with st.spinner("Parsing criteria..."):
            extractor = CriteriaExtractor(settings)
            criteria_set = extractor.extract(
                trial_data["eligibility_criteria"],
                trial_data["nct_id"],
            )

        with st.spinner("Running matching engine..."):
            engine = MatchingEngine(settings)
            result = engine.match(patient, criteria_set, trial_data["brief_title"])

        st.session_state["match_result"] = result.model_dump(mode="json")
        st.divider()
        render_trial_result(result)
    elif st.session_state.get("match_result"):
        result = TrialMatchResult.model_validate(st.session_state["match_result"])
        render_trial_result(result)
    else:
        st.info("Select a patient and trial, then click **Run Matching**.")
