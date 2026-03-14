"""Evaluation Dashboard — n2c2 benchmark, calibration, coverage-accuracy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trial_matcher.config import get_settings

st.set_page_config(page_title="Evaluation Dashboard — Clinical Trial Matcher", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption("n2c2 2018 Clinical Trial Eligibility Benchmark")

settings = get_settings()


@st.cache_data
def load_eval_results() -> dict | None:
    path = Path("results/n2c2_eval_latest.json")
    if path.exists():
        return json.loads(path.read_text())
    return None


eval_results = load_eval_results()

if eval_results is None:
    st.warning(
        "No evaluation results found. Run `scripts/run_n2c2_eval.py` after "
        "obtaining n2c2 2018 data access from Harvard DBMI.",
        icon="⚠️",
    )

    # Show placeholder benchmark table
    st.markdown("### Expected Output Format (Sample Data)")
    placeholder_data = {
        "Criterion": [
            "DRUG-ABUSE", "ALCOHOL-ABUSE", "ENGLISH", "MAKES-DECISIONS",
            "ABDOMINAL", "MAJOR-DIABETES", "ADVANCED-CAD", "MI-6MOS",
            "KETO-1YR", "DIETSUPP-2MOS", "ASP-FOR-MI", "HBA1C", "CREATININE",
        ],
        "Precision": [0.82, 0.79, 0.95, 0.88, 0.76, 0.89, 0.81, 0.72, 0.65, 0.74, 0.83, 0.91, 0.87],
        "Recall":    [0.71, 0.68, 0.93, 0.84, 0.69, 0.92, 0.75, 0.61, 0.58, 0.67, 0.79, 0.94, 0.83],
        "F1":        [0.76, 0.73, 0.94, 0.86, 0.72, 0.90, 0.78, 0.66, 0.61, 0.70, 0.81, 0.92, 0.85],
        "Coverage":  [0.85, 0.81, 0.98, 0.95, 0.80, 0.97, 0.88, 0.75, 0.70, 0.78, 0.90, 0.99, 0.93],
        "Abstention": [0.15, 0.19, 0.02, 0.05, 0.20, 0.03, 0.12, 0.25, 0.30, 0.22, 0.10, 0.01, 0.07],
    }
    df = pd.DataFrame(placeholder_data)
    macro_f1 = round(sum(df["F1"]) / len(df["F1"]), 3)
    st.info(f"**Placeholder data** — Macro F1: {macro_f1:.3f} | ECE: 0.062 | Coverage: 86%")
    eval_results = {
        "criterion_metrics": df.to_dict("records"),
        "macro_f1": macro_f1,
        "macro_precision": 0.820,
        "macro_recall": 0.769,
        "ece": 0.062,
        "overall_coverage": 0.862,
        "n_patients": 288,
        "n_pairs": 3744,
    }

# ── Top-level metrics ─────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Macro F1", f"{eval_results['macro_f1']:.3f}")
with col2:
    st.metric("Macro Precision", f"{eval_results['macro_precision']:.3f}")
with col3:
    st.metric("Macro Recall", f"{eval_results['macro_recall']:.3f}")
with col4:
    st.metric("ECE", f"{eval_results['ece']:.3f}", help="Expected Calibration Error (lower = better)")
with col5:
    coverage_pct = int(eval_results["overall_coverage"] * 100)
    st.metric("Coverage", f"{coverage_pct}%", help="Fraction of criteria where system makes a prediction")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Per-Criterion F1",
    "📈 Calibration Curve",
    "📉 Coverage-Accuracy",
    "🗂️ Error Taxonomy",
])

with tab1:
    st.subheader("Per-Criterion Performance")
    metrics_data = eval_results["criterion_metrics"]
    if isinstance(metrics_data[0], dict):
        df = pd.DataFrame(metrics_data)
        if "criterion" in df.columns:
            df = df.rename(columns={"criterion": "Criterion"})
        # Round numeric columns
        for col in ["precision", "recall", "f1", "coverage", "abstention_rate"]:
            if col in df.columns:
                df[col] = df[col].round(3)

        # Style F1 with color
        styled = df.style.background_gradient(
            subset=["f1"] if "f1" in df.columns else [],
            cmap="RdYlGn",
            vmin=0.5, vmax=1.0,
        )
        st.dataframe(styled, use_container_width=True)

        # Bar chart
        fig = px.bar(
            df,
            x="Criterion" if "Criterion" in df.columns else df.columns[0],
            y=["precision", "recall", "f1"] if all(c in df.columns for c in ["precision", "recall", "f1"]) else [df.columns[1]],
            barmode="group",
            title="Per-Criterion Precision, Recall, F1",
            color_discrete_map={"precision": "#2196F3", "recall": "#FF9800", "f1": "#4CAF50"},
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Reliability Diagram (Calibration)")
    st.caption("Perfectly calibrated = diagonal line. Points above = underconfident, below = overconfident.")

    calibration_path = Path("results/calibration_data.json")
    if calibration_path.exists():
        cal_data = json.loads(calibration_path.read_text())
    else:
        # Placeholder calibration curve (slightly overconfident in mid-range)
        import numpy as np
        np.random.seed(42)
        cal_data = {
            "bin_centers": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "mean_confidence": [0.10, 0.21, 0.31, 0.41, 0.51, 0.60, 0.71, 0.81, 0.90],
            "accuracy": [0.08, 0.18, 0.27, 0.38, 0.48, 0.57, 0.69, 0.79, 0.88],
            "counts": [45, 89, 120, 200, 280, 340, 290, 180, 95],
        }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=cal_data["mean_confidence"],
        y=cal_data["accuracy"],
        mode="lines+markers",
        name="Model",
        marker=dict(size=[max(5, c // 20) for c in cal_data["counts"]], color="#2196F3"),
        line=dict(color="#2196F3"),
        hovertemplate="Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Reliability Diagram (ECE = {eval_results['ece']:.3f})",
        xaxis_title="Mean Confidence",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Coverage-Accuracy Tradeoff")
    st.caption(
        "Abstaining on uncertain cases improves accuracy at the cost of coverage. "
        "Use the slider to explore the tradeoff."
    )

    threshold = st.slider(
        "Abstention Confidence Threshold",
        min_value=0.0, max_value=1.0,
        value=settings.abstention_confidence_threshold,
        step=0.05,
        format="%.2f",
        help="Predictions below this confidence are abstained",
    )

    coverage_path = Path("results/coverage_accuracy_curve.json")
    if coverage_path.exists():
        curve_data = json.loads(coverage_path.read_text())
    else:
        # Placeholder curve
        thresholds = [i / 20 for i in range(21)]
        coverage = [1.0 - 0.5 * t for t in thresholds]
        accuracy = [0.75 + 0.20 * t for t in thresholds]
        curve_data = {"thresholds": thresholds, "coverage": coverage, "accuracy": accuracy}

    df_curve = pd.DataFrame(curve_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curve["coverage"],
        y=df_curve["accuracy"],
        mode="lines+markers",
        name="Coverage-Accuracy",
        line=dict(color="#4CAF50"),
        hovertemplate="Coverage: %{x:.0%}<br>Accuracy: %{y:.0%}<extra></extra>",
    ))

    # Highlight selected threshold
    if "thresholds" in df_curve.columns:
        idx = (df_curve["thresholds"] - threshold).abs().idxmin()
        fig.add_trace(go.Scatter(
            x=[df_curve.loc[idx, "coverage"]],
            y=[df_curve.loc[idx, "accuracy"]],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name=f"Threshold = {threshold:.2f}",
        ))

    fig.update_layout(
        title="Coverage-Accuracy Curve",
        xaxis_title="Coverage (fraction of cases predicted)",
        yaxis_title="Accuracy (on predicted cases)",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0.5, 1.0]),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Error Taxonomy")
    st.caption("Where does the system fail?")

    taxonomy_path = Path("results/error_taxonomy.json")
    if taxonomy_path.exists():
        taxonomy_data = json.loads(taxonomy_path.read_text())
    else:
        taxonomy_data = {
            "CORRECT": 2890,
            "ABSTAINED": 420,
            "CRITERIA_PARSING_ERROR": 85,
            "TEMPORAL_FAILURE": 120,
            "NEGATION_ERROR": 95,
            "PATIENT_DATA_GAP": 65,
            "THRESHOLD_ERROR": 45,
            "COMPOSITE_LOGIC_ERROR": 24,
        }

    errors_only = {k: v for k, v in taxonomy_data.items()
                   if k not in ("CORRECT", "ABSTAINED")}
    total_errors = sum(errors_only.values())

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            names=list(errors_only.keys()),
            values=list(errors_only.values()),
            title=f"Error Breakdown (n={total_errors})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Error Category Descriptions**")
        descriptions = {
            "CRITERIA_PARSING_ERROR": "Parser flagged criterion as ambiguous",
            "TEMPORAL_FAILURE": "Date/time window evaluation uncertain",
            "NEGATION_ERROR": "Negation detection failure in notes",
            "PATIENT_DATA_GAP": "Required data missing from patient record",
            "THRESHOLD_ERROR": "Numeric boundary evaluated incorrectly",
            "COMPOSITE_LOGIC_ERROR": "AND/OR/AT_LEAST_N logic error",
        }
        for cat, desc in descriptions.items():
            count = errors_only.get(cat, 0)
            pct = count / total_errors * 100 if total_errors else 0
            st.markdown(f"**{cat}** ({count}, {pct:.1f}%): {desc}")

    # Overall stats
    st.divider()
    total = sum(taxonomy_data.values())
    correct = taxonomy_data.get("CORRECT", 0)
    abstained = taxonomy_data.get("ABSTAINED", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct", f"{correct} ({correct/total:.0%})")
    with col2:
        st.metric("Abstained", f"{abstained} ({abstained/total:.0%})")
    with col3:
        st.metric("Errors", f"{total_errors} ({total_errors/total:.0%})")
