"""Clinical Trial Matcher — Streamlit app entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from trial_matcher.config import get_settings

settings = get_settings()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 Clinical Trial Matcher")
    st.divider()

    if settings.demo_mode:
        st.warning("**Demo Mode** — pre-computed results", icon="ℹ️")
    elif settings.llm_configured:
        backend = settings.llm_backend.upper()
        model = settings.groq_model if settings.llm_backend == "groq" else settings.ollama_model
        st.success(f"**Live Mode** — {backend}", icon="✅")
        st.caption(f"Model: `{model}`")
    else:
        st.error("**No LLM key** — add `TRIAL_MATCHER_GROQ_API_KEY`", icon="🔴")

    st.divider()
    st.markdown("""
**Pages**
1. 👤 Patient Profile
2. 🔍 Trial Search
3. ✅ Match Results
4. 📊 Evaluation Dashboard
""")
    st.divider()
    st.caption("[GitHub](https://github.com/Kush-Meta/clinical-trial-matcher)")

# ── Home ──────────────────────────────────────────────────────────────────────

st.title("🏥 Clinical Trial Eligibility Matching Engine")
st.markdown(
    "Match patient charts to clinical trial eligibility criteria — "
    "with **typed criterion parsing**, **calibrated confidence**, and **abstention** "
    "when evidence is insufficient."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Criterion types", "8", help="LAB_VALUE · DIAGNOSIS · MEDICATION · BEHAVIORAL · DEMOGRAPHIC · FUNCTIONAL · PROCEDURE · COMPOSITE")
c2.metric("n2c2 benchmark", "13 criteria", help="288 patients × 13 binary criteria")
c3.metric("Abstention threshold", f"{int(settings.abstention_confidence_threshold*100)}%")
c4.metric("LLM backend", settings.llm_backend.upper() if not settings.demo_mode else "DEMO")

st.divider()

if not settings.demo_mode and not settings.llm_configured:
    st.error(
        "### ⚙️ Setup Required\n\n"
        "Add your Groq API key to Streamlit secrets to enable live matching:\n\n"
        "```toml\n"
        "TRIAL_MATCHER_GROQ_API_KEY = \"gsk_...\"\n"
        "TRIAL_MATCHER_LLM_BACKEND = \"groq\"\n"
        "```\n\n"
        "Get a **free** key at [console.groq.com](https://console.groq.com) — no credit card needed.",
        icon="🔑",
    )
else:
    mode_label = "Demo" if settings.demo_mode else "Live"
    st.success(f"**{mode_label} mode active.** Use the sidebar to navigate.", icon="✅")

st.markdown("""
## How It Works

```
Patient Chart  +  Trial Eligibility Text
        ↓
  [Two-Pass Parser]      Deterministic segmentation → LLM per-criterion extraction
        ↓                (Groq llama-3.1-8b-instant — free, fast)
  Typed Criteria         LabValueCriterion · DiagnosisCriterion · MedicationCriterion …
        ↓
  [Matching Engine]      Per-criterion evaluators + evidence extraction
        ↓
  CriterionMatchResult   SATISFIED | NOT_SATISFIED | ABSTAIN
  + Confidence [0–1]     Penalty model: source, age, temporal, unit mismatch
  + Evidence Snippets    Audit trail for every decision
        ↓
  TrialMatchResult       ELIGIBLE | INELIGIBLE | UNCERTAIN + ranked match score
```

## Key Design Choices

| Decision | Why |
|---|---|
| ICD-10 prefix match | `I21` matches `I21.0`, `I21.9` — captures all specificity variants |
| AND confidence = min(children) | Weakest-link principle — clinically correct |
| Abstention at <35% confidence | Below this threshold, false-positive risk > benefit |
| Groq (free) for cloud | Same llama-3.1-8b as Ollama, OpenAI-compatible, 14k req/day free |
| Pre-computed demo data | Enables Streamlit Cloud deploy without any API key |
""")
