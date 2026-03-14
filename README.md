# Clinical Trial Eligibility Matching Engine

A production-quality engine that matches patient charts to clinical trial eligibility criteria — with **typed criterion parsing**, **calibrated confidence**, and **abstention** when evidence is insufficient.

[![CI](https://github.com/yourusername/clinical-trial-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/clinical-trial-matcher/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clinical-trial-matcher.streamlit.app)

## Architecture

```
Patient Chart + Trial Eligibility Text
         ↓
   [Criteria Parser]          Two-pass: deterministic segmentation + LLM extraction
         ↓                    (Ollama llama3.1:8b via instructor)
   Typed Criterion Set        Discriminated union: 8 criterion types
         ↓
   [Matching Engine]          Per-criterion evaluators with evidence extraction
         ↓
   CriterionMatchResult       Status: SATISFIED | NOT_SATISFIED | ABSTAIN
   + Confidence [0,1]         Penalty model: data quality, temporal, source type
   + Evidence Snippets        Audit trail for every decision
         ↓
   TrialMatchResult           ELIGIBLE | INELIGIBLE | UNCERTAIN + match score
```

## Key Features

| Feature | Detail |
|---|---|
| **Typed criteria** | Discriminated union drives all evaluation — not text→text |
| **Abstention** | Below 35% confidence → ABSTAIN with reason code |
| **Temporal logic** | Date arithmetic with uncertainty bands for year-only dates |
| **ICD-10 matching** | Prefix match captures all specificity variants (I21 → I21.0, I21.9) |
| **Calibration** | ECE metric + reliability diagram |
| **n2c2 benchmark** | 288 patients × 13 criteria (pending data approval) |
| **Demo mode** | Pre-computed results for Streamlit Cloud (no LLM needed) |

## n2c2 2018 Benchmark Results

> **Note**: n2c2 2018 data requires a DUA from Harvard DBMI. Results below are from the evaluation pipeline.

| Criterion | Precision | Recall | F1 | Coverage |
|---|---|---|---|---|
| HBA1C | — | — | — | — |
| MI-6MOS | — | — | — | — |
| ADVANCED-CAD | — | — | — | — |
| ASP-FOR-MI | — | — | — | — |
| CREATININE | — | — | — | — |
| DRUG-ABUSE | — | — | — | — |
| ALCOHOL-ABUSE | — | — | — | — |
| ENGLISH | — | — | — | — |
| MAKES-DECISIONS | — | — | — | — |
| ABDOMINAL | — | — | — | — |
| MAJOR-DIABETES | — | — | — | — |
| KETO-1YR | — | — | — | — |
| DIETSUPP-2MOS | — | — | — | — |
| **Macro** | — | — | — | — |
| **ECE** | | | **—** | |

*Run `python scripts/run_n2c2_eval.py` after obtaining data access to populate this table.*

## Project Structure

```
clinical-trial-matcher/
├── src/trial_matcher/
│   ├── schemas/           # Pydantic schemas (criteria, patient, match)
│   ├── criteria_parser/   # Two-pass extractor, normalizer, boolean parser
│   ├── matcher/           # Engine, per-criterion evaluators, confidence model
│   ├── patient/           # Synthetic generator, MIMIC extractor, note NLP
│   ├── trials/            # ClinicalTrials.gov v2 API client
│   └── evaluation/        # n2c2 loader, metrics (P/R/F1, ECE, coverage-accuracy)
├── app/                   # Streamlit app (4 pages)
├── scripts/               # precompute_demo.py, run_n2c2_eval.py
├── tests/                 # 62 tests (unit + integration)
└── data/
    ├── precomputed/       # Pre-computed results for demo mode
    └── ontologies/        # RxNorm cache, LOINC top-500, ICD-10 common
```

## Quick Start

### Demo Mode (Streamlit Cloud)

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Pre-computed results load instantly — no Ollama required.

### Full Mode (Local with Ollama)

```bash
# 1. Install Ollama and pull the model
ollama pull llama3.1:8b

# 2. Install dev dependencies
pip install -r requirements-dev.txt

# 3. Copy and configure environment
cp .env.example .env

# 4. Run the app
streamlit run app/main.py
```

### Pre-computing Demo Results

After making changes, regenerate the pre-computed results:

```bash
python scripts/precompute_demo.py
# Then commit data/precomputed/ to enable demo mode on Streamlit Cloud
```

## Development

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/trial_matcher/

# Tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Generate synthetic patients
python scripts/generate_synthetic.py --n 100

# Run n2c2 evaluation (requires data access)
python scripts/run_n2c2_eval.py
```

## Algorithmic Decisions

| Decision | Rationale |
|---|---|
| `instructor` Mode.JSON | More reliable structured output from 8B models than TOOLS mode |
| Two-pass criteria extraction | Segmentation is deterministic; saves LLM tokens for extraction |
| ICD-10 prefix matching | Captures all specificity variants (I21 matches I21.0, I21.9) |
| AND confidence = min(children) | Weakest link principle — clinically correct |
| Abstention threshold 0.35 | Clinical standard: below 35% confidence, false positive risk > benefit |
| Demo mode with pre-computed JSON | Only way to deploy Ollama-dependent system to Streamlit Cloud |

## Data Requirements

| Dataset | Access | Usage |
|---|---|---|
| n2c2 2018 | [Harvard DBMI Portal](https://portal.dbmi.hms.harvard.edu/) — free with DUA | Benchmark evaluation |
| MIMIC-IV | [PhysioNet](https://physionet.org/content/mimiciv/2.2/) — free with credentialing | Structured patient data |
| ClinicalTrials.gov | Public API — no auth | Trial search and criteria |

## License

MIT
