#!/usr/bin/env python3
"""
Pre-compute demo results for Streamlit Cloud deployment.

Generates:
  data/precomputed/demo_patients.json
  data/precomputed/demo_trials.json
  data/precomputed/demo_match_results.json

Run locally before deploying:
  python scripts/precompute_demo.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from trial_matcher.config import get_settings
    from trial_matcher.criteria_parser.extractor import CriteriaExtractor
    from trial_matcher.matcher.engine import MatchingEngine
    from trial_matcher.patient.synthetic_generator import generate_demo_patients
    from trial_matcher.trials.ctgov_client import CTGovClient

    settings = get_settings()
    if settings.demo_mode:
        logger.error("Cannot precompute in demo mode — set TRIAL_MATCHER_DEMO_MODE=false")
        sys.exit(1)

    out_dir = Path("data/precomputed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic patients
    logger.info("Generating 5 demo patients...")
    patients = generate_demo_patients()
    patient_dicts = [p.model_dump(mode="json") for p in patients]
    (out_dir / "demo_patients.json").write_text(
        json.dumps(patient_dicts, indent=2, default=str)
    )
    logger.info("Wrote %d patients", len(patients))

    # 2. Fetch 10 CT.gov trials
    logger.info("Fetching 10 trials from ClinicalTrials.gov...")
    trial_dicts = []
    search_terms = [
        "type 2 diabetes mellitus cardiac",
        "coronary artery disease HbA1c",
    ]
    with CTGovClient() as client:
        for term in search_terms:
            trials = client.search(term, max_results=5)
            for t in trials:
                if t.eligibility_criteria and len(trial_dicts) < 10:
                    trial_dicts.append({
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
                    })
            if len(trial_dicts) >= 10:
                break

    (out_dir / "demo_trials.json").write_text(
        json.dumps(trial_dicts, indent=2, default=str)
    )
    logger.info("Wrote %d trials", len(trial_dicts))

    if not trial_dicts:
        logger.error("No trials fetched — check network connection")
        sys.exit(1)

    # 3. Parse criteria and run matching
    logger.info("Parsing criteria and running matching engine...")
    extractor = CriteriaExtractor(settings)
    engine = MatchingEngine(settings)
    match_results = []

    for patient in patients:
        for trial_dict in trial_dicts[:5]:  # 5 trials per patient = 25 results
            try:
                logger.info("Matching %s vs %s", patient.patient_id, trial_dict["nct_id"])
                criteria_set = extractor.extract(
                    trial_dict["eligibility_criteria"],
                    trial_dict["nct_id"],
                )
                result = engine.match(patient, criteria_set, trial_dict["brief_title"])
                match_results.append(result.model_dump(mode="json"))
            except Exception as e:
                logger.warning("Failed to match %s vs %s: %s",
                               patient.patient_id, trial_dict["nct_id"], e)

    (out_dir / "demo_match_results.json").write_text(
        json.dumps(match_results, indent=2, default=str)
    )
    logger.info("Wrote %d match results", len(match_results))
    logger.info("Done! Commit data/precomputed/ to enable demo mode.")


if __name__ == "__main__":
    main()
