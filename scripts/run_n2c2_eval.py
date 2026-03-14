#!/usr/bin/env python3
"""
Run n2c2 2018 benchmark evaluation.

Requires:
  - n2c2 2018 data at data/raw/n2c2_2018/ (Harvard DBMI DUA)
  - Local Ollama running

Usage:
  python scripts/run_n2c2_eval.py [--split train] [--max-patients 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Map n2c2 criteria names to criteria text for parsing
N2C2_CRITERIA_TEXTS = {
    "DRUG-ABUSE": "Inclusion: Patient has a history of drug abuse",
    "ALCOHOL-ABUSE": "Inclusion: Patient has a history of alcohol abuse",
    "ENGLISH": "Inclusion: Patient speaks English",
    "MAKES-DECISIONS": "Inclusion: Patient is able to make their own informed decisions",
    "ABDOMINAL": "Inclusion: Patient has undergone abdominal surgery within the past 5 years",
    "MAJOR-DIABETES": "Inclusion: Patient has a major diabetes diagnosis (Type 1 or Type 2)",
    "ADVANCED-CAD": (
        "Inclusion: Patient has advanced coronary artery disease, defined as at least 2 of: "
        "history of MI, coronary artery bypass surgery, percutaneous coronary intervention, "
        "or catheterization showing >50% stenosis"
    ),
    "MI-6MOS": "Inclusion: Patient has had a myocardial infarction within the past 6 months",
    "KETO-1YR": "Inclusion: Patient has had ketoacidosis within the past 1 year",
    "DIETSUPP-2MOS": "Inclusion: Patient has used dietary supplements within the past 2 months",
    "ASP-FOR-MI": "Inclusion: Patient takes aspirin 81mg daily for MI prevention",
    "HBA1C": "Inclusion: Patient HbA1c is between 6.5% and 9.5%",
    "CREATININE": "Inclusion: Patient creatinine is above the upper limit of normal",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run n2c2 2018 evaluation")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    from trial_matcher.config import get_settings
    from trial_matcher.criteria_parser.extractor import CriteriaExtractor
    from trial_matcher.evaluation.error_taxonomy import build_error_taxonomy, taxonomy_summary
    from trial_matcher.evaluation.metrics import evaluate_batch, generate_reliability_diagram_data
    from trial_matcher.evaluation.n2c2_loader import N2C2Loader
    from trial_matcher.matcher.engine import MatchingEngine
    from trial_matcher.patient.note_nlp import NoteNLPPipeline
    from trial_matcher.schemas.match import MatchStatus

    settings = get_settings()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load n2c2 data
    loader = N2C2Loader(settings.n2c2_data_dir)
    try:
        dataset = loader.load(split=args.split)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Loaded %d patients from n2c2", dataset.n_patients)

    patients = dataset.patients
    if args.max_patients:
        patients = patients[: args.max_patients]

    # Convert to PatientProfile with NLP
    nlp_pipeline = NoteNLPPipeline()
    patient_profiles = loader.to_patient_profiles(
        type("DS", (), {"patients": patients})()  # type: ignore
    )

    # Run NLP on each patient
    for profile in patient_profiles:
        processed_notes = [nlp_pipeline.process_note(n) for n in profile.notes]
        profile.notes = processed_notes

    # Parse criteria for all n2c2 criteria
    extractor = CriteriaExtractor(settings)
    engine = MatchingEngine(settings)

    # Collect predictions and labels per criterion
    all_labels: dict[str, list[int]] = {c: [] for c in N2C2_CRITERIA_TEXTS}
    all_predictions: dict[str, list[int]] = {c: [] for c in N2C2_CRITERIA_TEXTS}
    all_confidences: dict[str, list[float]] = {c: [] for c in N2C2_CRITERIA_TEXTS}

    from trial_matcher.criteria_parser.extractor import CriteriaExtractor
    from trial_matcher.schemas.criteria import EligibilityCriteriaSet

    for n2c2_patient, profile in zip(patients, patient_profiles):
        for criterion_name, criterion_text in N2C2_CRITERIA_TEXTS.items():
            ground_truth = n2c2_patient.labels.get(criterion_name, -1)
            if ground_truth == -1:
                continue

            # Parse single criterion
            criteria_set = extractor.extract(criterion_text, nct_id="N2C2")
            all_criteria = criteria_set.inclusion + criteria_set.exclusion
            if not all_criteria:
                all_predictions[criterion_name].append(-1)
                all_confidences[criterion_name].append(0.0)
                all_labels[criterion_name].append(ground_truth)
                continue

            result = engine.match(profile, criteria_set, criterion_name)
            all_results = result.inclusion_results + result.exclusion_results

            if not all_results:
                pred = -1
                conf = 0.0
            else:
                r = all_results[0]
                if r.status == MatchStatus.SATISFIED:
                    pred = 1
                elif r.status == MatchStatus.NOT_SATISFIED:
                    pred = 0
                else:
                    pred = -1
                conf = r.confidence

            all_labels[criterion_name].append(ground_truth)
            all_predictions[criterion_name].append(pred)
            all_confidences[criterion_name].append(conf)

            logger.debug("%s | %s: gt=%d pred=%d conf=%.2f",
                         profile.patient_id, criterion_name, ground_truth, pred, conf)

    # Compute metrics
    from trial_matcher.evaluation.metrics import evaluate_batch
    eval_result = evaluate_batch(all_labels, all_predictions, all_confidences)

    logger.info("Macro F1: %.3f | ECE: %.3f | Coverage: %.1f%%",
                eval_result.macro_f1, eval_result.ece,
                eval_result.overall_coverage * 100)

    # Print per-criterion table
    print("\nPer-Criterion Results:")
    print(f"{'Criterion':<20} {'P':>6} {'R':>6} {'F1':>6} {'Cov':>6} {'Abstain':>8}")
    print("-" * 60)
    for m in eval_result.criterion_metrics:
        print(f"{m.criterion:<20} {m.precision:>6.3f} {m.recall:>6.3f} "
              f"{m.f1:>6.3f} {m.coverage:>6.2%} {m.abstention_rate:>8.2%}")
    print(f"\nMacro F1: {eval_result.macro_f1:.3f}")
    print(f"ECE: {eval_result.ece:.3f}")
    print(f"Coverage: {eval_result.overall_coverage:.1%}")

    # Save results
    result_dict = {
        "criterion_metrics": [
            {
                "criterion": m.criterion,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "accuracy": m.accuracy,
                "coverage": m.coverage,
                "abstention_rate": m.abstention_rate,
                "support": m.support,
            }
            for m in eval_result.criterion_metrics
        ],
        "macro_f1": eval_result.macro_f1,
        "macro_precision": eval_result.macro_precision,
        "macro_recall": eval_result.macro_recall,
        "ece": eval_result.ece,
        "overall_coverage": eval_result.overall_coverage,
        "n_patients": len(patients),
        "n_pairs": eval_result.n_pairs,
    }
    (output_dir / "n2c2_eval_latest.json").write_text(json.dumps(result_dict, indent=2))

    # Save calibration data
    flat_labels = [l for labels in all_labels.values() for l in labels]
    flat_preds = [p for preds in all_predictions.values() for p in preds]
    flat_confs = [c for confs in all_confidences.values() for c in confs]
    cal_data = generate_reliability_diagram_data(flat_labels, flat_preds, flat_confs)
    (output_dir / "calibration_data.json").write_text(json.dumps(cal_data, indent=2))

    logger.info("Results written to %s/", output_dir)


if __name__ == "__main__":
    main()
