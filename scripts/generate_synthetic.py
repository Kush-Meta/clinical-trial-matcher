#!/usr/bin/env python3
"""Generate synthetic patient data for testing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic patients")
    parser.add_argument("--n", type=int, default=50, help="Number of patients to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data/synthetic/patients", help="Output directory")
    args = parser.parse_args()

    from trial_matcher.patient.synthetic_generator import SyntheticPatientGenerator

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d synthetic patients (seed=%d)...", args.n, args.seed)
    gen = SyntheticPatientGenerator(seed=args.seed)
    patients = gen.generate(args.n)

    # Write one file per patient
    for patient in patients:
        path = out_dir / f"{patient.patient_id}.json"
        path.write_text(patient.model_dump_json(indent=2))

    # Write combined file
    all_path = out_dir / "all_patients.json"
    all_path.write_text(
        json.dumps([p.model_dump(mode="json") for p in patients], indent=2, default=str)
    )

    logger.info("Wrote %d patients to %s/", len(patients), out_dir)


if __name__ == "__main__":
    main()
