"""n2c2 2018 shared task data loader."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

from trial_matcher.schemas.patient import ClinicalNote, PatientProfile

logger = logging.getLogger(__name__)

# n2c2 2018: 13 binary criteria
N2C2_CRITERIA = [
    "DRUG-ABUSE",
    "ALCOHOL-ABUSE",
    "ENGLISH",
    "MAKES-DECISIONS",
    "ABDOMINAL",
    "MAJOR-DIABETES",
    "ADVANCED-CAD",
    "MI-6MOS",
    "KETO-1YR",
    "DIETSUPP-2MOS",
    "ASP-FOR-MI",
    "HBA1C",
    "CREATININE",
]

N2C2_LABEL_MAP = {"met": 1, "not met": 0}


@dataclass
class N2C2Patient:
    patient_id: str
    notes: list[str]  # raw note texts
    labels: dict[str, int]  # criterion → 1/0 (-1 for unlabeled)
    raw_xml: str = ""


@dataclass
class N2C2Dataset:
    patients: list[N2C2Patient] = field(default_factory=list)
    n_criteria: int = 13

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    @property
    def n_pairs(self) -> int:
        return self.n_patients * self.n_criteria


class N2C2Loader:
    """Loads and parses the n2c2 2018 clinical trial eligibility dataset."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._note_dir = self.data_dir / "train"
        self._label_dir = self.data_dir / "train"

    def load(self, split: str = "train") -> N2C2Dataset:
        """Load n2c2 dataset from XML files."""
        note_dir = self.data_dir / split
        if not note_dir.exists():
            raise FileNotFoundError(
                f"n2c2 data not found at {note_dir}. "
                "Request access from Harvard DBMI: "
                "https://portal.dbmi.hms.harvard.edu/"
            )

        dataset = N2C2Dataset()
        note_files = sorted(note_dir.glob("*.xml"))
        label_files = {f.stem: f for f in note_dir.glob("*-tags.xml")}

        logger.info("Loading %d note files from %s", len(note_files), note_dir)

        for note_file in note_files:
            # Skip label files
            if note_file.stem.endswith("-tags"):
                continue

            patient_id = note_file.stem
            tag_file = label_files.get(f"{patient_id}-tags")

            notes = self._parse_notes(note_file)
            labels = self._parse_labels(tag_file) if tag_file else {}

            dataset.patients.append(N2C2Patient(
                patient_id=patient_id,
                notes=notes,
                labels=labels,
            ))

        logger.info("Loaded %d patients", len(dataset.patients))
        return dataset

    def _parse_notes(self, path: Path) -> list[str]:
        """Parse patient notes from XML."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            notes = []
            for text_elem in root.findall(".//TEXT"):
                if text_elem.text:
                    notes.append(text_elem.text.strip())
            return notes
        except ET.ParseError as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return []

    def _parse_labels(self, path: Path) -> dict[str, int]:
        """Parse ground-truth labels from XML tags file."""
        labels: dict[str, int] = {c: -1 for c in N2C2_CRITERIA}
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            for tag_elem in root.findall(".//TAGS/*"):
                criterion = tag_elem.tag.upper().replace("_", "-")
                met_value = tag_elem.get("met", "").lower()
                if criterion in labels and met_value in N2C2_LABEL_MAP:
                    labels[criterion] = N2C2_LABEL_MAP[met_value]
        except ET.ParseError as e:
            logger.warning("Failed to parse labels %s: %s", path, e)

        return labels

    def to_patient_profiles(
        self, dataset: N2C2Dataset, enrollment_date: date | None = None
    ) -> list[PatientProfile]:
        """Convert n2c2 patients to PatientProfile objects for the matching engine."""
        if enrollment_date is None:
            enrollment_date = date(2012, 1, 1)  # n2c2 2018 data era

        profiles = []
        for n2c2_patient in dataset.patients:
            notes = [
                ClinicalNote(
                    note_id=f"{n2c2_patient.patient_id}-{i}",
                    note_type="discharge_summary",
                    text=note_text,
                )
                for i, note_text in enumerate(n2c2_patient.notes)
            ]
            profile = PatientProfile(
                patient_id=n2c2_patient.patient_id,
                source="n2c2",
                enrollment_date=enrollment_date,
                notes=notes,
                data_completeness_score=0.7,  # n2c2 only has notes, no structured data
            )
            profiles.append(profile)
        return profiles
