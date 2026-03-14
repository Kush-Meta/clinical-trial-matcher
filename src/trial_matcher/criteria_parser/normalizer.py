"""Concept normalization: drug→RxNorm, lab→LOINC, diagnosis→ICD-10."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

_RXNORM_CACHE: dict[str, str | None] = {}
_LOINC_INDEX: dict[str, str] = {}
_ICD10_INDEX: dict[str, str] = {}
_LOINC_LOADED = False
_ICD10_LOADED = False

ONTOLOGIES_DIR = Path("data/ontologies")
RXNORM_CACHE_PATH = ONTOLOGIES_DIR / "rxnorm_cache.json"
LOINC_PATH = ONTOLOGIES_DIR / "loinc_top500.json"
ICD10_PATH = ONTOLOGIES_DIR / "icd10_common.json"

FUZZY_THRESHOLD = 85  # minimum score for fuzzy match (0-100)


def _load_rxnorm_cache() -> None:
    global _RXNORM_CACHE
    if RXNORM_CACHE_PATH.exists():
        _RXNORM_CACHE = json.loads(RXNORM_CACHE_PATH.read_text())


def _save_rxnorm_cache() -> None:
    RXNORM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RXNORM_CACHE_PATH.write_text(json.dumps(_RXNORM_CACHE, indent=2))


def _load_loinc() -> None:
    global _LOINC_INDEX, _LOINC_LOADED
    if not _LOINC_LOADED and LOINC_PATH.exists():
        _LOINC_INDEX = json.loads(LOINC_PATH.read_text())
        _LOINC_LOADED = True


def _load_icd10() -> None:
    global _ICD10_INDEX, _ICD10_LOADED
    if not _ICD10_LOADED and ICD10_PATH.exists():
        data = json.loads(ICD10_PATH.read_text())
        # Flatten: {icd10_code: description} or {description: icd10_code}
        if data and isinstance(list(data.values())[0], str):
            # Check format: if values look like ICD codes, it's description→code
            sample_value = list(data.values())[0]
            if len(sample_value) <= 8 and sample_value[0].isalpha():
                _ICD10_INDEX = data  # description → ICD10
            else:
                _ICD10_INDEX = {v: k for k, v in data.items()}  # ICD10 → description, invert
        _ICD10_LOADED = True


# ── RxNorm ────────────────────────────────────────────────────────────────────


def lookup_rxnorm(drug_name: str) -> str | None:
    """Return RxCUI for a drug name. Caches locally, falls back to RxNav API."""
    if not _RXNORM_CACHE:
        _load_rxnorm_cache()

    key = drug_name.lower().strip()
    if key in _RXNORM_CACHE:
        return _RXNORM_CACHE[key]

    # Query RxNav (free, no auth)
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}&search=1"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
            rxcui = rxcui_list[0] if rxcui_list else None
        time.sleep(0.1)  # polite rate limiting
    except Exception as e:
        logger.debug("RxNorm lookup failed for '%s': %s", drug_name, e)
        rxcui = None

    _RXNORM_CACHE[key] = rxcui
    _save_rxnorm_cache()
    return rxcui


# ── LOINC ─────────────────────────────────────────────────────────────────────


def lookup_loinc(lab_name: str) -> str | None:
    """Return LOINC code for a lab name via fuzzy matching."""
    _load_loinc()
    if not _LOINC_INDEX:
        return None

    result = process.extractOne(
        lab_name.lower(),
        {k.lower(): v for k, v in _LOINC_INDEX.items()},
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if result:
        _, _, key = result
        return _LOINC_INDEX.get(key) or list(_LOINC_INDEX.values())[0]
    return None


# ── ICD-10 ───────────────────────────────────────────────────────────────────

# Common synonyms for ICD-10 expansion
_SYNONYMS: dict[str, str] = {
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
    "cad": "coronary artery disease",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "htn": "hypertension",
    "ckd": "chronic kidney disease",
    "afib": "atrial fibrillation",
    "chf": "congestive heart failure",
    "hf": "heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "tia": "transient ischemic attack",
    "cva": "cerebrovascular accident",
}


def lookup_icd10(condition_name: str) -> str | None:
    """Return ICD-10 code for a condition name via fuzzy/synonym matching."""
    _load_icd10()
    if not _ICD10_INDEX:
        return None

    query = condition_name.lower().strip()
    # Synonym expansion
    query = _SYNONYMS.get(query, query)

    result = process.extractOne(
        query,
        list(_ICD10_INDEX.keys()),
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if result:
        matched_key, score, _ = result
        logger.debug("ICD-10 match '%s' → '%s' (score %d)", condition_name, matched_key, score)
        return _ICD10_INDEX[matched_key]
    return None


def normalize_criterion(criterion: object) -> None:
    """In-place normalization of criterion codes (mutates the object)."""
    from trial_matcher.schemas.criteria import (
        DiagnosisCriterion,
        LabValueCriterion,
        MedicationCriterion,
    )

    if isinstance(criterion, LabValueCriterion) and criterion.loinc_code is None:
        criterion.loinc_code = lookup_loinc(criterion.lab_name)

    elif isinstance(criterion, DiagnosisCriterion) and criterion.icd10_code is None:
        criterion.icd10_code = lookup_icd10(criterion.condition_name)

    elif isinstance(criterion, MedicationCriterion) and criterion.rxnorm_code is None:
        criterion.rxnorm_code = lookup_rxnorm(criterion.drug_name)
