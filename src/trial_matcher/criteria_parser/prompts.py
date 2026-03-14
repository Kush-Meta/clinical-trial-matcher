"""Versioned prompt templates for criteria extraction."""

from __future__ import annotations

PROMPT_VERSION = "0.1.0"

SYSTEM_PROMPT = """You are a clinical trial eligibility parser. Your task is to extract structured information from a single eligibility criterion line.

Rules:
1. Output ONLY valid JSON matching the requested schema — no prose, no markdown.
2. If the criterion is ambiguous or unclear, set "ambiguous": true and explain in "ambiguity_note".
3. NEVER infer information not present in the text.
4. For temporal expressions, extract exact values (e.g., "within 6 months" → operator="WITHIN", value=6, unit="months").
5. For composite criteria ("at least 2 of the following"), use type="COMPOSITE".
6. Preserve exact numeric thresholds — do not round or interpret.
7. If you cannot determine the criterion type, use type="UNKNOWN".
"""

LAB_VALUE_EXAMPLE = """Input criterion: "HbA1c between 6.5% and 9.5% within the last 3 months"
Output:
{
  "type": "LAB_VALUE",
  "lab_name": "HbA1c",
  "loinc_code": null,
  "threshold": {"operator": "BETWEEN", "value": 6.5, "value_high": 9.5, "unit": "%"},
  "temporal": {"operator": "WITHIN", "value": 3, "unit": "months", "raw_text": "within the last 3 months"},
  "raw_text": "HbA1c between 6.5% and 9.5% within the last 3 months",
  "ambiguous": false,
  "ambiguity_note": ""
}"""

COMPOSITE_EXAMPLE = """Input criterion: "Patient must have at least 2 of the following: history of MI, coronary artery bypass graft surgery, or percutaneous coronary intervention"
Output:
{
  "type": "COMPOSITE",
  "operator": "AT_LEAST_N",
  "n": 2,
  "children": [
    {"type": "DIAGNOSIS", "condition_name": "myocardial infarction", "icd10_code": null, "temporal": null, "required_absent": false, "raw_text": "history of MI", "ambiguous": false, "ambiguity_note": ""},
    {"type": "PROCEDURE", "procedure_name": "coronary artery bypass graft surgery", "cpt_code": null, "temporal": null, "required_absent": false, "raw_text": "coronary artery bypass graft surgery", "ambiguous": false, "ambiguity_note": ""},
    {"type": "PROCEDURE", "procedure_name": "percutaneous coronary intervention", "cpt_code": null, "temporal": null, "required_absent": false, "raw_text": "percutaneous coronary intervention", "ambiguous": false, "ambiguity_note": ""}
  ],
  "raw_text": "Patient must have at least 2 of the following...",
  "ambiguous": false,
  "ambiguity_note": ""
}"""

BEHAVIORAL_EXAMPLE = """Input criterion: "No current use of dietary supplements within 2 months"
Output:
{
  "type": "BEHAVIORAL",
  "behavior": "DIETARY_SUPPLEMENT",
  "details": "dietary supplements",
  "temporal": {"operator": "WITHIN", "value": 2, "unit": "months", "raw_text": "within 2 months"},
  "required_absent": true,
  "raw_text": "No current use of dietary supplements within 2 months",
  "ambiguous": false,
  "ambiguity_note": ""
}"""

FUNCTIONAL_EXAMPLE = """Input criterion: "Patient is able to make their own informed decisions"
Output:
{
  "type": "FUNCTIONAL",
  "capability": "makes own informed decisions",
  "scale": null,
  "threshold": null,
  "raw_text": "Patient is able to make their own informed decisions",
  "ambiguous": false,
  "ambiguity_note": ""
}"""


def build_extraction_prompt(criterion_text: str, section: str = "INCLUSION") -> str:
    """Build the user message for a single criterion extraction."""
    return f"""Extract the structured criterion from this {section} criterion line.

Few-shot examples:
{LAB_VALUE_EXAMPLE}

{COMPOSITE_EXAMPLE}

{BEHAVIORAL_EXAMPLE}

{FUNCTIONAL_EXAMPLE}

Now extract from this criterion:
Input criterion: "{criterion_text}"

Output a single JSON object. For the "type" field, use one of: LAB_VALUE, DIAGNOSIS, MEDICATION, PROCEDURE, BEHAVIORAL, FUNCTIONAL, DEMOGRAPHIC, COMPOSITE, UNKNOWN.
"""


SEGMENTATION_HEADERS = [
    "inclusion criteria",
    "inclusion criterion",
    "exclusion criteria",
    "exclusion criterion",
    "eligibility criteria",
    "key inclusion",
    "key exclusion",
    "select inclusion",
    "select exclusion",
]
