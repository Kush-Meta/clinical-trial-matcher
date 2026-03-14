"""ClinicalTrials.gov v2 API client (no authentication required)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx

logger = logging.getLogger(__name__)

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_PAGE_SIZE = 20
REQUEST_DELAY = 0.1  # seconds between requests


@dataclass
class ClinicalTrial:
    nct_id: str
    brief_title: str
    official_title: str = ""
    overall_status: str = ""
    phase: str = ""
    conditions: list[str] = field(default_factory=list)
    minimum_age: str = ""
    maximum_age: str = ""
    sex: str = "ALL"
    eligibility_criteria: str = ""
    lead_sponsor: str = ""
    location_countries: list[str] = field(default_factory=list)
    start_date: str = ""
    primary_completion_date: str = ""
    enrollment: int | None = None

    @property
    def display_age_range(self) -> str:
        parts = []
        if self.minimum_age:
            parts.append(f"≥ {self.minimum_age}")
        if self.maximum_age:
            parts.append(f"≤ {self.maximum_age}")
        return " and ".join(parts) if parts else "Any age"


def _parse_trial(study: dict[str, Any]) -> ClinicalTrial:
    """Parse a ClinicalTrials.gov v2 API study response."""
    proto = study.get("protocolSection", {})
    id_module = proto.get("identificationModule", {})
    status_module = proto.get("statusModule", {})
    design_module = proto.get("designModule", {})
    elig_module = proto.get("eligibilityModule", {})
    sponsor_module = proto.get("sponsorCollaboratorsModule", {})
    conditions_module = proto.get("conditionsModule", {})
    contacts_module = proto.get("contactsLocationsModule", {})

    # Conditions
    conditions = conditions_module.get("conditions", [])

    # Phase
    phases = design_module.get("phases", [])
    phase = "/".join(phases) if phases else ""

    # Location countries
    locations = contacts_module.get("locations", [])
    countries = list({loc.get("country", "") for loc in locations if loc.get("country")})

    # Lead sponsor
    lead_sponsor_info = sponsor_module.get("leadSponsor", {})
    lead_sponsor = lead_sponsor_info.get("name", "")

    # Enrollment
    enroll = design_module.get("enrollmentInfo", {})
    try:
        enrollment = int(enroll.get("count", 0)) or None
    except (ValueError, TypeError):
        enrollment = None

    return ClinicalTrial(
        nct_id=id_module.get("nctId", ""),
        brief_title=id_module.get("briefTitle", ""),
        official_title=id_module.get("officialTitle", ""),
        overall_status=status_module.get("overallStatus", ""),
        phase=phase,
        conditions=conditions,
        minimum_age=elig_module.get("minimumAge", ""),
        maximum_age=elig_module.get("maximumAge", ""),
        sex=elig_module.get("sex", "ALL"),
        eligibility_criteria=elig_module.get("eligibilityCriteria", ""),
        lead_sponsor=lead_sponsor,
        location_countries=countries,
        start_date=status_module.get("startDateStruct", {}).get("date", ""),
        primary_completion_date=status_module.get("primaryCompletionDateStruct", {}).get("date", ""),
        enrollment=enrollment,
    )


class CTGovClient:
    """Client for ClinicalTrials.gov v2 REST API."""

    def __init__(
        self,
        base_url: str = CTGOV_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "clinical-trial-matcher/0.1.0"},
        )

    def search(
        self,
        query: str,
        status: str = "RECRUITING",
        page_size: int = DEFAULT_PAGE_SIZE,
        max_results: int = 100,
    ) -> list[ClinicalTrial]:
        """Search for trials matching a query term."""
        results = []
        for trial in self.paginate(query, status=status, page_size=page_size):
            results.append(trial)
            if len(results) >= max_results:
                break
        return results

    def get_trial(self, nct_id: str) -> ClinicalTrial | None:
        """Fetch a single trial by NCT ID."""
        url = f"{self.base_url}/studies/{nct_id}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return _parse_trial(data)
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch trial %s: %s", nct_id, e)
            return None

    def paginate(
        self,
        query: str,
        status: str = "RECRUITING",
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Generator[ClinicalTrial, None, None]:
        """Paginate through all results for a query."""
        params: dict[str, Any] = {
            "query.term": query,
            "pageSize": page_size,
            "format": "json",
        }
        if status:
            params["filter.overallStatus"] = status

        next_page_token: str | None = None

        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            elif "pageToken" in params:
                del params["pageToken"]

            try:
                resp = self._client.get(f"{self.base_url}/studies", params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.error("ClinicalTrials.gov API error: %s", e)
                break

            studies = data.get("studies", [])
            for study in studies:
                trial = _parse_trial(study)
                if trial.nct_id:
                    yield trial

            next_page_token = data.get("nextPageToken")
            if not next_page_token or not studies:
                break

            time.sleep(REQUEST_DELAY)

    def get_trials_by_condition(
        self,
        condition: str,
        max_results: int = 10,
    ) -> list[ClinicalTrial]:
        """Search by specific condition (e.g., 'type 2 diabetes')."""
        return self.search(
            query=condition,
            status="RECRUITING",
            max_results=max_results,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CTGovClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
