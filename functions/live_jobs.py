# =============================================================================
# functions/live_jobs.py
#
# Fetches live job listings from JSearch (RapidAPI) and normalises them into
# the exact same schema used by data/jobs3.json so the rest of the app
# (semantic search, fit analysis, bookmarks, filters) works without changes.
#
# Schema target:
# {
#   "job_id", "title", "company", "location", "employment_type",
#   "experience_required", "education_required", "skills_required",
#   "job_description", "responsibilities", "preferred_qualifications",
#   "salary_range", "posting_date"
# }
# =============================================================================

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com/search"

def _get_headers() -> dict:
    """Read key at call time so dotenv has a chance to load it first."""
    return {
        "X-RapidAPI-Key":  os.environ.get("JSEARCH_API_KEY", ""),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

# Path where fetched jobs are cached so we don't hit the API on every request.
CACHE_PATH = Path("data/live_jobs_cache.json")


# =============================================================================
# Public API
# =============================================================================

def get_live_jobs(query: str, num_pages: int = 5, location: str = "") -> list[dict]:
    """
    Fetch jobs from JSearch for the given query and return them normalised
    to the app's job schema.

    - num_pages: each page = 10 results, so 5 pages = up to 50 jobs per query.
    - location:  optional location string appended to query, e.g. "London, UK"
                 JSearch resolves location via the query string.

    Falls back to an empty list (with a logged warning) if the API key is
    missing or the request fails — the app keeps working with static data.
    """
    api_key = os.environ.get("JSEARCH_API_KEY", "")
    if not api_key:
        logger.warning("[live_jobs] JSEARCH_API_KEY not set — skipping live fetch.")
        return []

    # JSearch handles location by including it in the query string
    full_query = f"{query} in {location}" if location else query

    try:
        raw  = _fetch_jsearch(full_query, num_pages)
        jobs = [_normalise(j) for j in raw]
        logger.info(
            f"[live_jobs] fetched {len(jobs)} jobs for query='{full_query}'"
        )
        return jobs
    except Exception as e:
        logger.error(f"[live_jobs] fetch failed: {e}")
        return []


def refresh_cache(
    queries: list[str],
    num_pages: int = 5,
    locations: list[str] | None = None,
) -> None:
    """
    Fetch jobs for a list of seed queries (optionally per location) and write
    them to data/live_jobs_cache.json.  Called by the background scheduler.

    If locations is provided, every query is fetched for every location,
    e.g. queries=["ML engineer"], locations=["London", "Remote"] fetches both.
    """
    seen_ids: set[str] = set()
    all_jobs: list[dict] = []

    location_list = locations or [""]   # empty string = no location filter

    for query in queries:
        for loc in location_list:
            for job in get_live_jobs(query, num_pages, location=loc):
                if job["job_id"] not in seen_ids:
                    seen_ids.add(job["job_id"])
                    all_jobs.append(job)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(all_jobs, f, indent=2)

    logger.info(f"[live_jobs] cache refreshed — {len(all_jobs)} unique jobs written.")


def load_cached_jobs() -> list[dict]:
    """Load previously cached live jobs, or return [] if cache doesn't exist."""
    if not CACHE_PATH.exists():
        return []
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[live_jobs] failed to load cache: {e}")
        return []


# =============================================================================
# Internal helpers
# =============================================================================

def _fetch_jsearch(query: str, num_pages: int) -> list[dict]:
    """Call the JSearch API and return the raw data list."""
    all_results: list[dict] = []
    headers = _get_headers()

    for page in range(1, num_pages + 1):
        params = {
            "query":       query,
            "page":        str(page),
            "num_pages":   "1",   # fetch 1 page at a time, loop for more
            "date_posted": "all",
        }
        with httpx.Client(timeout=20) as client:
            resp = client.get(JSEARCH_BASE_URL, headers=headers, params=params)
            resp.raise_for_status()

        data    = resp.json()
        results = data.get("data", [])
        if not results:
            break  # no more pages available
        all_results.extend(results)

    return all_results


def _normalise(j: dict) -> dict:
    """
    Map a raw JSearch job dict → the app's job schema.

    JSearch field reference:
      job_id, job_title, employer_name, job_city, job_state, job_country,
      job_employment_type, job_description, job_required_skills,
      job_required_experience, job_required_education, job_highlights,
      job_min_salary, job_max_salary, job_salary_period,
      job_posted_at_datetime_utc
    """
    return {
        "job_id":                   j.get("job_id", ""),
        "title":                    j.get("job_title", "Untitled"),
        "company":                  j.get("employer_name", "Unknown Company"),
        "location":                 _build_location(j),
        "employment_type":          _clean_employment_type(j.get("job_employment_type", "")),
        "experience_required":      _extract_experience(j),
        "education_required":       _extract_education(j),
        "skills_required":          j.get("job_required_skills") or [],
        "job_description":          j.get("job_description", ""),
        "responsibilities":         _extract_highlights(j, "Responsibilities"),
        "preferred_qualifications": _extract_highlights(j, "Qualifications"),
        "salary_range":             _format_salary(j),
        "posting_date":             _format_date(j.get("job_posted_at_datetime_utc", "")),
    }


def _build_location(j: dict) -> str:
    parts = [
        j.get("job_city", ""),
        j.get("job_state", ""),
        j.get("job_country", ""),
    ]
    return ", ".join(p for p in parts if p).strip(", ") or "Remote"


def _clean_employment_type(raw: str) -> str:
    mapping = {
        "FULLTIME":  "Full-time",
        "PARTTIME":  "Part-time",
        "CONTRACTOR": "Contract",
        "INTERN":    "Internship",
    }
    return mapping.get(raw.upper(), raw.replace("_", " ").title() if raw else "Not specified")


def _extract_experience(j: dict) -> str:
    exp = j.get("job_required_experience", {}) or {}
    years = exp.get("required_experience_in_months")
    if years:
        yr = round(years / 12)
        return f"{yr}+ years"
    no_exp = exp.get("no_experience_required")
    if no_exp:
        return "No experience required"
    return "Not specified"


def _extract_education(j: dict) -> str:
    edu = j.get("job_required_education", {}) or {}
    # Check fields from most to least specific
    for field in (
        "postgraduate_degree",
        "professional_certification",
        "bachelors_degree",
        "associates_degree",
        "high_school",
    ):
        if edu.get(field):
            return field.replace("_", " ").title()
    return "Not specified"


def _extract_highlights(j: dict, section: str) -> list[str]:
    """Pull bullet points from job_highlights under the given section key."""
    highlights = j.get("job_highlights", {}) or {}
    return highlights.get(section, [])


def _format_salary(j: dict) -> str:
    lo     = j.get("job_min_salary")
    hi     = j.get("job_max_salary")
    period = j.get("job_salary_period", "")

    period_label = {
        "YEAR":  "per year",
        "MONTH": "per month",
        "HOUR":  "per hour",
    }.get((period or "").upper(), period)

    if lo and hi:
        return f"${lo:,.0f} – ${hi:,.0f} {period_label}".strip()
    if lo:
        return f"${lo:,.0f}+ {period_label}".strip()
    return "Not specified"


def _format_date(raw: str) -> str:
    """Convert ISO datetime string → YYYY-MM-DD, or today's date as fallback."""
    if raw:
        try:
            return raw[:10]  # "2024-03-15T..."  → "2024-03-15"
        except Exception:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")