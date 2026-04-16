"""
functions/career.py

Career trajectory recommendation engine.

Responsibilities:
  - Infer candidate profile from parsed resume
  - Build a skill vocabulary from the job dataset
  - Score each unique job title by reachability (overlap + gap size)
  - Filter by seniority and deduplication
  - Return structured data that the orchestrator (main.py) can enrich with
    LLM narrative and live market stats.

"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# SENIORITY
# ─────────────────────────────────────────────────────────────────────────────

# Ordered low → high.  Used for both inferring candidate level and filtering
# which target titles are genuine "next steps".
SENIORITY_LEVELS = ["entry", "mid", "senior", "lead", "principal", "director"]

_SENIORITY_KEYWORDS: dict[str, list[str]] = {
    "entry":     ["junior", "associate", "graduate", "intern", "trainee", "entry"],
    "mid":       [],          # default when no keyword matches
    "senior":    ["senior", "sr.", "sr "],
    "lead":      ["lead", "staff", "tech lead", "team lead"],
    "principal": ["principal", "distinguished", "architect", "head of"],
    "director":  ["director", "vp ", "vice president", "chief", "cto", "cpo"],
}


def infer_seniority(title: str) -> str:
    """
    Return a seniority string for a job title string.

    >>> infer_seniority("Senior Machine Learning Engineer")
    'senior'
    >>> infer_seniority("Junior Data Analyst")
    'entry'
    >>> infer_seniority("Data Scientist")
    'mid'
    """
    lower = title.lower()
    for level, keywords in _SENIORITY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return level
    return "mid"


def next_seniority_levels(current: str) -> list[str]:
    """
    Return seniority levels that are valid "next step" targets for the candidate.
    We allow same level (lateral) and up to two levels above.

    >>> next_seniority_levels("mid")
    ['mid', 'senior', 'lead']
    """
    if current not in SENIORITY_LEVELS:
        current = "mid"
    idx = SENIORITY_LEVELS.index(current)
    return SENIORITY_LEVELS[idx: idx + 3]


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE PROFILE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_years_of_experience(work_experience: list[dict]) -> float:
    """
    Sum total months across all work experience entries and return as years.
    Handles 'present' / 'Present' as today's date.
    Gracefully skips unparseable entries.
    """
    _DATE_FORMATS = ["%B %Y", "%b %Y", "%m/%Y", "%Y"]
    today = date.today()
    total_months = 0.0

    for exp in work_experience:
        start_raw = exp.get("start_date") or ""
        end_raw   = (exp.get("end_date") or "present").strip()

        start_dt = _parse_date_flexible(start_raw, _DATE_FORMATS)
        if start_dt is None:
            continue

        if end_raw.lower() in ("present", "current", "now", ""):
            end_dt = today
        else:
            end_dt = _parse_date_flexible(end_raw, _DATE_FORMATS)
            if end_dt is None:
                end_dt = today

        delta_months = (
            (end_dt.year - start_dt.year) * 12
            + (end_dt.month - start_dt.month)
        )
        total_months += max(0, delta_months)

    return round(total_months / 12, 1)


def _parse_date_flexible(raw: str, formats: list[str]) -> Optional[date]:
    raw = raw.strip()
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def extract_candidate_skills(resume_text: str, skill_vocabulary: set[str]) -> set[str]:
    """
    Find which skills from skill_vocabulary appear in resume_text.
    Case-insensitive whole-word / whole-phrase match.

    Args:
        resume_text:      Full resume text (from build_resume_text).
        skill_vocabulary: All skills seen across the job dataset.

    Returns:
        Set of matched skill strings (in their original vocabulary casing).
    """
    lower_text = resume_text.lower()
    found: set[str] = set()

    for skill in skill_vocabulary:
        # Escape and match as a whole token (handles "C++", "Node.js", etc.)
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lower_text):
            found.add(skill)

    return found


def get_current_title(work_experience: list[dict]) -> str:
    """
    Return the most recent job title, or empty string if unavailable.
    Assumes work_experience is ordered most-recent first (as our parser outputs).
    """
    if not work_experience:
        return ""
    return work_experience[0].get("position", "")


# ─────────────────────────────────────────────────────────────────────────────
# JOB MARKET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_skill_vocabulary(jobs: list[dict]) -> set[str]:
    """
    Return the set of all unique skills that appear across the job dataset.
    """
    vocab: set[str] = set()
    for job in jobs:
        vocab.update(job.get("skills_required", []))
    return vocab


def build_title_skill_profiles(
    jobs: list[dict],
    min_frequency: float = 0.25,
) -> dict[str, set[str]]:
    """
    For each unique job title, compute the set of skills that appear in at
    least `min_frequency` fraction of postings for that title.

    Args:
        jobs:          Full job list.
        min_frequency: Skill must appear in ≥ this fraction of a title's
                       postings to be considered "typical". Default 25%.

    Returns:
        { title: {skill, skill, ...} }
    """
    # Accumulate skills per title
    title_skill_counts: dict[str, Counter] = defaultdict(Counter)
    title_job_counts:   dict[str, int]     = defaultdict(int)

    for job in jobs:
        title = job.get("title", "").strip()
        if not title:
            continue
        title_job_counts[title] += 1
        for skill in job.get("skills_required", []):
            title_skill_counts[title][skill] += 1

    profiles: dict[str, set[str]] = {}
    for title, skill_counter in title_skill_counts.items():
        n_jobs = title_job_counts[title]
        threshold = max(1, round(min_frequency * n_jobs))
        profiles[title] = {
            skill for skill, count in skill_counter.items()
            if count >= threshold
        }

    return profiles


def score_reachable_titles(
    candidate_skills:     set[str],
    title_skill_profiles: dict[str, set[str]],
    current_title:        str,
    candidate_seniority:  str,
    min_gap:              int = 1,
    max_gap:              int = 5,
) -> list[dict]:
    """
    Score every job title by how reachable it is for this candidate.

    A title is "reachable" when:
      - It is not the candidate's current title (exact, case-insensitive)
      - Its seniority is within the candidate's next valid levels
      - The skill gap is between min_gap and max_gap (inclusive)

    Returns a list of dicts sorted by overlap_score descending:
        {
          "title":         str,
          "required_skills": set,
          "owned_skills":  set,   # candidate skills that match
          "gap_skills":    list,  # missing skills (sorted)
          "gap_count":     int,
          "overlap_score": float, # fraction of required skills already owned
          "seniority":     str,
        }
    """
    current_lower  = current_title.lower()
    valid_levels   = set(next_seniority_levels(candidate_seniority))
    results        = []

    for title, required_skills in title_skill_profiles.items():
        if not required_skills:
            continue

        # Exclude current title
        if title.lower() == current_lower:
            continue

        # Seniority gate
        if infer_seniority(title) not in valid_levels:
            continue

        owned   = candidate_skills & required_skills
        missing = required_skills - candidate_skills
        gap     = len(missing)

        if not (min_gap <= gap <= max_gap):
            continue

        overlap_score = len(owned) / len(required_skills)

        results.append({
            "title":           title,
            "required_skills": required_skills,
            "owned_skills":    owned,
            "gap_skills":      sorted(missing),
            "gap_count":       gap,
            "overlap_score":   round(overlap_score, 3),
            "seniority":       infer_seniority(title),
        })

    return sorted(results, key=lambda x: x["overlap_score"], reverse=True)




def deduplicate_titles(
    scored_titles: list[dict],
    model,                        # sentence-transformers model (injected)
    similarity_threshold: float = 0.88,
    top_n: int = 5,
) -> list[dict]:
    """
    Remove near-duplicate titles using embedding cosine similarity so we don't
    return "ML Engineer", "Machine Learning Engineer", "ML Software Engineer"
    as three separate recommendations.

    Greedy: keep the highest-scoring title from each cluster.

    Args:
        scored_titles:        Output of score_reachable_titles (sorted desc).
        model:                Sentence-transformers model with .encode().
        similarity_threshold: Titles with cosine sim ≥ this are considered dupes.
        top_n:                Return at most this many titles after dedup.
    """
    if not scored_titles:
        return []

    titles = [t["title"] for t in scored_titles]
    embeddings = model.encode(titles)                    # (N, dim)
    sims = cosine_similarity(embeddings)                 # (N, N)

    kept_indices: list[int] = []
    seen: set[int] = set()

    for i in range(len(scored_titles)):
        if i in seen:
            continue
        kept_indices.append(i)
        if len(kept_indices) == top_n:
            break
        # Mark all near-duplicates of i as seen
        for j in range(i + 1, len(scored_titles)):
            if sims[i][j] >= similarity_threshold:
                seen.add(j)

    return [scored_titles[i] for i in kept_indices]


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MARKET ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_market_data(
    recommendations: list[dict],
    jobs:            list[dict],
) -> list[dict]:
    """
    For each recommended title, attach live market stats from the job dataset:
      - job_count:       number of open postings
      - avg_salary:      average midpoint salary (int, USD) or None
      - top_companies:   up to 3 company names hiring for this title

    Mutates and returns the list in-place.
    """
    for rec in recommendations:
        target = rec["title"].lower()
        matching = [j for j in jobs if j.get("title", "").lower() == target]

        rec["job_count"]    = len(matching)
        rec["avg_salary"]   = _avg_salary(matching)
        rec["top_companies"] = _top_companies(matching, n=3)

    return recommendations


def _parse_salary_midpoint(salary_str: str | None) -> float | None:
    if not salary_str or not isinstance(salary_str, str):
        return None
    cleaned = re.sub(r"[$£€,\s]", "", salary_str)
    numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not numbers:
        return None
    values = [float(n) for n in numbers if float(n) >= 1000]
    return sum(values) / len(values) if values else None


def _avg_salary(jobs: list[dict]) -> int | None:
    midpoints = [
        m for m in
        (_parse_salary_midpoint(j.get("salary_range")) for j in jobs)
        if m is not None
    ]
    if not midpoints:
        return None
    return int(round(sum(midpoints) / len(midpoints)))


def _top_companies(jobs: list[dict], n: int = 3) -> list[str]:
    counter: Counter = Counter(
        j.get("company", "").strip()
        for j in jobs
        if j.get("company", "").strip()
    )
    return [company for company, _ in counter.most_common(n)]