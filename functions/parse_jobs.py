from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import re as _re
from typing import Optional

from models.model import model, embedding_model


# =============================================================================
# Original functions — unchanged
# =============================================================================

def build_job_text(job):
    """
    Convert job JSON into semantic text block.
    """
    skills = ", ".join(job.get("skills_required", []))
    responsibilities = " ".join(job.get("responsibilities", []))

    job_text = f"""
    Job Title: {job.get('title', '')}
    Company: {job.get('company', '')}
    Location: {job.get('location', '')}
    Experience Required: {job.get('experience_required', '')}
    Education Required: {job.get('education_required', '')}
    Required Skills: {skills}
    Job Description: {job.get('job_description', '')}
    Responsibilities: {responsibilities}
    """

    return job_text.strip()


def compute_title_embeddings(jobs):
    titles = []
    embeddings = []

    for job in jobs:
        title = job["title"]
        titles.append(title)
        emb = embedding_model.encode(title)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    return titles, embeddings


def find_similar_jobs(user_query, jobs, title_embeddings, threshold=0.75):
    query_embedding = embedding_model.encode(user_query)

    similarities = cosine_similarity(
        [query_embedding],
        title_embeddings
    )[0]

    matched_jobs = []

    for i, score in enumerate(similarities):
        if score >= threshold:
            job = jobs[i].copy()
            job["title_similarity"] = float(score)
            matched_jobs.append(job)

    matched_jobs = sorted(
        matched_jobs,
        key=lambda x: x["title_similarity"],
        reverse=True
    )

    return matched_jobs


def get_top_skills(matched_jobs, top_n=10):
    skill_counter = Counter()

    for job in matched_jobs:
        skills = job.get("skills_required", [])
        skill_counter.update(skills)

    return skill_counter.most_common(top_n)


def parse_salary(salary_str) -> float | None:
    """
    Parse a salary_range string like "$140,000 - $175,000" into a midpoint float.
    Returns None if the string is missing or unparseable.
    """
    if not salary_str or not isinstance(salary_str, str):
        return None

    cleaned = _re.sub(r"[$£€,\s]", "", salary_str)
    numbers = _re.findall(r"\d+(?:\.\d+)?", cleaned)

    if not numbers:
        return None

    values = [float(n) for n in numbers]
    values = [v for v in values if v >= 1000]   # drop implausible values

    if not values:
        return None

    return sum(values) / len(values)


def search_jobs(query, jobs, top_k=50):
    """
    Semantic search using sentence embeddings.
    Returns up to top_k jobs ranked by cosine similarity to the query.
    top_k raised from 10 → 50 to show more results.
    """
    if not query or len(jobs) == 0:
        return []

    query_embedding = model.encode(query)

    job_texts = []
    for job in jobs:
        text = f"""
        {job.get('title', '')}
        {job.get('job_description', '')}
        {' '.join(job.get('skills_required', []))}
        """
        job_texts.append(text)

    job_embeddings = model.encode(job_texts)
    similarities   = cosine_similarity([query_embedding], job_embeddings)[0]

    results = []
    for i, job in enumerate(jobs):
        job_copy = job.copy()
        job_copy["score"] = float(similarities[i])
        results.append(job_copy)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# NEW — location filter (added without touching any original functions above)
# =============================================================================

def filter_by_location(jobs: list, location: str) -> list:
    """
    Filter jobs to only those whose location field matches the given location.
    Called after search_jobs() in app.py so relevance scoring is unaffected.

    Matching strategy (most → least strict):
      1. Exact match          "London, UK" == "London, UK"
      2. All tokens present   every word in the query appears in the job location
      3. Short tokens         "uk", "us" matched together with at least one long token
      4. Remote special case  jobs with "remote" in location/employment_type always
                              included when user searches "remote"
      5. Fallback             if nothing matched strictly, relax to any-token match
                              so the user never sees a blank page

    Returns jobs unchanged if location is empty/blank.
    """
    location = location.strip()
    if not location:
        return jobs

    def _tokenize(text: str) -> list:
        return _re.findall(r"[a-z0-9]+", text.lower())

    def _field_text(job: dict, field: str) -> str:
        val = job.get(field, "")
        if isinstance(val, list):
            return " ".join(str(v) for v in val)
        return str(val) if val else ""

    loc_lower    = location.lower()
    all_tokens   = _tokenize(location)
    loc_tokens   = [t for t in all_tokens if len(t) > 2]   # "london", "york"
    short_tokens = [t for t in all_tokens if len(t) <= 2]  # "uk", "us", "ny"
    wants_remote = "remote" in loc_lower

    filtered = []
    for job in jobs:
        job_loc  = _field_text(job, "location").lower()
        job_type = _field_text(job, "employment_type").lower()

        # Remote shortcut
        if wants_remote and ("remote" in job_loc or "remote" in job_type):
            filtered.append(job)
            continue

        # Exact match
        if loc_lower == job_loc:
            filtered.append(job)
            continue

        # All long tokens present
        if loc_tokens and all(t in job_loc for t in loc_tokens):
            filtered.append(job)
            continue

        # Short tokens + at least one long token
        if short_tokens and all(t in job_loc for t in short_tokens):
            if not loc_tokens or any(t in job_loc for t in loc_tokens):
                filtered.append(job)
                continue

    # Fallback: relax to any-token if strict pass returned nothing
    if not filtered and loc_tokens:
        for job in jobs:
            job_loc = _field_text(job, "location").lower()
            if any(t in job_loc for t in loc_tokens):
                filtered.append(job)

    return filtered