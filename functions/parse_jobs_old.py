from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import re as _re

from models.model import model, embedding_model




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


# job_texts = [build_job_text(job) for job in jobs]

# print(job_texts[0][:500])  # preview first job


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

    # sort by similarity
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




def search_jobs(query, jobs, top_k=10):
    if not query or len(jobs) == 0:
        return []

    # Create query embedding
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

    similarities = cosine_similarity([query_embedding], job_embeddings)[0]

    # Attach score safely
    results = []
    for i, job in enumerate(jobs):
        job_copy = job.copy()
        job_copy["score"] = float(similarities[i])
        results.append(job_copy)

    # Sort
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]