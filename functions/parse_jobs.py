from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


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




def find_similar_jobs(user_query, jobs, title_embeddings, threshold=0.7):

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