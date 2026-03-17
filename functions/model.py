# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer("all-MiniLM-L6-v2")
# print("Embedding model loaded successfully.")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once (important)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


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