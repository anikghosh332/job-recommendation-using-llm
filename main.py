from functions.parse_resume import ResumeParser, build_resume_text
from functions.parse_jobs import build_job_text, compute_title_embeddings, find_similar_jobs, get_top_skills, parse_salary

import json
from sklearn.metrics.pairwise import cosine_similarity
from functions.llm import llm_call
from functions.llm_recommendations import identify_skill_gaps, explain_matching_quality
from functions.career import (
    build_skill_vocabulary,
    build_title_skill_profiles,
    extract_candidate_skills,
    extract_years_of_experience,
    get_current_title,
    infer_seniority,
    score_reachable_titles,
    deduplicate_titles,
    enrich_with_market_data,
)
from models.model import model



parser = ResumeParser(static_folder="static")


# Extract the data from the resume
data = parser.parse_resume("Resume Full Time ML V260.pdf")


# Create a string containing info of the resume
resume_text = build_resume_text(data)



# Load job dataset
jobs_path = "data/jobs2.json"

with open(jobs_path, "r") as f:
    jobs = json.load(f)
    
# print(f"Total jobs loaded: {len(jobs)}")

job_texts = [build_job_text(job) for job in jobs]
# print(job_texts[0][:500])  # preview first job


resume_embedding = model.encode(resume_text)
job_embeddings = model.encode(job_texts)

# Compute cosine similarity
similarities = cosine_similarity(
    [resume_embedding],
    job_embeddings
)[0]  # extract row

# Attach similarity scores to jobs
for i, job in enumerate(jobs):
    job["semantic_score"] = float(similarities[i])
    
# Sort jobs by similarity descending
ranked_jobs = sorted(jobs, key=lambda x: x["semantic_score"], reverse=True)    

    

n = len(ranked_jobs)
top_n_jobs = ranked_jobs[:n]

# print(f"\nTop {n} Jobs After Semantic Filtering:\n")

# for job in top_n_jobs:
#     print(f"{job['title']} | Score: {job['semantic_score']:.4f}") 



final_ranked_jobs = top_n_jobs



print("\nRECOMMENDED JOBS\n")

for i, job in enumerate(final_ranked_jobs):
    print("="*100)
    print(f"RANK {i+1}")
    print(f"Title: {job['title']}")
    print(f"Company: {job['company']}")
    print(f"Company: {job['job_description']}")

   
    


index = 3

# print('\n\nFinal ranked jobs \n\n')
# print(final_ranked_jobs[index])

# # Get insights for a job
# print('\n\n Summary of your skill gaps \n\n')
# identify_skill_gaps(resume_text,  final_ranked_jobs,index)

# Explain Matching Quality for a jobs
# print('\n\n Summary of your matching skills for the job \n\n')
# explain_matching_quality(resume_text,  final_ranked_jobs,index)    

job_titles, job_title_embeddings = compute_title_embeddings(jobs)
print("Total job titles embedded:", len(job_titles))

user_query = "Cloud Engineer"
     

matched_jobs = find_similar_jobs(
    user_query,
    jobs,
    job_title_embeddings
)     

print("Matched Jobs:")
print("--------------------------------")

for job in matched_jobs:
    print(
        f"{job['title']} | {job['company']} | similarity={job['title_similarity']:.2f}"
    )

top_skills = get_top_skills(matched_jobs, top_n=10)

print("\nTop In-Demand Skills")
print("--------------------------------")

for rank, (skill, count) in enumerate(top_skills, 1):

    print(f"{rank}. {skill} ({count} jobs)")    
    
    
def semantic_recommendation(resume_text, jobs):
    """
    Reuse existing semantic matching logic
    """

    job_texts = [build_job_text(job) for job in jobs]

    resume_embedding = model.encode(resume_text)
    job_embeddings = model.encode(job_texts)

    similarities = cosine_similarity(
        [resume_embedding],
        job_embeddings
    )[0]

    for i, job in enumerate(jobs):
        job["semantic_score"] = float(similarities[i])

    ranked_jobs = sorted(
        jobs,
        key=lambda x: x["semantic_score"],
        reverse=True
    )

    return ranked_jobs    



def analyze_job_title(user_query, jobs):
    """
    Given a job title, return:
    - matched jobs
    - top in-demand skills
    """

    job_titles, job_title_embeddings = compute_title_embeddings(jobs)

    matched_jobs = find_similar_jobs(
        user_query,
        jobs,
        job_title_embeddings
    )

    top_skills = get_top_skills(matched_jobs, top_n=15)

    return matched_jobs, top_skills



from collections import defaultdict, Counter
from datetime import datetime



    
    
def get_skill_trends(jobs, job_title, top_n=5):
    year_skill_map = defaultdict(list)
    
    matched_jobs = find_similar_jobs(
        job_title,
        jobs,
        job_title_embeddings
    )   

    # Filter relevant jobs
    for job in matched_jobs:
        date_str = job.get("posting_date")

        if not date_str:
            continue

        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
        except:
            continue

        skills = job.get("skills_required", [])
        year_skill_map[year].extend(skills)

    # Count + normalize
    trend_data = {}
    all_skills_counter = Counter()

    # First pass: count total occurrences
    for year, skills in year_skill_map.items():
        counter = Counter(skills)
        trend_data[year] = counter
        all_skills_counter.update(counter)

    # Get global top N skills
    top_skills = [skill for skill, _ in all_skills_counter.most_common(top_n)]

    # Build final structured output
    final_trend = {}

    for year in sorted(trend_data.keys()):
        year_counts = trend_data[year]
        total_jobs = sum(year_counts.values()) or 1

        final_trend[year] = {
            skill: round((year_counts.get(skill, 0) / total_jobs) * 100, 2)
            for skill in top_skills
        }

    return {
        "years": sorted(final_trend.keys()),
        "skills": top_skills,
        "data": final_trend
    }    
    
    
    
from collections import defaultdict, Counter
from datetime import datetime




