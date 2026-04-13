from functions.parse_resume import ResumeParser, build_resume_text
from functions.parse_jobs import build_job_text, compute_title_embeddings, find_similar_jobs, get_top_skills, parse_salary
from functions.model import model
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

def compute_skill_trends(jobs, query, top_n=5):
    year_skill_map = defaultdict(list)

    for job in jobs:
        if job.get("title", "").lower() == query.lower():
            date_str = job.get("posting_date")
            if not date_str:
                continue

            try:
                year = datetime.strptime(date_str, "%Y-%m-%d").year
            except:
                continue

            year_skill_map[year].extend(job.get("skills_required", []))

    # Count skills
    yearly_counts = {}
    global_counter = Counter()

    for year, skills in year_skill_map.items():
        counter = Counter(skills)
        yearly_counts[year] = counter
        global_counter.update(counter)

    # Top N skills overall
    top_skills = [s for s, _ in global_counter.most_common(top_n)]

    # Build final structure
    years_sorted = sorted(yearly_counts.keys())
    trend_data = {}

    for year in years_sorted:
        total = sum(yearly_counts[year].values()) or 1
        trend_data[year] = {
            skill: round((yearly_counts[year].get(skill, 0) / total) * 100, 2)
            for skill in top_skills
        }

    return years_sorted, top_skills, trend_data  






def compute_salary_trends(matched_jobs) -> tuple:
    """
    Given already-matched jobs (from analyze_job_title / find_similar_jobs),
    compute average salary midpoint per year.

    Returns:
        salary_years   - sorted list of years (int)
        salary_by_year - dict { year: rounded avg midpoint }
        current_salary - avg salary for the most recent year, or None
    """
    year_salaries = defaultdict(list)

    for job in matched_jobs:
        date_str = job.get("posting_date")
        if not date_str:
            continue
        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
        except Exception:
            continue

        midpoint = parse_salary(job.get("salary_range"))
        if midpoint is None:
            continue

        year_salaries[year].append(midpoint)

    if not year_salaries:
        return [], {}, None

    salary_years = sorted(year_salaries.keys())
    salary_by_year = {
        year: int(round(sum(vals) / len(vals)))
        for year, vals in year_salaries.items()
    }
    current_salary = salary_by_year[salary_years[-1]]

    return salary_years, salary_by_year, current_salary 




def get_career_recommendations(
    resume_text:   str,
    parsed_resume: dict,
    jobs:          list[dict],
    top_n:         int = 4,
) -> dict:
    """
    Full career trajectory pipeline.
 
    Args:
        resume_text:   Plain text resume (from build_resume_text).
        parsed_resume: Structured resume dict (from ResumeParser.parse_resume).
        jobs:          Full job dataset list.
        top_n:         Number of role recommendations to return.
 
    Returns a dict with keys:
        candidate       – profile summary (title, seniority, years_exp, skills)
        recommendations – list of enriched, deduplicated role dicts
        narrative       – LLM-generated growth summary string
    """
    from functions.llm_recommendations import generate_career_narrative
 
    work_exp = parsed_resume.get("work_experience", [])
 
    # ── Step 1: Candidate profile ────────────────────────────────────────────
    current_title  = get_current_title(work_exp)
    years_exp      = extract_years_of_experience(work_exp)
    seniority      = infer_seniority(current_title)
    skill_vocab    = build_skill_vocabulary(jobs)
    candidate_skills = extract_candidate_skills(resume_text, skill_vocab)
 
    candidate = {
        "current_title":   current_title,
        "years_exp":       years_exp,
        "seniority":       seniority,
        "skills":          sorted(candidate_skills),
    }
 
    # ── Step 2: Score reachable titles ───────────────────────────────────────
    title_profiles = build_title_skill_profiles(jobs, min_frequency=0.25)
 
    scored = score_reachable_titles(
        candidate_skills    = candidate_skills,
        title_skill_profiles = title_profiles,
        current_title       = current_title,
        candidate_seniority = seniority,
        min_gap             = 1,
        max_gap             = 5,
    )
 
    if not scored:
        return {
            "candidate":       candidate,
            "recommendations": [],
            "narrative":       "Not enough data to generate career recommendations.",
        }
 
    # ── Step 3: Deduplicate and take top N ───────────────────────────────────
    recommendations = deduplicate_titles(scored, model=model, top_n=top_n)
 
    # ── Step 4: Enrich with live market data ─────────────────────────────────
    recommendations = enrich_with_market_data(recommendations, jobs)
 
    # ── Step 5: LLM narrative ────────────────────────────────────────────────
    narrative = generate_career_narrative(candidate, recommendations)
 
    # Convert sets to sorted lists for JSON serialisation / template rendering
    for rec in recommendations:
        rec["required_skills"] = sorted(rec["required_skills"])
        rec["owned_skills"]    = sorted(rec["owned_skills"])
 
    return {
        "candidate":       candidate,
        "recommendations": recommendations,
        "narrative":       narrative,
    }