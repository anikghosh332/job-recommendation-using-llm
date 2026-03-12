from functions.parse_resume import ResumeParser, build_resume_text
from functions.parse_jobs import build_job_text
from functions.model import model
import json
from sklearn.metrics.pairwise import cosine_similarity
from functions.llm import llm_call
from functions.llm_recommendations import identify_skill_gaps, explain_matching_quality


parser = ResumeParser(static_folder="static")


# Extract the data from the resume
data = parser.parse_resume("Resume Full Time ML V260.pdf")

# print(data["education"])
# print(data["work_experience"])
# print(data["projects"])

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

   
    
    
# required_vars = ["data", "resume_text", "final_ranked_jobs"]
# missing = [var for var in required_vars if var not in globals()]

# if missing:
#     raise ValueError(f"Missing required variables: {missing}. Run ranking pipeline first.")
# if not final_ranked_jobs:
#     raise ValueError("final_ranked_jobs is empty.")
# print("Required data validated successfully.")    


index = 3

print('\n\nFinal ranked jobs \n\n')
print(final_ranked_jobs[index])

# Get insights for a job
print('\n\n Summary of your skill gaps \n\n')
identify_skill_gaps(resume_text,  final_ranked_jobs,index)

# Explain Matching Quality for a jobs
print('\n\n Summary of your matching skills for the job \n\n')
explain_matching_quality(resume_text,  final_ranked_jobs,index)    
     