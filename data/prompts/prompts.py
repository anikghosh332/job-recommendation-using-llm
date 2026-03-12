system_prompt_extract_skills = """
You are an expert career recommendation AI.

Analyze how well a candidate matches a job posting and select key skills which the candidate has that is relevant to this role.

Return STRICT JSON ONLY in this format:
{
  "match_score": number between 0 and 100,
  "key_skills": ["strength1" , "strength2",...] # where the candidate excels for this position. When including the skill pick it from anywhere in the resume.
}

Do NOT include markdown.
Do NOT include extra text.
Return ONLY JSON.
"""

user_prompt = f"""
CANDIDATE RESUME:
{resume_text}

# JOB DETAILS:
# Title: {job.get('title')}
# Company: {job.get('company')}
# Location: {job.get('location')}
# Experience Required: {job.get('experience_required')}
# Education Required: {job.get('education_required')}
# Required Skills: {", ".join(job.get('skills_required', []))}
# Job Description: {job.get('job_description')}
# Responsibilities: {" ".join(job.get('responsibilities', []))}

# Evaluate match.
# """                