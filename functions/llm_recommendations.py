from functions.llm import llm_call
from collections import Counter
import json as _json



def identify_skill_gaps(resume_text, final_ranked_jobs, job_index):
    """
    Shows skill gaps and generates improvement roadmap
    for a specific selected job.
    """

    # --- Validation ---
    if job_index < 1 or job_index > len(final_ranked_jobs):
        print("❌ Invalid job index.")
        return

    selected_job = final_ranked_jobs[job_index - 1]
    
    
    
    job_title = selected_job.get("title")
    company = selected_job.get("company")
    job_description = selected_job.get("job_description")
    experience = selected_job.get("experience_required")
    education = selected_job.get("education_required")


    # --- Generate Roadmap ---
    system_prompt = """
                    You are a senior career advisor AI.

                    Idenfy any missing skill gap or experience gap for a specific job, create:

                    0. The skills missing in the form of a list  
                    1. Priority order of skills to learn
                    2. Practical learning roadmap (3-6 months)
                    3. Suggested project ideas to strengthen profile
                    4. Industry-relevant advice

                    Be structured and practical.
                    """

    user_prompt = f"""
                    Candidate Resume:
                    {resume_text}

                    Selected Job:
                    Title: {job_title}
                    Company: {company}
                    Description: {selected_job.get('job_description')}
                    Required Skills: {selected_job.get('skills_required')}
                    Required Experience:  {job_description}
                    Required Education: {education}
                    Requireed Experience: {experience}


                    And Create a focused improvement roadmap for THIS job only.
                    """

    # roadmap = llm_call(resume_text, selected_job, system_prompt, user_prompt)
    roadmap = llm_call(system_prompt, user_prompt)

    print("\n Improvement Roadmap:\n")
    # print(roadmap['reasoning'])
    print('Roadmap : \n',roadmap)
    
    
    
    

def explain_matching_quality(resume_text, final_ranked_jobs, job_index):

    # summary = []

     # --- Validation ---
    if job_index < 1 or job_index > len(final_ranked_jobs):
        print("❌ Invalid job index.")
        return

    selected_job = final_ranked_jobs[job_index - 1]
    
    
    
    job_title = selected_job.get("title")
    company = selected_job.get("company")
    job_description = selected_job.get("job_description")
    experience = selected_job.get("experience_required")
    education = selected_job.get("education_required")

    system_prompt = """
                    You are a career analytics AI.
                    Explain how well the candidate matches these jobs.
                    Highlight patterns in strengths and highlight similar work that the candidate did from the candidate's resume.
                    Be analytical and clear.
                    """

    user_prompt = f"""
                    Candidate Resume:
                    {resume_text}

                    Selected Job:
                    Title: {job_title}
                    Company: {company}
                    Description: {selected_job.get('job_description')}
                    Required Skills: {selected_job.get('skills_required')}
                    Required Experience:  {job_description}
                    Required Education: {education}
                    Requireed Experience: {experience}
                    
                    Highlight patterns in strengths and highlight similar work that the candidate did from the candidate's resume.
                    """

    # explanation = llm_call(resume_text, selected_job, system_prompt, user_prompt)
    explanation = llm_call(system_prompt, user_prompt)

    return explanation 




 
def generate_career_narrative(candidate: dict, recommendations: list[dict],) -> dict:
    """
    Ask the LLM to produce:
      1. A one-sentence "why it's a natural next step" for each recommended role.
      2. A one-sentence "what you'd gain" for each role.
      3. A 2-sentence overall career trajectory summary.
 
    Returns a parsed dict with keys:
        {
          "roles": { "<title>": {"why": "...", "gain": "..."}, ... },
          "trajectory": "..."
        }
 
    On any failure returns a safe fallback dict so the page always renders.
 
    llm_call signature (from llm.py):
        llm_call(resume_text, job, system_prompt, user_prompt, expect_json=False)
 
    We map:
        resume_text   <- serialised candidate profile string
        job           <- empty dict  (not a job-match call)
        system_prompt <- role/instruction
        user_prompt   <- the structured data + JSON format spec
        expect_json   <- True  (llm.py will parse and return a dict directly)
    """
 
    roles_block = "\n".join(
        f"- {rec['title']} (gap skills: {', '.join(rec['gap_skills'][:4])})"
        for rec in recommendations
    )
 
    # Serialise candidate profile into a plain string for the resume_text arg
    candidate_summary = (
        f"Current role: {candidate['current_title'] or 'Not specified'}\n"
        f"Years of experience: {candidate['years_exp']}\n"
        f"Seniority level: {candidate['seniority']}\n"
        f"Key skills: {', '.join(candidate['skills'][:15]) or 'Not specified'}"
    )
 
    system_prompt = (
        "You are a senior career advisor. "
        "Given a candidate profile and a list of recommended next-step roles, "
        "you write concise, honest, and motivating career guidance. "
        "You always respond in valid JSON — no markdown, no extra text."
    )
 
    user_prompt = f"""Candidate profile:
            {candidate_summary}
            
            Recommended next-step roles (with the skills the candidate still needs to acquire):
            {roles_block}
            
            For EACH recommended role write:
            - "why": one sentence explaining why this is a natural next step for THIS candidate
            - "gain": one sentence on the scope, impact, or salary uplift this move brings
            
            Then write:
            - "trajectory": a 2-sentence overall career trajectory narrative for this candidate
            
            Respond ONLY with valid JSON in exactly this structure:
            {{
            "roles": {{
                "<role title>": {{"why": "...", "gain": "..."}},
                ...
            }},
            "trajectory": "..."
            }}"""
 
    try:
        result = llm_call(
            system_prompt = system_prompt,
            user_prompt   = user_prompt,
            expect_json   = True,
        )
 
        # llm_call returns a dict when expect_json=True and parsing succeeds,
        # or {"raw_output": "..."} when JSON parsing fails inside llm.py.
        if isinstance(result, dict) and "raw_output" not in result:
            return result
 
        # Fallback: llm_call returned {"raw_output": ...} — JSON parse failed
        raise ValueError(f"LLM returned unparseable JSON: {result.get('raw_output', '')[:120]}")
 
    except Exception as e:
        print(f"[generate_career_narrative] LLM call failed: {e}")
        return _career_narrative_fallback(candidate, recommendations)
 
 
def _career_narrative_fallback(candidate: dict, recommendations: list[dict]) -> dict:
    """Return a deterministic fallback narrative when the LLM call fails."""
    return {
        "roles": {
            rec["title"]: {
                "why": (
                    f"This role is a natural progression from your experience "
                    f"as a {candidate['current_title'] or 'professional'}."
                ),
                "gain": (
                    f"Acquiring {', '.join(rec['gap_skills'][:2]) or 'these skills'} "
                    f"would unlock this opportunity and broaden your impact."
                ),
            }
            for rec in recommendations
        },
        "trajectory": (
            f"Based on your {candidate['years_exp']} years of experience as a "
            f"{candidate['current_title'] or 'professional'}, you are well positioned "
            f"to move into more senior roles. "
            f"Focus on closing the identified skill gaps to accelerate your progression."
        ),
    }
 