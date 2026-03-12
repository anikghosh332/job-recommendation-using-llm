from functions.llm import llm_call
from collections import Counter



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

    roadmap = llm_call(resume_text, selected_job, system_prompt, user_prompt)

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

    explanation = llm_call(resume_text, selected_job, system_prompt, user_prompt)

    print("\n Matching Explanation:\n")
    print(explanation)    