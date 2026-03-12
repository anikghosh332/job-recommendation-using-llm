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