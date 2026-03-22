from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import json
import pdfplumber
import io
import markdown
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64


from functions.parse_resume import ResumeParser, build_resume_text
from functions.model import search_jobs   # your embedding search
from main import semantic_recommendation, analyze_job_title
from functions.llm_recommendations import explain_matching_quality

# resume_text_global = None

app = FastAPI()

templates = Jinja2Templates(directory="templates")

parser = ResumeParser(static_folder="static")

# UPLOAD_DIR = "static/resumes"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load jobs
with open("data/jobs2.json") as f:
    jobs = json.load(f)
    

def get_resume_text():
    if not os.path.exists(RESUME_PATH):
        return None

    try:
        parser_path = os.path.join("uploads", "resume.pdf")
        parsed_data = parser.parse_resume(parser_path)

        if not parsed_data:
            return None

        return build_resume_text(parsed_data)

    except Exception as e:
        print("Error reading resume:", e)
        return None    
    

# -------------------------------
# 1️⃣ HOME PAGE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index_new.html", {"request": request})


# -------------------------------
# 2️⃣ SEARCH RESULTS
# -------------------------------
@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    results = search_jobs(query, jobs)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "jobs": results,
        "query": query,
        "recommended": False
    })



# -------------------------------
# 3️⃣ PROFILE (Resume Upload + Parse)
# -------------------------------
from fastapi import UploadFile, File

@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "parsed": None
    })


UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESUME_PATH = os.path.join(UPLOAD_DIR, "resume.pdf")


@app.post("/profile", response_class=HTMLResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)):

    try:
        filename = file.filename

        # ✅ Full path (for saving)
        # RESUME_PATH = os.path.join(UPLOAD_DIR, "resume.pdf")
        # full_path = os.path.join(UPLOAD_DIR, filename)
        
        full_path = RESUME_PATH

        # ✅ Read file (ONLY valid inside async function)
        content = await file.read()

        with open(full_path, "wb") as f:
            f.write(content)

        # ✅ Pass relative path to parser
        parser_path = os.path.join("uploads", filename)

        parsed_data = parser.parse_resume(parser_path)
        # print('Parsed Data : ',parsed_data)
        
        # global resume_text_global
        # resume_text_global = build_resume_text(parsed_data)

        if not parsed_data:
            raise ValueError("Parsing failed")

    except Exception as e:
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "error": str(e),
            "parsed": None
        })

    # print('Parsed Data : ',parsed_data)
    
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "parsed": parsed_data
    })
    
    
    
# -------------------------------
# 4 RECOMMENDED (SEMANTIC MATCH)
# -------------------------------
@app.post("/recommend", response_class=HTMLResponse)
def recommend(request: Request, query: str = Form(...)):

    # global resume_text_global
    
    # print(f'Resume text : {resume_text_global}')
    
    resume_text = get_resume_text()

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": query,
            "recommended": True,
            "error": "Upload resume first"
        })

    # Step 1: filter jobs (you already have this)
    filtered_jobs = search_jobs(query, jobs)

    # Step 2: reuse your semantic logic
    recommended_jobs = semantic_recommendation(
        resume_text,
        filtered_jobs
    )

    return templates.TemplateResponse("results.html", {
        "request": request,
        "jobs": recommended_jobs,
        "query": query,
        "recommended": True
    })
    
    
    
    
@app.post("/fit", response_class=HTMLResponse)
def find_fit(request: Request, job_id: str = Form(...)):

    # Step 1: Get resume
    resume_text = get_resume_text()

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": "",
            "recommended": True,
            "error": "Please upload resume first"
        })

    # Step 2: Find job using job_id
    selected_job = None
    selected_index = None

    for i, job in enumerate(jobs):
        if str(job.get("job_id")) == str(job_id):
            selected_job = job
            selected_index = i
            break

    if not selected_job:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": "",
            "recommended": True,
            "error": "Job not found"
        })
    
    # print('selected job index : ',selected_index)    
    
    # Step 3: Call LLM function
    explanation = explain_matching_quality(
        resume_text,
        [selected_job],   
        1                 
    )
    
    print('Job fit explaination : ',explanation)
    
    explanation = markdown.markdown(explanation, extensions=["tables"])

    # Step 4: Show result
    return templates.TemplateResponse("fit.html", {
        "request": request,
        "job": selected_job,
        "explanation": explanation
    })
    
    
    
    
# @app.post("/title_summary", response_class=HTMLResponse)
# def title_summary(request: Request, query: str = Form(...)):

#         matched_jobs, top_skills = analyze_job_title(query, jobs)

#         return templates.TemplateResponse("title_summary.html", {
#             "request": request,
#             "query": query,
#             "matched_jobs": matched_jobs,
#             "top_skills": top_skills
#         })


@app.post("/title_summary", response_class=HTMLResponse)
def title_summary(request: Request, query: str = Form(...)):

    matched_jobs, top_skills = analyze_job_title(query, jobs)

    # Convert to dict for wordcloud
    skill_freq = {skill: count for skill, count in top_skills}

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(skill_freq)

    # Convert to image
    img_buffer = io.BytesIO()
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    plt.close()

    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    return templates.TemplateResponse("title_summary copy.html", {
        "request": request,
        "query": query,
        "matched_jobs": matched_jobs,
        "top_skills": top_skills,
        "wordcloud": img_base64
    })