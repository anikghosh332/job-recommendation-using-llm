from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
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
from datetime import datetime


from functions.parse_resume import ResumeParser, build_resume_text
from functions.model import search_jobs
from main import semantic_recommendation, analyze_job_title, compute_skill_trends
from functions.llm_recommendations import explain_matching_quality

app = FastAPI()

templates = Jinja2Templates(directory="templates")

parser = ResumeParser(static_folder="static")

# Load jobs
with open("data/jobs3.json") as f:
    jobs = json.load(f)


# ─────────────────────────────────────────────
# MULTI-RESUME STORAGE SETUP
# ─────────────────────────────────────────────

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

META_PATH = os.path.join(UPLOAD_DIR, "resumes_meta.json")


def load_meta() -> dict:
    """Load resume metadata. Returns dict with 'resumes' list and 'active' filename."""
    if not os.path.exists(META_PATH):
        return {"resumes": [], "active": None}
    with open(META_PATH, "r") as f:
        return json.load(f)


def save_meta(meta: dict):
    """Persist resume metadata to disk."""
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def get_resume_text() -> str | None:
    """Return parsed text of the currently active resume, or None."""
    meta = load_meta()
    active = meta.get("active")

    if not active:
        return None

    full_path = os.path.join(UPLOAD_DIR, active)
    if not os.path.exists(full_path):
        return None

    try:
        # parser expects path relative to static folder
        relative_path = os.path.join("uploads", active)
        parsed_data = parser.parse_resume(relative_path)
        if not parsed_data:
            return None
        return build_resume_text(parsed_data)
    except Exception as e:
        print("Error reading resume:", e)
        return None


# ─────────────────────────────────────────────
# 1️⃣  HOME PAGE
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index_new.html", {"request": request})


# ─────────────────────────────────────────────
# 2️⃣  SEARCH RESULTS
# ─────────────────────────────────────────────

@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    results = search_jobs(query, jobs)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "jobs": results,
        "query": query,
        "recommended": False
    })


# ─────────────────────────────────────────────
# 3️⃣  PROFILE — list resumes + upload
# ─────────────────────────────────────────────

@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    meta = load_meta()
    parsed = None

    if meta.get("active"):
        try:
            relative_path = os.path.join("uploads", meta["active"])
            parsed = parser.parse_resume(relative_path)
        except Exception as e:
            print("Error parsing active resume:", e)

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "resumes": meta["resumes"],
        "active": meta["active"],
        "parsed": parsed,
        "error": None,
    })


@app.post("/profile", response_class=HTMLResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)):
    meta = load_meta()

    try:
        original_name = file.filename
        # Create a unique filename: timestamp + original name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = original_name.replace(" ", "_")
        unique_name = f"{timestamp}_{safe_name}"

        full_path = os.path.join(UPLOAD_DIR, unique_name)

        content = await file.read()
        with open(full_path, "wb") as f:
            f.write(content)

        # Parse immediately to validate
        relative_path = os.path.join("uploads", unique_name)
        parsed_data = parser.parse_resume(relative_path)

        if not parsed_data:
            raise ValueError("Parsing failed — check file format.")

        # Add to metadata
        entry = {
            "filename": unique_name,
            "display_name": original_name,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        meta["resumes"].append(entry)

        # Auto-select if this is the first resume
        if meta["active"] is None:
            meta["active"] = unique_name

        save_meta(meta)

    except Exception as e:
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "resumes": meta["resumes"],
            "active": meta["active"],
            "error": str(e),
            "parsed": None,
        })

    # Reload meta after save and show parsed result
    meta = load_meta()
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "resumes": meta["resumes"],
        "active": meta["active"],
        "parsed": parsed_data,
        "error": None,
    })


# ─────────────────────────────────────────────
# 4️⃣  SELECT ACTIVE RESUME
# ─────────────────────────────────────────────

@app.post("/profile/select", response_class=HTMLResponse)
def select_resume(request: Request, filename: str = Form(...)):
    meta = load_meta()

    # Validate the filename exists in our list
    known = [r["filename"] for r in meta["resumes"]]
    if filename in known:
        meta["active"] = filename
        save_meta(meta)

    return RedirectResponse(url="/profile", status_code=303)


# ─────────────────────────────────────────────
# 5️⃣  DELETE A RESUME
# ─────────────────────────────────────────────

@app.post("/profile/delete", response_class=HTMLResponse)
def delete_resume(request: Request, filename: str = Form(...)):
    meta = load_meta()

    # Remove from list
    meta["resumes"] = [r for r in meta["resumes"] if r["filename"] != filename]

    # If deleted the active one, pick the first remaining (or None)
    if meta["active"] == filename:
        meta["active"] = meta["resumes"][0]["filename"] if meta["resumes"] else None

    # Delete file from disk
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    save_meta(meta)
    return RedirectResponse(url="/profile", status_code=303)


# ─────────────────────────────────────────────
# 6️⃣  RECOMMEND (SEMANTIC MATCH)
# ─────────────────────────────────────────────

@app.post("/recommend", response_class=HTMLResponse)
def recommend(request: Request, query: str = Form(...)):
    resume_text = get_resume_text()

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": query,
            "recommended": True,
            "error": "Please upload and select a resume first."
        })

    filtered_jobs = search_jobs(query, jobs)
    recommended_jobs = semantic_recommendation(resume_text, filtered_jobs)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "jobs": recommended_jobs,
        "query": query,
        "recommended": True
    })


# ─────────────────────────────────────────────
# 7️⃣  JOB FIT EXPLANATION
# ─────────────────────────────────────────────

@app.post("/fit", response_class=HTMLResponse)
def find_fit(request: Request, job_id: str = Form(...)):
    resume_text = get_resume_text()

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": "",
            "recommended": True,
            "error": "Please upload and select a resume first."
        })

    selected_job = None
    for job in jobs:
        if str(job.get("job_id")) == str(job_id):
            selected_job = job
            break

    if not selected_job:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "jobs": [],
            "query": "",
            "recommended": True,
            "error": "Job not found."
        })

    explanation = explain_matching_quality(resume_text, [selected_job], 1)
    explanation = markdown.markdown(explanation, extensions=["tables"])

    return templates.TemplateResponse("fit.html", {
        "request": request,
        "job": selected_job,
        "explanation": explanation
    })


# ─────────────────────────────────────────────
# 8️⃣  TITLE SUMMARY + SKILL TRENDS
# ─────────────────────────────────────────────

@app.post("/title_summary", response_class=HTMLResponse)
def title_summary(request: Request, query: str = Form(...)):
    matched_jobs, top_skills = analyze_job_title(query, jobs)

    skill_freq = {skill: count for skill, count in top_skills}

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_freq)

    img_buffer = io.BytesIO()
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    plt.close()

    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    years, trend_skills, trend_data = compute_skill_trends(jobs, query, top_n=6)

    return templates.TemplateResponse("title_summary.html", {
        "request": request,
        "query": query,
        "matched_jobs": matched_jobs,
        "top_skills": top_skills,
        "wordcloud": img_base64,
        "trend_years": years,
        "trend_skills": trend_skills,
        "trend_data": trend_data,
    })
