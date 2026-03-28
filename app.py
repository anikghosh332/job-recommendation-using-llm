from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import json
import io
import markdown
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from typing import Optional

from functions.parse_resume import ResumeParser, build_resume_text
from functions.model import search_jobs
from functions.llm_recommendations import explain_matching_quality
from functions.auth import (
    authenticate_user,
    create_user,
    get_user_by_id,
    get_user_by_email,
    make_reset_token,
    verify_reset_token,
    update_password,
    session_set_user,
    session_get_user_id,
    session_clear,
    user_upload_dir,
    user_meta_path,
)
from main import (
    semantic_recommendation,
    analyze_job_title,
    compute_skill_trends,
    compute_salary_trends,
    get_career_recommendations,
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI()

# SESSION_SECRET must be set in your environment before running.
# e.g. add to .env:  SESSION_SECRET=change-me-to-something-long-and-random
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-in-production")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=86400)

templates = Jinja2Templates(directory="templates")
parser    = ResumeParser(static_folder="static")

# ── Constants ────────────────────────────────
BASE_UPLOAD_DIR = "static/uploads"
USERS_PATH      = "data/users.json"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load jobs once at startup
with open("data/jobs3.json") as f:
    jobs = json.load(f)


# ─────────────────────────────────────────────
# AUTH DEPENDENCY HELPERS
# ─────────────────────────────────────────────

def get_current_user(request: Request) -> Optional[dict]:
    """
    Returns the logged-in user dict or None.
    Injected via Depends() into every route.
    """
    user_id = session_get_user_id(request.session)
    if not user_id:
        return None
    return get_user_by_id(user_id, USERS_PATH)


def _redirect_if_unauthenticated(user: Optional[dict]) -> Optional[RedirectResponse]:
    """
    Call at the top of every protected route.
    Returns a RedirectResponse to /login if user is None, else None.
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return None


# ─────────────────────────────────────────────
# PER-USER RESUME HELPERS
# All three functions now accept user_id so storage is fully isolated.
# ─────────────────────────────────────────────

def load_meta(user_id: str) -> dict:
    """Load resume metadata for a specific user."""
    path = user_meta_path(BASE_UPLOAD_DIR, user_id)
    if not os.path.exists(path):
        return {"resumes": [], "active": None}
    with open(path, "r") as f:
        return json.load(f)


def save_meta(meta: dict, user_id: str) -> None:
    """Persist resume metadata for a specific user."""
    path = user_meta_path(BASE_UPLOAD_DIR, user_id)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def get_resume_text(user_id: str) -> Optional[str]:
    """Return parsed text of the active resume for user_id, or None."""
    meta   = load_meta(user_id)
    active = meta.get("active")
    if not active:
        return None

    upload_dir = user_upload_dir(BASE_UPLOAD_DIR, user_id)
    full_path  = os.path.join(upload_dir, active)
    if not os.path.exists(full_path):
        return None

    try:
        relative_path = os.path.join("uploads", user_id, active)
        parsed_data   = parser.parse_resume(relative_path)
        if not parsed_data:
            return None
        return build_resume_text(parsed_data)
    except Exception as e:
        print(f"[get_resume_text] Error for user {user_id}: {e}")
        return None


# ─────────────────────────────────────────────
# 1  HOME PAGE  (public)
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: Optional[dict] = Depends(get_current_user)):
    return templates.TemplateResponse("index_new.html", {
        "request": request,
        "user":    user,
    })


# ─────────────────────────────────────────────
# 2  SEARCH RESULTS  (public)
# ─────────────────────────────────────────────

@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query:   str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    results = search_jobs(query, jobs)
    return templates.TemplateResponse("results.html", {
        "request":     request,
        "user":        user,
        "jobs":        results,
        "query":       query,
        "recommended": False,
    })


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "error":   None,
    })


@app.post("/register", response_class=HTMLResponse)
def register(
    request:          Request,
    name:             str = Form(...),
    email:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
):
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error":   "Passwords do not match.",
        })
    try:
        user = create_user(name, email, password, USERS_PATH)
    except ValueError as e:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error":   str(e),
        })

    session_set_user(request.session, user["user_id"])
    return RedirectResponse(url="/profile", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if session_get_user_id(request.session):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error":   None,
    })


@app.post("/login", response_class=HTMLResponse)
def login(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
):
    user = authenticate_user(email, password, USERS_PATH)
    if user is None:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error":   "Incorrect email or password.",
        })
    session_set_user(request.session, user["user_id"])
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    session_clear(request.session)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {
        "request":   request,
        "message":   None,
        "reset_url": None,
        "error":     None,
    })


@app.post("/forgot-password", response_class=HTMLResponse)
def forgot_password(request: Request, email: str = Form(...)):
    """
    Generate a signed reset token and display the reset link on-screen.
    In production replace this with an email send using your mail provider.
    """
    user      = get_user_by_email(email, USERS_PATH)
    reset_url = None

    if user:
        token     = make_reset_token(email, SESSION_SECRET)
        reset_url = f"/reset-password?token={token}"

    # Same message regardless of whether account exists — prevents enumeration.
    return templates.TemplateResponse("forgot_password.html", {
        "request":   request,
        "message":   "If an account exists for that email a reset link has been generated below.",
        "reset_url": reset_url,
        "error":     None,
    })


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(request: Request, token: str):
    email = verify_reset_token(token, SESSION_SECRET)
    if email is None:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token":   token,
            "error":   "This reset link has expired or is invalid. Please request a new one.",
            "success": False,
        })
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token":   token,
        "error":   None,
        "success": False,
    })


@app.post("/reset-password", response_class=HTMLResponse)
def reset_password(
    request:          Request,
    token:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
):
    email = verify_reset_token(token, SESSION_SECRET)
    if email is None:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token":   token,
            "error":   "This reset link has expired or is invalid.",
            "success": False,
        })

    if password != confirm_password:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token":   token,
            "error":   "Passwords do not match.",
            "success": False,
        })

    try:
        update_password(email, password, USERS_PATH)
    except ValueError as e:
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "token":   token,
            "error":   str(e),
            "success": False,
        })

    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token":   "",
        "error":   None,
        "success": True,
    })


# ─────────────────────────────────────────────
# 3  PROFILE  (protected)
# ─────────────────────────────────────────────

@app.get("/profile", response_class=HTMLResponse)
def profile_page(
    request: Request,
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    meta   = load_meta(user["user_id"])
    parsed = None

    if meta.get("active"):
        try:
            relative_path = os.path.join("uploads", user["user_id"], meta["active"])
            parsed        = parser.parse_resume(relative_path)
        except Exception as e:
            print(f"[profile_page] Parse error: {e}")

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user":    user,
        "resumes": meta["resumes"],
        "active":  meta["active"],
        "parsed":  parsed,
        "error":   None,
    })


@app.post("/profile", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    file:    UploadFile = File(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    user_id    = user["user_id"]
    upload_dir = user_upload_dir(BASE_UPLOAD_DIR, user_id)
    meta       = load_meta(user_id)

    try:
        original_name = file.filename
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name     = original_name.replace(" ", "_")
        unique_name   = f"{timestamp}_{safe_name}"

        full_path = os.path.join(upload_dir, unique_name)
        content   = await file.read()
        with open(full_path, "wb") as f:
            f.write(content)

        relative_path = os.path.join("uploads", user_id, unique_name)
        parsed_data   = parser.parse_resume(relative_path)

        if not parsed_data:
            raise ValueError("Parsing failed — check file format.")

        entry = {
            "filename":     unique_name,
            "display_name": original_name,
            "uploaded_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        meta["resumes"].append(entry)

        if meta["active"] is None:
            meta["active"] = unique_name

        save_meta(meta, user_id)

    except Exception as e:
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "user":    user,
            "resumes": meta["resumes"],
            "active":  meta["active"],
            "error":   str(e),
            "parsed":  None,
        })

    meta = load_meta(user_id)
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user":    user,
        "resumes": meta["resumes"],
        "active":  meta["active"],
        "parsed":  parsed_data,
        "error":   None,
    })


# ─────────────────────────────────────────────
# 4  SELECT ACTIVE RESUME  (protected)
# ─────────────────────────────────────────────

@app.post("/profile/select", response_class=HTMLResponse)
def select_resume(
    request:  Request,
    filename: str = Form(...),
    user:     Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    user_id = user["user_id"]
    meta    = load_meta(user_id)
    known   = [r["filename"] for r in meta["resumes"]]

    if filename in known:
        meta["active"] = filename
        save_meta(meta, user_id)

    return RedirectResponse(url="/profile", status_code=303)


# ─────────────────────────────────────────────
# 5  DELETE A RESUME  (protected)
# ─────────────────────────────────────────────

@app.post("/profile/delete", response_class=HTMLResponse)
def delete_resume(
    request:  Request,
    filename: str = Form(...),
    user:     Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    user_id    = user["user_id"]
    upload_dir = user_upload_dir(BASE_UPLOAD_DIR, user_id)
    meta       = load_meta(user_id)

    meta["resumes"] = [r for r in meta["resumes"] if r["filename"] != filename]

    if meta["active"] == filename:
        meta["active"] = meta["resumes"][0]["filename"] if meta["resumes"] else None

    file_path = os.path.join(upload_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    save_meta(meta, user_id)
    return RedirectResponse(url="/profile", status_code=303)


# ─────────────────────────────────────────────
# 6  RECOMMEND  (protected)
# ─────────────────────────────────────────────

@app.post("/recommend", response_class=HTMLResponse)
def recommend(
    request: Request,
    query:   str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    resume_text = get_resume_text(user["user_id"])

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request":     request,
            "user":        user,
            "jobs":        [],
            "query":       query,
            "recommended": True,
            "error":       "Please upload and select a resume first.",
        })

    filtered_jobs    = search_jobs(query, jobs)
    recommended_jobs = semantic_recommendation(resume_text, filtered_jobs)

    return templates.TemplateResponse("results.html", {
        "request":     request,
        "user":        user,
        "jobs":        recommended_jobs,
        "query":       query,
        "recommended": True,
    })


# ─────────────────────────────────────────────
# 7  JOB FIT EXPLANATION  (protected)
# ─────────────────────────────────────────────

@app.post("/fit", response_class=HTMLResponse)
def find_fit(
    request: Request,
    job_id:  str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    resume_text = get_resume_text(user["user_id"])

    if not resume_text:
        return templates.TemplateResponse("results.html", {
            "request":     request,
            "user":        user,
            "jobs":        [],
            "query":       "",
            "recommended": True,
            "error":       "Please upload and select a resume first.",
        })

    selected_job = next(
        (j for j in jobs if str(j.get("job_id")) == str(job_id)),
        None,
    )

    if not selected_job:
        return templates.TemplateResponse("results.html", {
            "request":     request,
            "user":        user,
            "jobs":        [],
            "query":       "",
            "recommended": True,
            "error":       "Job not found.",
        })

    explanation = explain_matching_quality(resume_text, [selected_job], 1)
    explanation = markdown.markdown(explanation, extensions=["tables"])

    return templates.TemplateResponse("fit.html", {
        "request":     request,
        "user":        user,
        "job":         selected_job,
        "explanation": explanation,
    })


# ─────────────────────────────────────────────
# 8  TITLE SUMMARY + SKILL + SALARY TRENDS  (public)
# ─────────────────────────────────────────────

@app.post("/title_summary", response_class=HTMLResponse)
def title_summary(
    request: Request,
    query:   str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    matched_jobs, top_skills = analyze_job_title(query, jobs)

    skill_freq = {skill: count for skill, count in top_skills}
    wordcloud  = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(skill_freq)

    img_buffer = io.BytesIO()
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(img_buffer, format="png")
    plt.close()
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    years, trend_skills, trend_data               = compute_skill_trends(jobs, query, top_n=6)
    salary_years, salary_by_year, current_salary  = compute_salary_trends(matched_jobs)

    return templates.TemplateResponse("title_summary.html", {
        "request":        request,
        "user":           user,
        "query":          query,
        "matched_jobs":   matched_jobs,
        "top_skills":     top_skills,
        "wordcloud":      img_base64,
        "trend_years":    years,
        "trend_skills":   trend_skills,
        "trend_data":     trend_data,
        "salary_years":   salary_years,
        "salary_by_year": salary_by_year,
        "current_salary": current_salary,
    })


# ─────────────────────────────────────────────
# 9  CAREER TRAJECTORY  (protected)
# ─────────────────────────────────────────────

@app.get("/career", response_class=HTMLResponse)
def career_page(
    request: Request,
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    meta = load_meta(user["user_id"])

    if not meta.get("active"):
        return templates.TemplateResponse("career.html", {
            "request":         request,
            "user":            user,
            "error":           "Please upload and select a resume on your Profile page first.",
            "candidate":       None,
            "recommendations": [],
            "narrative":       None,
        })

    try:
        relative_path = os.path.join("uploads", user["user_id"], meta["active"])
        parsed_resume = parser.parse_resume(relative_path)
        resume_text   = build_resume_text(parsed_resume)
    except Exception as e:
        return templates.TemplateResponse("career.html", {
            "request":         request,
            "user":            user,
            "error":           f"Could not read active resume: {e}",
            "candidate":       None,
            "recommendations": [],
            "narrative":       None,
        })

    result = get_career_recommendations(
        resume_text   = resume_text,
        parsed_resume = parsed_resume,
        jobs          = jobs,
        top_n         = 4,
    )

    return templates.TemplateResponse("career.html", {
        "request":         request,
        "user":            user,
        "error":           None,
        "candidate":       result["candidate"],
        "recommendations": result["recommendations"],
        "narrative":       result["narrative"],
    })