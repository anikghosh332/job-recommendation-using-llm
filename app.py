# =============================================================================
# app.py — Workit FastAPI Application
# =============================================================================
# Route map:
#   GET  /                     → home (public)
#   POST /search               → search results (public)
#   GET  /register             → register page
#   POST /register             → create account
#   GET  /login                → login page
#   POST /login                → authenticate
#   GET  /logout               → clear session
#   GET  /forgot-password      → forgot password page
#   POST /forgot-password      → generate reset token
#   GET  /reset-password       → reset password page
#   POST /reset-password       → apply new password
#   GET  /profile              → resume manager (protected)
#   POST /profile              → upload resume (protected)
#   POST /profile/select       → set active resume (protected)
#   POST /profile/delete       → delete resume (protected)
#   POST /recommend            → AI recommendations (protected)
#   POST /fit                  → job fit explanation (protected)
#   POST /title_summary        → title analysis + charts (public)
#   GET  /career               → career trajectory (protected)
#   GET  /fit/paste            → paste-JD fit form (protected)
#   POST /fit/paste            → analyse pasted JD (protected)
#   GET  /bookmarks            → saved jobs (protected)
#   POST /bookmarks/add        → bookmark a job (protected)
#   POST /bookmarks/remove     → remove a bookmark (protected)
# =============================================================================

from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Optional

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from wordcloud import WordCloud

from functions.auth import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
    make_reset_token,
    session_clear,
    session_get_user_id,
    session_set_user,
    update_password,
    user_meta_path,
    user_upload_dir,
    verify_reset_token,
)
from functions.llm_recommendations import explain_matching_quality, get_career_recommendations
from functions.parse_jobs import search_jobs
from functions.parse_resume import ResumeParser, build_resume_text
from functions.trends import compute_salary_trends, compute_skill_trends
from main import analyze_job_title, semantic_recommendation


# =============================================================================
# App setup & configuration
# =============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-in-production")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=86400)

templates = Jinja2Templates(directory="templates")
parser    = ResumeParser(static_folder="static")

BASE_UPLOAD_DIR = "static/uploads"
USERS_PATH      = "data/users.json"

os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Loaded once at startup — read-only throughout the process lifetime.
with open("data/jobs3.json") as f:
    jobs: list[dict] = json.load(f)


# =============================================================================
# Job-description validator
# Lightweight regex heuristic — no LLM call.
# Mirrors the client-side check so validation is enforced on both sides.
# =============================================================================

_JD_SIGNAL_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(
        r"\b(responsibilities|qualifications|requirements|what you(?:\'ll| will) do"
        r"|about the role|about the job|job description|job summary|position summary"
        r"|key duties|duties and responsibilities)\b", re.I), 3),
    (re.compile(
        r"\b(salary|compensation|pay range|benefits|health insurance|401k|pto"
        r"|paid time off|remote|hybrid|on.?site|full.?time|part.?time"
        r"|contract|permanent)\b", re.I), 2),
    (re.compile(
        r"\b(\d+\+?\s*years?(?: of)? experience|bachelor|master|phd|degree in"
        r"|bsc|msc|mba|equivalent experience)\b", re.I), 3),
    (re.compile(
        r"\b(we are (?:looking for|hiring|seeking)|join our team|apply now"
        r"|submit your (?:resume|cv|application)|equal opportunity employer|eoe"
        r"|candidates will|you will be responsible)\b", re.I), 3),
    (re.compile(
        r"\b(proficiency in|experience with|knowledge of|familiarity with"
        r"|strong understanding of|preferred skills|nice to have|must have"
        r"|required skills)\b", re.I), 2),
    (re.compile(
        r"\b(work with (?:a |the )?(?:team|cross.functional)|collaborate"
        r"|stakeholders|fast.?paced|start.?up|fortune 500|series [a-d]"
        r"|growth stage|mission.?driven)\b", re.I), 1),
]

_SPAM_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(buy now|click here|limited offer|discount|promo code|subscribe"
        r"|unsubscribe|dear (?:sir|madam)|lottery|winner"
        r"|congratulations you(?:\'ve| have) won|nigerian"
        r"|bitcoin|crypto investment)\b", re.I),
    re.compile(r"<script|<iframe|javascript:|on(?:click|load|error)\s*=", re.I),
    re.compile(r"(.)\1{10,}"),  # excessive repetition
]

_JD_SCORE_THRESHOLD = 4


def _is_valid_job_description(text: str) -> tuple[bool, str]:
    """
    Returns (True, '') when text looks like a genuine job description.
    Returns (False, reason) otherwise — called before any LLM is invoked.
    """
    if len(text) < 100:
        return False, "Job description is too short. Please paste the full posting."

    for pattern in _SPAM_PATTERNS:
        if pattern.search(text):
            return False, (
                "This content appears suspicious or irrelevant. "
                "Please paste a genuine job description."
            )

    score = sum(w for p, w in _JD_SIGNAL_PATTERNS if p.search(text))
    if score < _JD_SCORE_THRESHOLD:
        return False, (
            "This doesn't look like a job description. "
            "Please paste a real job posting that includes responsibilities, "
            "required skills, or qualifications."
        )

    return True, ""


# =============================================================================
# Auth helpers
# =============================================================================

def get_current_user(request: Request) -> Optional[dict]:
    """FastAPI dependency — returns the logged-in user dict or None."""
    user_id = session_get_user_id(request.session)
    if not user_id:
        return None
    return get_user_by_id(user_id, USERS_PATH)


def _redirect_if_unauthenticated(user: Optional[dict]) -> Optional[RedirectResponse]:
    """Returns a redirect to /login when user is None, otherwise None."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return None


# =============================================================================
# Resume helpers  (per-user, storage-isolated)
# =============================================================================

def load_meta(user_id: str) -> dict:
    """Load resume metadata for a specific user."""
    path = user_meta_path(BASE_UPLOAD_DIR, user_id)
    if not os.path.exists(path):
        return {"resumes": [], "active": None}
    with open(path) as f:
        return json.load(f)


def save_meta(meta: dict, user_id: str) -> None:
    """Persist resume metadata for a specific user."""
    path = user_meta_path(BASE_UPLOAD_DIR, user_id)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def get_resume_text(user_id: str) -> Optional[str]:
    """Return the parsed semantic text of the user's active resume, or None."""
    meta   = load_meta(user_id)
    active = meta.get("active")
    if not active:
        return None

    full_path = os.path.join(user_upload_dir(BASE_UPLOAD_DIR, user_id), active)
    if not os.path.exists(full_path):
        return None

    try:
        parsed = parser.parse_resume(os.path.join("uploads", user_id, active))
        return build_resume_text(parsed) if parsed else None
    except Exception as e:
        print(f"[get_resume_text] user={user_id} error={e}")
        return None


# =============================================================================
# Bookmark helpers
# =============================================================================

def load_bookmarks(user_id: str) -> list:
    """Return bookmark list for user_id. Each entry: {job_id, saved_at}."""
    path = os.path.join(BASE_UPLOAD_DIR, user_id, "bookmarks.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_bookmarks(user_id: str, bookmarks: list) -> None:
    """Persist the bookmark list for user_id."""
    dir_path = os.path.join(BASE_UPLOAD_DIR, user_id)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "bookmarks.json"), "w") as f:
        json.dump(bookmarks, f, indent=2)


def get_bookmarked_ids(user_id: str) -> set:
    """Return a set of bookmarked job_id strings for fast membership tests."""
    return {str(b["job_id"]) for b in load_bookmarks(user_id)}


# =============================================================================
# Template helpers
# =============================================================================

def _render(template: str, request: Request, **ctx) -> HTMLResponse:
    """Thin wrapper so route handlers stay concise."""
    return templates.TemplateResponse(template, {"request": request, **ctx})


def _no_resume_error(request: Request, user: dict, template: str) -> HTMLResponse:
    """Shared 'no active resume' error response for protected routes."""
    return _render(
        template, request,
        user=user,
        jobs=[], query="", recommended=True,
        error="Please upload and select a resume first.",
        bookmarked_ids=set(),
    )


# =============================================================================
# Routes — Public
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: Optional[dict] = Depends(get_current_user)):
    return _render("index_new.html", request, user=user)


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    query:   str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    results        = search_jobs(query, jobs)
    bookmarked_ids = get_bookmarked_ids(user["user_id"]) if user else set()
    return _render(
        "results.html", request,
        user=user, jobs=results, query=query,
        recommended=False, bookmarked_ids=bookmarked_ids,
    )


@app.post("/title_summary", response_class=HTMLResponse)
def title_summary(
    request: Request,
    query:   str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    matched_jobs, top_skills = analyze_job_title(query, jobs)

    # Build word-cloud image
    wc         = WordCloud(width=800, height=400, background_color="white")
    wc         = wc.generate_from_frequencies({skill: count for skill, count in top_skills})
    buf        = io.BytesIO()
    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    years, trend_skills, trend_data              = compute_skill_trends(jobs, query, top_n=6)
    salary_years, salary_by_year, current_salary = compute_salary_trends(matched_jobs)

    return _render(
        "title_summary.html", request,
        user=user, query=query,
        matched_jobs=matched_jobs, top_skills=top_skills,
        wordcloud=img_base64,
        trend_years=years, trend_skills=trend_skills, trend_data=trend_data,
        salary_years=salary_years, salary_by_year=salary_by_year,
        current_salary=current_salary,
    )


# =============================================================================
# Routes — Auth
# =============================================================================

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return _render("register.html", request, error=None)


@app.post("/register", response_class=HTMLResponse)
def register(
    request:          Request,
    name:             str = Form(...),
    email:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
):
    if password != confirm_password:
        return _render("register.html", request, error="Passwords do not match.")

    try:
        user = create_user(name, email, password, USERS_PATH)
    except ValueError as e:
        return _render("register.html", request, error=str(e))

    session_set_user(request.session, user["user_id"])
    return RedirectResponse(url="/profile", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if session_get_user_id(request.session):
        return RedirectResponse(url="/", status_code=303)
    return _render("login.html", request, error=None)


@app.post("/login", response_class=HTMLResponse)
def login(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
):
    user = authenticate_user(email, password, USERS_PATH)
    if user is None:
        return _render("login.html", request, error="Incorrect email or password.")
    session_set_user(request.session, user["user_id"])
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    session_clear(request.session)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return _render("forgot_password.html", request, message=None, reset_url=None, error=None)


@app.post("/forgot-password", response_class=HTMLResponse)
def forgot_password(request: Request, email: str = Form(...)):
    """
    Generate a signed reset token and display the link on-screen.
    In production, send this via email instead.
    Same response message whether or not the account exists (prevents enumeration).
    """
    reset_url = None
    user      = get_user_by_email(email, USERS_PATH)
    if user:
        token     = make_reset_token(email, SESSION_SECRET)
        reset_url = f"/reset-password?token={token}"

    return _render(
        "forgot_password.html", request,
        message="If an account exists for that email a reset link has been generated below.",
        reset_url=reset_url, error=None,
    )


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(request: Request, token: str):
    email = verify_reset_token(token, SESSION_SECRET)
    if email is None:
        return _render(
            "reset_password.html", request,
            token=token, success=False,
            error="This reset link has expired or is invalid. Please request a new one.",
        )
    return _render("reset_password.html", request, token=token, error=None, success=False)


@app.post("/reset-password", response_class=HTMLResponse)
def reset_password(
    request:          Request,
    token:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
):
    email = verify_reset_token(token, SESSION_SECRET)
    if email is None:
        return _render(
            "reset_password.html", request,
            token=token, success=False,
            error="This reset link has expired or is invalid.",
        )

    if password != confirm_password:
        return _render(
            "reset_password.html", request,
            token=token, success=False, error="Passwords do not match.",
        )

    try:
        update_password(email, password, USERS_PATH)
    except ValueError as e:
        return _render(
            "reset_password.html", request,
            token=token, success=False, error=str(e),
        )

    return _render("reset_password.html", request, token="", error=None, success=True)


# =============================================================================
# Routes — Profile / Resume management  (protected)
# =============================================================================

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
            print(f"[profile_page] parse error: {e}")

    return _render(
        "profile.html", request,
        user=user, resumes=meta["resumes"],
        active=meta["active"], parsed=parsed, error=None,
    )


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
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name   = file.filename.replace(" ", "_")
        unique_name = f"{timestamp}_{safe_name}"
        full_path   = os.path.join(upload_dir, unique_name)

        content = await file.read()
        with open(full_path, "wb") as f:
            f.write(content)

        parsed_data = parser.parse_resume(os.path.join("uploads", user_id, unique_name))
        if not parsed_data:
            raise ValueError("Parsing failed — check file format.")

        meta["resumes"].append({
            "filename":     unique_name,
            "display_name": file.filename,
            "uploaded_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        if meta["active"] is None:
            meta["active"] = unique_name
        save_meta(meta, user_id)

    except Exception as e:
        return _render(
            "profile.html", request,
            user=user, resumes=meta["resumes"],
            active=meta["active"], parsed=None, error=str(e),
        )

    meta = load_meta(user_id)
    return _render(
        "profile.html", request,
        user=user, resumes=meta["resumes"],
        active=meta["active"], parsed=parsed_data, error=None,
    )


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
    if filename in [r["filename"] for r in meta["resumes"]]:
        meta["active"] = filename
        save_meta(meta, user_id)

    return RedirectResponse(url="/profile", status_code=303)


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


# =============================================================================
# Routes — AI features  (protected)
# =============================================================================

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
        return _render(
            "results.html", request,
            user=user, jobs=[], query=query, recommended=True,
            error="Please upload and select a resume first.",
            bookmarked_ids=get_bookmarked_ids(user["user_id"]),
        )

    recommended_jobs = semantic_recommendation(resume_text, search_jobs(query, jobs))
    return _render(
        "results.html", request,
        user=user, jobs=recommended_jobs, query=query,
        recommended=True, bookmarked_ids=get_bookmarked_ids(user["user_id"]),
    )


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
        return _no_resume_error(request, user, "results.html")

    selected_job = next((j for j in jobs if str(j.get("job_id")) == str(job_id)), None)
    if not selected_job:
        return _render(
            "results.html", request,
            user=user, jobs=[], query="", recommended=True, error="Job not found.",
            bookmarked_ids=set(),
        )

    explanation = explain_matching_quality(resume_text, [selected_job], 1)
    explanation = markdown.markdown(explanation, extensions=["tables"])
    return _render("fit.html", request, user=user, job=selected_job, explanation=explanation)


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
        return _render(
            "career.html", request,
            user=user, error="Please upload and select a resume on your Profile page first.",
            candidate=None, recommendations=[], narrative=None,
        )

    try:
        relative_path = os.path.join("uploads", user["user_id"], meta["active"])
        parsed_resume = parser.parse_resume(relative_path)
        resume_text   = build_resume_text(parsed_resume)
    except Exception as e:
        return _render(
            "career.html", request,
            user=user, error=f"Could not read active resume: {e}",
            candidate=None, recommendations=[], narrative=None,
        )

    result = get_career_recommendations(
        resume_text=resume_text,
        parsed_resume=parsed_resume,
        jobs=jobs,
        top_n=4,
    )
    return _render(
        "career.html", request,
        user=user, error=None,
        candidate=result["candidate"],
        recommendations=result["recommendations"],
        narrative=result["narrative"],
    )


# =============================================================================
# Routes — Paste-JD fit  (protected)
# =============================================================================

@app.get("/fit/paste", response_class=HTMLResponse)
def paste_fit_page(
    request: Request,
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    meta = load_meta(user["user_id"])
    if not meta.get("active"):
        return _render(
            "paste_fit.html", request, user=user, no_resume=True,
            error="Please upload and select a resume on your Profile page before checking a job fit.",
        )

    return _render("paste_fit.html", request, user=user, no_resume=False, error=None)


@app.post("/fit/paste", response_class=HTMLResponse)
def paste_fit(
    request:         Request,
    job_description: str = Form(...),
    job_title:       str = Form(""),
    company_name:    str = Form(""),
    user:            Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    resume_text = get_resume_text(user["user_id"])
    if not resume_text:
        return _render("paste_fit.html", request, user=user, no_resume=True,
                       error="Please upload and select a resume first.")

    job_description = job_description.strip()
    job_title       = job_title.strip()       or "Pasted Job"
    company_name    = company_name.strip()    or "Unknown Company"

    if len(job_description) < 50:
        return _render("paste_fit.html", request, user=user, no_resume=False,
                       error="Job description is too short. Please paste the full description.")

    is_valid, rejection_reason = _is_valid_job_description(job_description)
    if not is_valid:
        return _render("paste_fit.html", request, user=user, no_resume=False,
                       error=rejection_reason)

    # Build a minimal job dict matching the shape expected by explain_matching_quality.
    pasted_job = {
        "job_id":                   "pasted",
        "title":                    job_title,
        "company":                  company_name,
        "job_description":          job_description,
        "skills_required":          [],
        "responsibilities":         [],
        "preferred_qualifications": [],
        "salary_range":             "Not specified",
        "education_required":       "Not specified",
        "location":                 "Not specified",
        "employment_type":          "Not specified",
        "experience_required":      "Not specified",
    }

    try:
        explanation = explain_matching_quality(resume_text, [pasted_job], 1)
        explanation = markdown.markdown(explanation, extensions=["tables"])
    except Exception as e:
        return _render("paste_fit.html", request, user=user, no_resume=False,
                       error=f"Could not generate fit summary: {e}")

    return _render("fit.html", request, user=user, job=pasted_job, explanation=explanation)


# =============================================================================
# Routes — Bookmarks  (protected)
# =============================================================================

@app.get("/bookmarks", response_class=HTMLResponse)
def view_bookmarks(
    request: Request,
    user:    Optional[dict] = Depends(get_current_user),
):
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    job_index     = {str(j["job_id"]): j for j in jobs}
    raw_bookmarks = load_bookmarks(user["user_id"])
    bookmarks     = [
        {"job": job_index[str(b["job_id"])], "saved_at": b["saved_at"]}
        for b in raw_bookmarks
        if str(b["job_id"]) in job_index
    ]
    return _render("bookmarks.html", request, user=user, bookmarks=bookmarks)


@app.post("/bookmarks/add", response_class=HTMLResponse)
def add_bookmark(
    request: Request,
    job_id:  str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    """Idempotent — bookmarking an already-saved job is a no-op."""
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    bookmarks = load_bookmarks(user["user_id"])
    if str(job_id) not in {str(b["job_id"]) for b in bookmarks}:
        bookmarks.append({
            "job_id":   str(job_id),
            "saved_at": datetime.now().strftime("%d %b %Y"),
        })
        save_bookmarks(user["user_id"], bookmarks)

    return RedirectResponse(url="/bookmarks", status_code=303)


@app.post("/bookmarks/remove", response_class=HTMLResponse)
def remove_bookmark(
    request: Request,
    job_id:  str = Form(...),
    user:    Optional[dict] = Depends(get_current_user),
):
    """Idempotent — removing a non-existent bookmark is a no-op."""
    redirect = _redirect_if_unauthenticated(user)
    if redirect:
        return redirect

    bookmarks = [b for b in load_bookmarks(user["user_id"]) if str(b["job_id"]) != str(job_id)]
    save_bookmarks(user["user_id"], bookmarks)
    return RedirectResponse(url="/bookmarks", status_code=303)