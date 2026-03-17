from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import json
import pdfplumber
import io

from functions.parse_resume import ResumeParser, build_resume_text
from functions.model import search_jobs   # your embedding search


app = FastAPI()

templates = Jinja2Templates(directory="templates")

parser = ResumeParser(static_folder="static")

UPLOAD_DIR = "static/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load jobs
with open("data/jobs2.json") as f:
    jobs = json.load(f)

# -------------------------------
# 1️⃣ HOME PAGE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


# # -------------------------------
# # 3️⃣ RECOMMENDED (LLM)
# # -------------------------------
# @app.post("/recommend", response_class=HTMLResponse)
# def recommend(request: Request, query: str = Form(...)):

#     # Safety check
#     if not query:
#         return templates.TemplateResponse("results.html", {
#             "request": request,
#             "jobs": [],
#             "query": "",
#             "recommended": True,
#             "error": "Empty query provided"
#         })

#     try:
#         # Step 1: Semantic search
#         results = search_jobs(query, jobs)

#         # Step 2: LLM re-ranking
#         recommended_jobs = llm_recommend(results)

#     except Exception as e:
#         print("LLM ERROR:", e)

#         # fallback → show semantic results
#         recommended_jobs = results

#         return templates.TemplateResponse("results.html", {
#             "request": request,
#             "jobs": recommended_jobs,
#             "query": query,
#             "recommended": False,
#             "error": "LLM recommendation failed. Showing semantic results."
#         })

#     return templates.TemplateResponse("results.html", {
#         "request": request,
#         "jobs": recommended_jobs,
#         "query": query,
#         "recommended": True
#     })



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


@app.post("/profile", response_class=HTMLResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)):

    # --- Validation ---
    if not file:
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "error": "No file uploaded",
            "parsed": None
        })

    try:
        content = await file.read()
        
        if file.filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                resume_text = "\n".join([p for p in pages if p])

        # # Convert bytes → text
        # resume_text = content.decode("utf-8", errors="ignore")

        # ✅ USE YOUR EXISTING PARSER
        parsed_data = parser.parse_resume(resume_text)

        if not parsed_data:
            raise ValueError("Parsing failed")

    except Exception as e:
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "error": f"Error processing file: {str(e)}",
            "parsed": None
        })

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "parsed": parsed_data
    })











# # # # -------------------------------
# # # # 4️⃣ PROFILE (UPLOAD RESUME)
# # # # -------------------------------
# # # @app.get("/profile", response_class=HTMLResponse)
# # # def profile_page(request: Request):
# # #     return templates.TemplateResponse("profile.html", {
# # #         "request": request,
# # #         "parsed": None
# # #     })


# # @app.post("/profile/upload", response_class=HTMLResponse)
# # def upload_resume(request: Request, file: UploadFile = File(...)):

# #     if not file.filename.endswith(".pdf"):
# #         return templates.TemplateResponse("profile.html", {
# #             "request": request,
# #             "error": "Only PDF files allowed"
# #         })

# #     file_path = os.path.join(UPLOAD_DIR, file.filename)

# #     with open(file_path, "wb") as buffer:
# #         shutil.copyfileobj(file.file, buffer)

# #     try:
# #         parsed = parse_resume(file_path)
# #     except Exception as e:
# #         return templates.TemplateResponse("profile.html", {
# #             "request": request,
# #             "error": "Failed to parse resume"
# #         })

# #     return templates.TemplateResponse("profile.html", {
# #         "request": request,
# #         "parsed": parsed
# #     })

# @app.post("/profile", response_class=HTMLResponse)
# async def upload_resume(request: Request, file: UploadFile = File(...)):

#     if not file:
#         return templates.TemplateResponse("profile.html", {
#             "request": request,
#             "error": "No file uploaded",
#             "parsed": None
#         })

#     try:
#         content = await file.read()

#         # ✅ Handle PDF properly
#         if file.filename.endswith(".pdf"):
#             with pdfplumber.open(io.BytesIO(content)) as pdf:
#                 pages = [page.extract_text() for page in pdf.pages]
#                 resume_text = "\n".join([p for p in pages if p])

#         else:
#             # fallback for txt files
#             resume_text = content.decode("utf-8", errors="ignore")

#         # Parse resume
#         parsed_data = parser.parse_resume(resume_text)

#         if not parsed_data:
#             raise ValueError("Parsing failed")

#     except Exception as e:
#         return templates.TemplateResponse("profile.html", {
#             "request": request,
#             "error": f"Error processing file: {str(e)}",
#             "parsed": None
#         })

#     return templates.TemplateResponse("profile.html", {
#         "request": request,
#         "parsed": parsed_data
#     })