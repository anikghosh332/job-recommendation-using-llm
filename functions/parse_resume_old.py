import os
import re
import pdfplumber
from docx import Document




class ResumeParser:
    def __init__(self, static_folder="static"):
        self.static_folder = static_folder

    # -------------------------
    # Public Method
    # -------------------------
    def parse_resume(self, filename: str) -> dict:
        """
        Main function to parse resume from /static directory.
        Returns structured dictionary.
        """

        file_path = os.path.join(self.static_folder, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at {file_path}")

        text = self._extract_text(file_path)

        if not text or len(text.strip()) < 100:
            raise ValueError("Resume text extraction failed or document is empty.")

        sections = self._split_sections(text)

        return {
            "profile_header": self._parse_profile_header(sections.get("PROFILE", "")),
            "education": self._parse_education(sections.get("EDUCATION", "")),
            "work_experience": self._parse_work_experience(sections.get("WORK EXPERIENCE", "")),
            "projects": self._parse_projects(sections.get("PROJECTS", "")),
        }

    # -------------------------
    # Text Extraction
    # -------------------------
    def _extract_text(self, file_path: str) -> str:
        try:
            if file_path.lower().endswith(".pdf"):
                return self._extract_pdf(file_path)
            elif file_path.lower().endswith(".docx"):
                return self._extract_docx(file_path)
            else:
                raise ValueError("Unsupported file format. Only PDF and DOCX allowed.")
        except Exception as e:
            raise RuntimeError(f"Error extracting text: {str(e)}")

    def _extract_pdf(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _extract_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    # -------------------------
    # Section Splitter
    # -------------------------
    def _split_sections(self, text: str) -> dict:
        section_titles = [
            "PROFILE",
            "EDUCATION",
            "WORK EXPERIENCE",
            "PROJECTS",
        ]

        pattern = r"(?=\n?(PROFILE|EDUCATION|WORK EXPERIENCE|PROJECTS)\n)"
        splits = re.split(pattern, text)

        sections = {}
        current_title = None

        for chunk in splits:
            chunk = chunk.strip()
            if chunk in section_titles:
                current_title = chunk
                sections[current_title] = ""
            elif current_title:
                sections[current_title] += chunk.strip() + "\n"

        return sections

    # -------------------------
    # Profile Header
    # -------------------------
    def _parse_profile_header(self, profile_text: str) -> dict:
        return {
            "summary": profile_text.strip()
        }

    # -------------------------
    # Education Parser
    # -------------------------
    
    def _parse_education(self, education_text: str) -> list:
        education_entries = []

        if not education_text.strip():
            return education_entries

        entries = re.split(r"\n(?=[A-Z][A-Za-z\s&]+ - )", education_text.strip())

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue


            # UNIVERSITY NAME

            university_match = re.match(r"^(.*?) -", entry)
            if not university_match:
                continue

            university = university_match.group(1).strip()

 
            # DATE 
         
            date_pattern = r"([A-Za-z]{3,9}\s+\d{4})(?:\s*(?:-|to)\s*([A-Za-z]{3,9}\s+\d{4}|Present|present))?"
            date_match = re.search(date_pattern, entry)

            start_date = None
            end_date = "present"

            if date_match:
                start_date = date_match.group(1).strip()
                if date_match.group(2):
                    end_date = date_match.group(2).strip()

           
            # DEGREE FULL TEXT
            # Remove university and date portion
            degree_text = entry

            # Remove university part
            degree_text = re.sub(rf"^{re.escape(university)}\s*-\s*", "", degree_text)

            # Remove date portion
            if date_match:
                degree_text = degree_text.replace(date_match.group(0), "").strip()

            # Remove bullet points
            degree_text = degree_text.split("•")[0].strip()

           
            degree = None
            major = None

            # Common degree keywords
            degree_keywords = [
                "Bachelor of",
                "Master of",
                "B.Tech",
                "M.Tech",
                "BSc",
                "MSc",
                "PhD",
                "Doctor of"
            ]

            for keyword in degree_keywords:
                if degree_text.startswith(keyword):
                    parts = degree_text.split(" ", 3)

                    # Example:
                    # Bachelor of Technology Electronics and Communications Engineering
                    # We assume first 3 words = degree
                    degree_parts = degree_text.split()
                    if len(degree_parts) >= 3:
                        degree = " ".join(degree_parts[:3])
                        major = " ".join(degree_parts[3:]) if len(degree_parts) > 3 else None
                    else:
                        degree = degree_text
                    break

            # Fallback if pattern not matched
            if not degree:
                words = degree_text.split()
                if len(words) > 3:
                    degree = " ".join(words[:3])
                    major = " ".join(words[3:])
                else:
                    degree = degree_text

            

            education_entries.append({
                "university": university,
                "degree": degree.strip() if degree else None,
                "major": major.strip() if major else None,
                "start_date": start_date,
                "end_date": end_date,
            })

        return education_entries

    # -------------------------
    # Work Experience Parser
    # -------------------------
    def _parse_work_experience(self, work_text: str) -> list:
        experiences = []

        entries = re.split(r"\n(?=[A-Za-z\s]+ \| )", work_text)

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            header_match = re.match(r"(.*?) \| (.*?) ([A-Za-z]+\s\d{4}\s(?:to|-)\s[A-Za-z]+\s\d{4})", entry)

            if header_match:
                position = header_match.group(1).strip()
                company = header_match.group(2).strip()
                
                # dates = header_match.group(3).strip()
                
                # DATE 
         
                date_pattern = r"([A-Za-z]{3,9}\s+\d{4})(?:\s*(?:-|to)\s*([A-Za-z]{3,9}\s+\d{4}|Present|present))?"
                date_match = re.search(date_pattern, entry)

                start_date = None
                end_date = "present"

                if date_match:
                    start_date = date_match.group(1).strip()
                    if date_match.group(2):
                        end_date = date_match.group(2).strip()

                description = entry[header_match.end():].strip()

                experiences.append({
                    "company": company,
                    "position": position,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": description
                })

        return experiences

    # -------------------------
    # Projects Parser
    # -------------------------
    def _parse_projects(self, projects_text: str) -> list:
        projects = []

        entries = re.split(r"\n(?=[A-Z][A-Za-z0-9 &]+(?:\n|$))", projects_text)

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue

            title = lines[0].strip()
            description = "\n".join(lines[1:]).strip()

            projects.append({
                "project_title": title,
                "project_description": description
            })

        return projects
    
    

def build_resume_text(resume):
    """
    Convert parsed resume dictionary (based on our parser structure)
    into semantic-rich text for embedding.
    """

    # Profile Summary
    profile_text = resume.get("profile_header", {}).get("summary", "")

    # Education
    education_text = ""
    for edu in resume.get("education", []):
        degree = edu.get("degree", "")
        major = edu.get("major", "")
        university = edu.get("university", "")
        education_text += f"{degree} in {major} from {university}. "

    # Work Experience
    experience_text = ""
    for exp in resume.get("work_experience", []):
        position = exp.get("position", "")
        company = exp.get("company", "")
        description = exp.get("description", "")
        experience_text += f"{position} at {company}. {description} "

    # Projects
    projects_text = ""
    for proj in resume.get("projects", []):
        title = proj.get("project_title", "")
        description = proj.get("project_description", "")
        projects_text += f"{title}: {description} "

    resume_text = f"""
    Candidate Profile:
    {profile_text}

    Education:
    {education_text}

    Work Experience:
    {experience_text}

    Projects:
    {projects_text}
    """

    return resume_text.strip()    

