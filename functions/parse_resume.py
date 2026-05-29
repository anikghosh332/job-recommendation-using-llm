import os
import re
from dataclasses import dataclass
from typing import Optional
import pdfplumber
from docx import Document


# ─────────────────────────────────────────────────────────────
# Section result wrapper — carries parsed data + a confidence flag
# so the UI can decide whether to show a manual-edit prompt.
# ─────────────────────────────────────────────────────────────

@dataclass
class SectionResult:
    data: object          # list or dict — the actual parsed content
    parsed: bool = True   # False → auto-parse produced nothing; prompt manual entry


class ResumeParser:
    def __init__(self, static_folder="static"):
        self.static_folder = static_folder

    # ─────────────────────────────────────────────────────────
    # Public Method
    # ─────────────────────────────────────────────────────────

    def parse_resume(self, filename: str) -> dict:
        """
        Main entry point.  Returns a structured dictionary where every
        top-level value is a SectionResult so callers can tell whether
        parsing succeeded or whether manual entry is needed.

        Keys guaranteed to exist (even if empty / parsed=False):
            profile_header  → SectionResult(data={"summary": str})
            education       → SectionResult(data=[...])
            work_experience → SectionResult(data=[...])
            skills          → SectionResult(data={category: [skill, ...]})
            certifications  → SectionResult(data=[str, ...])
            projects        → SectionResult(data=[...])
        """
        file_path = os.path.join(self.static_folder, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume not found at {file_path}")

        text = self._extract_text(file_path)

        if not text or len(text.strip()) < 100:
            raise ValueError("Resume text extraction failed or document is empty.")

        sections = self._split_sections(text)

        def _sr(data, empty_check=None) -> SectionResult:
            if empty_check is None:
                empty_check = data
            return SectionResult(data=data, parsed=bool(empty_check))

        # Profile
        profile_data = self._parse_profile_header(sections.get("PROFILE", ""), text)
        profile_sr   = _sr(profile_data, profile_data.get("summary"))

        # Education
        edu_data = self._parse_education(sections.get("EDUCATION", ""))
        edu_sr   = _sr(edu_data)

        # Work Experience
        work_text = sections.get("WORK EXPERIENCE", "") or sections.get("EXPERIENCE", "")
        work_data = self._parse_work_experience(work_text)
        work_sr   = _sr(work_data)

        # Skills
        skills_data = self._parse_skills(sections.get("SKILLS", ""))
        skills_sr   = _sr(skills_data)

        # Certifications
        cert_text = (
            sections.get("CERTIFICATIONS", "")
            or sections.get("CERTIFICATION", "")
        )
        cert_data = self._parse_certifications(cert_text, text)
        cert_sr   = _sr(cert_data)

        # Projects
        proj_data = self._parse_projects(sections.get("PROJECTS", ""))
        proj_sr   = _sr(proj_data)

        return {
            "profile_header":  profile_sr,
            "education":       edu_sr,
            "work_experience": work_sr,
            "skills":          skills_sr,
            "certifications":  cert_sr,
            "projects":        proj_sr,
        }

    # ─────────────────────────────────────────────────────────
    # Text Extraction
    # ─────────────────────────────────────────────────────────

    def _extract_text(self, file_path: str) -> str:
        try:
            if file_path.lower().endswith(".pdf"):
                return self._extract_pdf(file_path)
            elif file_path.lower().endswith(".docx"):
                return self._extract_docx(file_path)
            elif file_path.lower().endswith((".txt", ".doc")):
                with open(file_path, "r", errors="ignore") as f:
                    return f.read()
            else:
                raise ValueError("Unsupported file format. Only PDF, DOCX, TXT supported.")
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

    # ─────────────────────────────────────────────────────────
    # Section Splitter
    # ─────────────────────────────────────────────────────────

    _SECTION_ALIASES = {
        "PROFILE":          "PROFILE",
        "EDUCATION":        "EDUCATION",
        "WORK EXPERIENCE":  "WORK EXPERIENCE",
        "EXPERIENCE":       "WORK EXPERIENCE",
        "PROJECTS":         "PROJECTS",
        "SKILLS":           "SKILLS",
        "CERTIFICATIONS":   "CERTIFICATIONS",
        "CERTIFICATION":    "CERTIFICATIONS",
        "ACHIEVEMENTS":     "ACHIEVEMENTS",
        "Profile":          "PROFILE",
        "Education":        "EDUCATION",
        "Work Experience":  "WORK EXPERIENCE",
        "Experience":       "WORK EXPERIENCE",
        "Projects":         "PROJECTS",
        "Skills":           "SKILLS",
        "Certifications":   "CERTIFICATIONS",
        "Certification":    "CERTIFICATIONS",
        "Achievements":     "ACHIEVEMENTS",
        "TECHNICAL SKILLS": "SKILLS",
        "Technical Skills": "SKILLS",
        "KEY SKILLS":       "SKILLS",
        "Key Skills":       "SKILLS",
    }

    def _split_sections(self, text: str) -> dict:
        all_headers = list(self._SECTION_ALIASES.keys())
        all_headers.sort(key=len, reverse=True)
        escaped = [re.escape(h) for h in all_headers]
        pattern = r"(?=\n?(" + "|".join(escaped) + r")\n)"

        splits = re.split(pattern, text)

        sections = {}
        current_key = None

        for chunk in splits:
            chunk_stripped = chunk.strip()
            if chunk_stripped in self._SECTION_ALIASES:
                current_key = self._SECTION_ALIASES[chunk_stripped]
                if current_key not in sections:
                    sections[current_key] = ""
            elif current_key:
                sections[current_key] += chunk_stripped + "\n"

        return sections

    # ─────────────────────────────────────────────────────────
    # Profile Header
    # ─────────────────────────────────────────────────────────

    def _parse_profile_header(self, profile_text: str, full_text: str = "") -> dict:
        summary = profile_text.strip()
        summary = re.sub(r"^PROFILE\s*\n?", "", summary).strip()
        summary = "\n".join(
            l for l in summary.split("\n")
            if not re.match(r"^[_\-=]{4,}$", l.strip())
        ).strip()

        if not summary and full_text:
            for line in full_text.split("\n"):
                line = line.strip()
                if (
                    len(line) >= 40
                    and not re.match(r"^[\+\d\s\|\@\.]+$", line)
                    and not re.match(r"^[_\-=]{4,}$", line)
                ):
                    summary = line
                    break

        return {"summary": summary}

    # ─────────────────────────────────────────────────────────
    # Education Parser
    # ─────────────────────────────────────────────────────────

    def _parse_education(self, education_text: str) -> list:
        education_entries = []

        if not education_text.strip():
            return education_entries

        education_text = re.sub(r"^(?:EDUCATION|Education)\s*\n?", "", education_text.strip())

        degree_starters = (
            "Bachelor", "Master", "B.Tech", "M.Tech",
            "BSc", "MSc", "PhD", "Doctor", "Diploma",
        )
        joined_lines = []
        for line in education_text.split("\n"):
            stripped = line.strip()
            if (
                joined_lines
                and stripped
                and not stripped.startswith("•")
                and not stripped.startswith(degree_starters)
                and not re.match(r"^[A-Z][A-Za-z\s&,\(\)]+ -", stripped)
                and not re.search(r"\d{4}", joined_lines[-1])
            ):
                joined_lines[-1] += " " + stripped
            else:
                joined_lines.append(stripped)

        rejoined = "\n".join(joined_lines)

        fmt_a_blocks = re.split(r"\n(?=[A-Z][A-Za-z\s&,\(\)]+ - )", rejoined.strip())

        if len(fmt_a_blocks) > 1:
            for block in fmt_a_blocks:
                block = block.strip()
                if not block:
                    continue
                entry = self._parse_single_education_block(block)
                if entry and entry.get("university") and entry["university"] not in ("EDUCATION",):
                    education_entries.append(entry)
            if education_entries:
                return education_entries

        raw_blocks = re.split(r"\n{2,}", rejoined.strip())
        if len(raw_blocks) <= 1:
            raw_blocks = re.split(
                r"\n(?=(?:Bachelor|Master|B\.Tech|M\.Tech|BSc|MSc|PhD|Doctor|Diploma))",
                rejoined.strip()
            )

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            entry = self._parse_single_education_block(block)
            if entry and (entry.get("university") or entry.get("degree")):
                if entry.get("university") not in (None, "Education", "EDUCATION"):
                    education_entries.append(entry)

        return education_entries

    def _parse_single_education_block(self, block: str):
        date_pattern = (
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|"
            r"(?:\d{1,2}/\d{4})"
            r")"
            r"(?:\s*(?:to|-)\s*"
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|(?:\d{1,2}/\d{4})"
            r"|present|Present|current|Current"
            r")"
            r")?"
        )

        date_match = re.search(date_pattern, block)
        start_date = date_match.group(1).strip() if date_match else None
        end_date   = date_match.group(2).strip() if (date_match and date_match.group(2)) else None

        first_line = block.split("\n")[0].strip()
        first_line_clean = first_line
        if date_match:
            first_line_clean = first_line_clean.replace(date_match.group(0), "").strip()

        dash_parts = re.split(r"\s*-\s*", first_line_clean, maxsplit=1)
        if len(dash_parts) == 2:
            university = dash_parts[0].strip()
            degree_raw = dash_parts[1].strip()
            if university and degree_raw:
                degree, major = self._split_degree_major(degree_raw)
                return {"university": university, "degree": degree, "major": major,
                        "start_date": start_date, "end_date": end_date}

        degree_kw_pattern = re.compile(
            r"^(Bachelor of|Master of|B\.Tech|M\.Tech|BSc|MSc|PhD|Doctor of|Diploma)\b"
        )
        kw_match = degree_kw_pattern.match(first_line_clean)
        if kw_match:
            uni_pattern = re.search(
                r"\b(University|Institute|College|School|Vellore|Liverpool|MIT|IIT|IIM)\b",
                first_line_clean
            )
            if uni_pattern:
                degree_raw = first_line_clean[:uni_pattern.start()].strip()
                university = first_line_clean[uni_pattern.start():].strip()
                degree, major = self._split_degree_major(degree_raw)
                return {"university": university, "degree": degree, "major": major,
                        "start_date": start_date, "end_date": end_date}

        lines = [l.strip() for l in block.split("\n") if l.strip() and not l.strip().startswith("•")]
        if len(lines) >= 2:
            degree_raw = lines[0]
            if date_match:
                degree_raw = degree_raw.replace(date_match.group(0), "").strip()
            university = lines[1]
            if date_match:
                university = university.replace(date_match.group(0), "").strip()
            degree, major = self._split_degree_major(degree_raw)
            return {"university": university, "degree": degree, "major": major,
                    "start_date": start_date, "end_date": end_date}

        if lines:
            text = lines[0]
            if date_match:
                text = text.replace(date_match.group(0), "").strip()
            degree, major = self._split_degree_major(text)
            return {"university": None, "degree": degree, "major": major,
                    "start_date": start_date, "end_date": end_date}

        return None

    def _split_degree_major(self, text: str):
        degree_keywords = [
            "Bachelor of", "Master of", "B.Tech", "M.Tech",
            "BSc", "MSc", "PhD", "Doctor of",
        ]
        for kw in degree_keywords:
            if text.startswith(kw):
                words = text.split()
                split_at = 3 if " of " in kw or kw in ("Bachelor of", "Master of", "Doctor of") else 1
                degree = " ".join(words[:split_at])
                major  = " ".join(words[split_at:]) if len(words) > split_at else None
                return degree, major
        words = text.split()
        if len(words) > 3:
            return " ".join(words[:3]), " ".join(words[3:])
        return text, None

    # ─────────────────────────────────────────────────────────
    # Work Experience Parser
    # ─────────────────────────────────────────────────────────

    def _parse_work_experience(self, work_text: str) -> list:
        experiences = []

        if not work_text.strip():
            return experiences

        work_text = re.sub(
            r"^(?:WORK EXPERIENCE|EXPERIENCE|Work Experience|Experience)\s*\n?",
            "", work_text.strip()
        )

        date_pattern = (
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|(?:\d{1,2}/\d{4})"
            r")"
            r"(?:\s*(?:to|-)\s*"
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|(?:\d{1,2}/\d{4})"
            r"|present|Present|current|Current"
            r")"
            r")?"
        )

        fmt_a_entries = re.split(r"\n(?=[A-Za-z][\w\s,&/]+\s\|\s)", work_text)

        parsed_any_fmt_a = False
        for entry in fmt_a_entries:
            entry = entry.strip()
            if not entry:
                continue
            header = re.match(
                r"^(.*?)\s*\|\s*(.*?)(?=\s+(?:\d{1,2}/\d{4}|[A-Za-z]{2,9}[\s\.]*\d{4}))",
                entry
            )
            if not header:
                continue

            position   = header.group(1).strip()
            company    = header.group(2).strip()
            date_match = re.search(date_pattern, entry)
            start_date = date_match.group(1).strip() if date_match else None
            end_date   = date_match.group(2).strip() if (date_match and date_match.group(2)) else "present"
            description = entry[header.end():].strip()
            if date_match:
                description = description.replace(date_match.group(0), "").strip()

            experiences.append({
                "company": company, "position": position,
                "start_date": start_date, "end_date": end_date, "description": description,
            })
            parsed_any_fmt_a = True

        if parsed_any_fmt_a:
            return experiences

        blocks = self._split_fmt_b_entries(work_text)

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) < 2:
                continue

            position   = lines[0].lstrip("•").strip()
            company    = None
            start_date = None
            end_date   = "present"
            desc_lines = []

            for i, line in enumerate(lines[1:], 1):
                date_match = re.search(date_pattern, line)
                if date_match and company is None:
                    company    = line[:date_match.start()].strip().rstrip(",").strip()
                    start_date = date_match.group(1).strip()
                    end_date   = date_match.group(2).strip() if date_match.group(2) else "present"
                else:
                    desc_lines.append(line.lstrip("•").strip())

            if not company:
                continue

            experiences.append({
                "company": company, "position": position,
                "start_date": start_date, "end_date": end_date,
                "description": "\n".join(desc_lines),
            })

        return experiences

    def _split_fmt_b_entries(self, text: str) -> list:
        lines  = text.split("\n")
        blocks = []
        current = []

        date_hint = re.compile(r"(\d{1,2}/\d{4}|[A-Za-z]{2,9}[\s\.]*\d{4})")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            is_title = (
                stripped
                and not stripped.startswith("•")
                and len(stripped) < 60
                and any(
                    date_hint.search(lines[j].strip())
                    for j in range(i + 1, min(i + 3, len(lines)))
                )
            )

            if is_title and current:
                blocks.append("\n".join(current))
                current = []

            current.append(stripped)

        if current:
            blocks.append("\n".join(current))

        return blocks

    # ─────────────────────────────────────────────────────────
    # Skills Parser  (NEW)
    # ─────────────────────────────────────────────────────────

    _SKILL_CATEGORY_PATTERN = re.compile(
        r"^("
        r"ML/AI|MLOps|Cloud|Data Engineering|Data Visualis?ation|Web/API"
        r"|Languages|Gen AI|Generative AI|Soft Skills|Technical Skills"
        r"|Programming|Frameworks?|Databases?|Tools?|Platforms?"
        r"|Front.?End|Back.?End|DevOps|Infrastructure|Analytics"
        r")\s*[:\-]?\s*",
        re.IGNORECASE
    )

    def _parse_skills(self, skills_text: str) -> dict:
        """
        Returns {category: [skill, ...]} — category may be 'General' when no
        explicit label is found.  Handles both inline and bulleted formats.
        """
        result: dict = {}

        if not skills_text.strip():
            return result

        skills_text = re.sub(
            r"^(?:SKILLS|Skills|TECHNICAL SKILLS|Technical Skills)\s*\n?",
            "", skills_text.strip()
        )

        lines = [l.strip() for l in skills_text.split("\n") if l.strip()]
        current_category = "General"

        for line in lines:
            if re.match(r"^Page\s+\d+$", line, re.I):
                continue

            cat_match = self._SKILL_CATEGORY_PATTERN.match(line)
            if cat_match:
                current_category = cat_match.group(1).strip()
                remainder = line[cat_match.end():].strip().rstrip(":")
                if remainder:
                    result.setdefault(current_category, []).extend(
                        self._split_skills_line(remainder)
                    )
            elif line.startswith("•"):
                content = line.lstrip("•").strip()
                inner_cat = self._SKILL_CATEGORY_PATTERN.match(content)
                if inner_cat:
                    cat = inner_cat.group(1).strip()
                    rem = content[inner_cat.end():].strip().rstrip(":")
                    if rem:
                        result.setdefault(cat, []).extend(self._split_skills_line(rem))
                    else:
                        current_category = cat
                else:
                    result.setdefault(current_category, []).extend(
                        self._split_skills_line(content)
                    )
            else:
                if "," in line or len(line.split()) > 4:
                    result.setdefault(current_category, []).extend(
                        self._split_skills_line(line)
                    )
                elif re.match(r"^[A-Z][a-zA-Z /]+$", line) and len(line) < 30:
                    current_category = line

        return {cat: list(dict.fromkeys(s for s in skills if s))
                for cat, skills in result.items()}

    def _split_skills_line(self, text: str) -> list:
        raw = re.split(r"[,;]", text)
        return [s.strip().strip("•").strip() for s in raw if s.strip()]

    # ─────────────────────────────────────────────────────────
    # Certifications Parser  (NEW)
    # ─────────────────────────────────────────────────────────

    _INLINE_CERT_PATTERN = re.compile(r"CERTIFICATION\s*[:\-]\s*(.+)", re.IGNORECASE)

    def _parse_certifications(self, cert_text: str, full_text: str = "") -> list:
        """
        Returns a list of certification strings.
        Looks in the dedicated section AND inline 'CERTIFICATION : ...' lines.
        """
        certs: list = []

        if cert_text.strip():
            cert_text = re.sub(
                r"^(?:CERTIFICATIONS?|Certifications?)\s*\n?", "", cert_text.strip()
            )
            for line in cert_text.split("\n"):
                line = line.strip().lstrip("•").strip()
                if line and not re.match(r"^Page\s+\d+$", line, re.I):
                    certs.append(line)

        if full_text:
            for match in self._INLINE_CERT_PATTERN.finditer(full_text):
                cert = match.group(1).strip()
                if cert and cert not in certs:
                    certs.append(cert)

        return [c for c in certs if c]

    # ─────────────────────────────────────────────────────────
    # Projects Parser
    # ─────────────────────────────────────────────────────────

    def _parse_projects(self, projects_text: str) -> list:
        projects = []

        if not projects_text.strip():
            return projects

        entries = re.split(r"\n(?=[A-Z][A-Za-z0-9 &\-\(\)]+(?:\n|$))", projects_text)

        _junk = re.compile(
            r"^(Page\s+\d+|ACHIEVEMENTS|SKILLS|CERTIFICATION|PROFILE|EDUCATION|WORK EXPERIENCE)$",
            re.I
        )

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue
            title = lines[0].strip()
            if _junk.match(title):
                continue
            description = "\n".join(lines[1:]).strip()
            projects.append({
                "project_title":       title,
                "project_description": description,
            })

        return projects


# ─────────────────────────────────────────────────────────────
# build_resume_text — updated for SectionResult + new sections
# ─────────────────────────────────────────────────────────────

def _unwrap(section_or_raw):
    """Accept either a SectionResult or a raw value (backwards compat)."""
    if isinstance(section_or_raw, SectionResult):
        return section_or_raw.data
    return section_or_raw


def build_resume_text(resume: dict) -> str:
    profile_text = _unwrap(resume.get("profile_header", {}))
    if isinstance(profile_text, dict):
        profile_text = profile_text.get("summary", "")

    education_text = ""
    for edu in _unwrap(resume.get("education", [])):
        degree     = edu.get("degree", "") or ""
        major      = edu.get("major", "") or ""
        university = edu.get("university", "") or ""
        education_text += f"{degree} in {major} from {university}. "

    experience_text = ""
    for exp in _unwrap(resume.get("work_experience", [])):
        position    = exp.get("position", "")
        company     = exp.get("company", "")
        description = exp.get("description", "")
        experience_text += f"{position} at {company}. {description} "

    skills_data  = _unwrap(resume.get("skills", {}))
    skills_parts = []
    if isinstance(skills_data, dict):
        for cat, items in skills_data.items():
            skills_parts.append(f"{cat}: {', '.join(items)}")
    skills_text = ". ".join(skills_parts)

    cert_list = _unwrap(resume.get("certifications", []))
    cert_text = ". ".join(cert_list) if cert_list else ""

    projects_text = ""
    for proj in _unwrap(resume.get("projects", [])):
        title       = proj.get("project_title", "")
        description = proj.get("project_description", "")
        projects_text += f"{title}: {description} "

    resume_text = f"""
    Candidate Profile:
    {profile_text}

    Education:
    {education_text}

    Work Experience:
    {experience_text}

    Skills:
    {skills_text}

    Certifications:
    {cert_text}

    Projects:
    {projects_text}
    """

    return resume_text.strip()