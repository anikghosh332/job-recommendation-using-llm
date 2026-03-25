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
            "profile_header": self._parse_profile_header(sections.get("PROFILE", ""), text),
            "education":       self._parse_education(sections.get("EDUCATION", "")),
            "work_experience": self._parse_work_experience(sections.get("WORK EXPERIENCE", "")),
            "projects":        self._parse_projects(sections.get("PROJECTS", "")),
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
    # FIX 1: Accept mixed-case section headers (e.g. "Work Experience", "Education")
    # and also map them to canonical keys.
    _SECTION_ALIASES = {
        "PROFILE":         "PROFILE",
        "EDUCATION":       "EDUCATION",
        "WORK EXPERIENCE": "WORK EXPERIENCE",
        "PROJECTS":        "PROJECTS",
        # mixed-case variants found in the hospitality resume
        "Work Experience": "WORK EXPERIENCE",
        "Education":       "EDUCATION",
        "Skills":          "SKILLS",
        "Achievements":    "ACHIEVEMENTS",
        "Certification":   "CERTIFICATION",
    }

    def _split_sections(self, text: str) -> dict:
        # Build a pattern that matches any known header (case-sensitive, exact line)
        all_headers = list(self._SECTION_ALIASES.keys())
        # Sort longest first so multi-word headers match before single words
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

    # -------------------------
    # Profile Header
    # -------------------------
    # FIX 2: If no explicit PROFILE section was found, fall back to grabbing
    # the first substantial paragraph from the raw text (the hospitality resume
    # has the summary as a loose paragraph before any header).
    def _parse_profile_header(self, profile_text: str, full_text: str = "") -> dict:
        summary = profile_text.strip()

        # Strip a leading "PROFILE" label if the section splitter left it in
        summary = re.sub(r"^PROFILE\s*\n?", "", summary).strip()

        # Skip decorative separator lines (e.g. "______...")
        summary = "\n".join(
            l for l in summary.split("\n")
            if not re.match(r"^[_\-=]{4,}$", l.strip())
        ).strip()

        if not summary and full_text:
            # Find first paragraph of 40+ chars that isn't a name/contact/separator line
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

    # -------------------------
    # Education Parser
    # -------------------------
    def _parse_education(self, education_text: str) -> list:
        education_entries = []

        if not education_text.strip():
            return education_entries

        # Strip any leading section label the splitter left in (all-caps or title-case)
        education_text = re.sub(r"^(?:EDUCATION|Education)\s*\n?", "", education_text.strip())

        # Join lines that are clearly continuations (no bullet, no date, short orphan line)
        # e.g. "Bachelor of Technology Electronics and Communications\nEngineering July 2017"
        # → "Bachelor of Technology Electronics and Communications Engineering July 2017"
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
                and not stripped.startswith(degree_starters)               # not a new degree line
                and not re.match(r"^[A-Z][A-Za-z\s&,\(\)]+ -", stripped)  # not a new entry
                and not re.search(r"\d{4}", joined_lines[-1])              # prev line has no year yet
            ):
                joined_lines[-1] += " " + stripped
            else:
                joined_lines.append(stripped)

        rejoined = "\n".join(joined_lines)

        # Format A: split on lines starting with "University - "
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

        # Format B: split on blank lines or degree-keyword lines
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

    def _parse_single_education_block(self, block: str) -> dict | None:
        """Parse one education block regardless of format."""

        # ── Shared date patterns ──────────────────────────────────────────────
        date_pattern = (
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|"
            r"(?:\d{1,2}/\d{4})"
            r")"
            r"(?:\s*(?:to|-)\s*"
            r"("
            r"(?:[A-Za-z]{2,9}[\s\.]*\d{4})"
            r"|"
            r"(?:\d{1,2}/\d{4})"
            r"|present|Present|current|Current"
            r")"
            r")?"
        )

        date_match = re.search(date_pattern, block)
        start_date = date_match.group(1).strip() if date_match else None
        end_date   = date_match.group(2).strip() if (date_match and date_match.group(2)) else "present"

        # Work on the first line only (before bullets)
        first_line = block.split("\n")[0].strip()

        # Strip date from first line for cleaner parsing
        first_line_clean = first_line
        if date_match:
            first_line_clean = first_line_clean.replace(date_match.group(0), "").strip()

        # ── Format A: "University - Degree [Major]"  ─────────────────────────
        # Split on the LAST " - " that separates institution from degree
        # (after date removal, only one " - " should remain)
        dash_parts = re.split(r"\s*-\s*", first_line_clean, maxsplit=1)
        if len(dash_parts) == 2:
            university = dash_parts[0].strip()
            degree_raw = dash_parts[1].strip()
            if university and degree_raw:
                degree, major = self._split_degree_major(degree_raw)
                return {
                    "university": university,
                    "degree":     degree,
                    "major":      major,
                    "start_date": start_date,
                    "end_date":   end_date,
                }

        # ── Format C: "Degree [Major] University"  on one line ───────────────
        # e.g. "Master of Science University of Liverpool"
        # Strategy: detect a known degree keyword, then split on the university
        # name which follows immediately after the degree+major words.
        degree_kw_pattern = re.compile(
            r"^(Bachelor of|Master of|B\.Tech|M\.Tech|BSc|MSc|PhD|Doctor of|Diploma)\b"
        )
        kw_match = degree_kw_pattern.match(first_line_clean)
        if kw_match:
            # Find where the university name starts — look for a capitalised word
            # that follows the degree portion and is NOT a common major word.
            # Heuristic: university names usually contain "University", "Institute",
            # "College", "School", or "of" after a proper noun.
            uni_pattern = re.search(
                r"\b(University|Institute|College|School|Vellore|Liverpool|MIT|IIT|IIM)\b",
                first_line_clean
            )
            if uni_pattern:
                degree_raw  = first_line_clean[:uni_pattern.start()].strip()
                university  = first_line_clean[uni_pattern.start():].strip()
                degree, major = self._split_degree_major(degree_raw)
                return {
                    "university": university,
                    "degree":     degree,
                    "major":      major,
                    "start_date": start_date,
                    "end_date":   end_date,
                }

        # ── Format B: degree on first line, university on second line ─────────
        lines = [l.strip() for l in block.split("\n") if l.strip() and not l.strip().startswith("•")]
        if len(lines) >= 2:
            degree_raw = lines[0]
            # Remove date from degree line
            if date_match:
                degree_raw = degree_raw.replace(date_match.group(0), "").strip()
            university = lines[1]
            # Remove date from university line too
            if date_match:
                university = university.replace(date_match.group(0), "").strip()
            degree, major = self._split_degree_major(degree_raw)
            return {
                "university": university,
                "degree":     degree,
                "major":      major,
                "start_date": start_date,
                "end_date":   end_date,
            }

        # ── Fallback: single line ─────────────────────────────────────────────
        if lines:
            text = lines[0]
            if date_match:
                text = text.replace(date_match.group(0), "").strip()
            degree, major = self._split_degree_major(text)
            return {
                "university": None,
                "degree":     degree,
                "major":      major,
                "start_date": start_date,
                "end_date":   end_date,
            }

        return None

    def _split_degree_major(self, text: str):
        """Split a degree string into (degree, major)."""
        degree_keywords = [
            "Bachelor of", "Master of", "B.Tech", "M.Tech",
            "BSc", "MSc", "PhD", "Doctor of",
        ]
        for kw in degree_keywords:
            if text.startswith(kw):
                words = text.split()
                # "Bachelor of Science Computer Science" → degree="Bachelor of Science", major="Computer Science"
                # Use 3 words for "X of Y" patterns, 1-2 words for abbreviations
                split_at = 3 if " of " in kw or kw in ("Bachelor of", "Master of", "Doctor of") else 1
                degree = " ".join(words[:split_at])
                major  = " ".join(words[split_at:]) if len(words) > split_at else None
                return degree, major

        # fallback
        words = text.split()
        if len(words) > 3:
            return " ".join(words[:3]), " ".join(words[3:])
        return text, None

    # -------------------------
    # Work Experience Parser
    # -------------------------
    # FIX 4: Support two layouts:
    #   Format A (ML resume):          "Position | Company  MonthYear to MonthYear"
    #   Format B (hospitality resume): "Position Title\nCompany  MM/YYYY to present"
    def _parse_work_experience(self, work_text: str) -> list:
        experiences = []

        if not work_text.strip():
            return experiences

        # ── Shared date pattern (word month OR MM/YYYY, with optional end) ────
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

        # ── Try Format A split first ──────────────────────────────────────────
        fmt_a_entries = re.split(r"\n(?=[A-Za-z][\w\s,&/]+\s\|\s)", work_text)

        parsed_any_fmt_a = False
        for entry in fmt_a_entries:
            entry = entry.strip()
            if not entry:
                continue
            # Match "Position | Company  date_or_end_of_line"
            # Use a greedy company match that stops at the date pattern
            header = re.match(
                r"^(.*?)\s*\|\s*(.*?)(?=\s+(?:\d{1,2}/\d{4}|[A-Za-z]{2,9}[\s\.]*\d{4}))",
                entry
            )
            if not header:
                continue

            position = header.group(1).strip()
            company  = header.group(2).strip()

            date_match  = re.search(date_pattern, entry)
            start_date  = date_match.group(1).strip() if date_match else None
            end_date    = date_match.group(2).strip() if (date_match and date_match.group(2)) else "present"
            description = entry[header.end():].strip()
            # strip the date line from description
            if date_match:
                description = description.replace(date_match.group(0), "").strip()

            experiences.append({
                "company":    company,
                "position":   position,
                "start_date": start_date,
                "end_date":   end_date,
                "description": description,
            })
            parsed_any_fmt_a = True

        if parsed_any_fmt_a:
            return experiences

        # ── Format B: "Position\nCompany  date\n• bullets..." ─────────────────
        # Split on a line that contains ONLY a date range (the company+date line
        # is on its own line in this format).  We split at lines where the NEXT
        # line carries a date, i.e. each new job starts with a bold/title line.
        #
        # Heuristic: split whenever we see a line that does NOT start with •
        # and is followed (within 2 lines) by a date string.
        blocks = self._split_fmt_b_entries(work_text)

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) < 2:
                continue

            # First non-bullet line = position title
            position = lines[0].lstrip("•").strip()

            # Find the line that has the date → that line also has the company
            company    = None
            start_date = None
            end_date   = "present"
            desc_lines = []

            for i, line in enumerate(lines[1:], 1):
                date_match = re.search(date_pattern, line)
                if date_match and company is None:
                    # This line = "Company  date"
                    company    = line[:date_match.start()].strip().rstrip(",").strip()
                    start_date = date_match.group(1).strip()
                    end_date   = date_match.group(2).strip() if date_match.group(2) else "present"
                else:
                    desc_lines.append(line.lstrip("•").strip())

            if not company:
                continue

            experiences.append({
                "company":     company,
                "position":    position,
                "start_date":  start_date,
                "end_date":    end_date,
                "description": "\n".join(desc_lines),
            })

        return experiences

    def _split_fmt_b_entries(self, text: str) -> list:
        """
        Split work text into per-job blocks for Format B resumes.
        A new block starts when a line is short (< 60 chars), not a bullet,
        and is followed within 2 lines by a date.
        """
        lines  = text.split("\n")
        blocks = []
        current = []

        date_hint = re.compile(
            r"(\d{1,2}/\d{4}|[A-Za-z]{2,9}[\s\.]*\d{4})"
        )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Detect start of new entry: short non-bullet line where a nearby
            # following line contains a date
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

    # -------------------------
    # Projects Parser
    # -------------------------
    def _parse_projects(self, projects_text: str) -> list:
        projects = []

        if not projects_text.strip():
            return projects

        entries = re.split(r"\n(?=[A-Z][A-Za-z0-9 &]+(?:\n|$))", projects_text)

        _junk = re.compile(r"^(Page\s+\d+|ACHIEVEMENTS|SKILLS|CERTIFICATION|PROFILE|EDUCATION)$", re.I)

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue
            title       = lines[0].strip()
            if _junk.match(title):
                continue
            description = "\n".join(lines[1:]).strip()
            projects.append({
                "project_title":       title,
                "project_description": description,
            })

        return projects


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def build_resume_text(resume):
    """
    Convert parsed resume dictionary into semantic-rich text for embedding.
    """
    profile_text = resume.get("profile_header", {}).get("summary", "")

    education_text = ""
    for edu in resume.get("education", []):
        degree     = edu.get("degree", "") or ""
        major      = edu.get("major", "") or ""
        university = edu.get("university", "") or ""
        education_text += f"{degree} in {major} from {university}. "

    experience_text = ""
    for exp in resume.get("work_experience", []):
        position    = exp.get("position", "")
        company     = exp.get("company", "")
        description = exp.get("description", "")
        experience_text += f"{position} at {company}. {description} "

    projects_text = ""
    for proj in resume.get("projects", []):
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

    Projects:
    {projects_text}
    """

    return resume_text.strip()
