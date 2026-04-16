from collections import defaultdict, Counter
from datetime import datetime
from functions.parse_jobs import parse_salary


def compute_skill_trends(jobs, query, top_n=5):
    year_skill_map = defaultdict(list)

    for job in jobs:
        if job.get("title", "").lower() == query.lower():
            date_str = job.get("posting_date")
            if not date_str:
                continue

            try:
                year = datetime.strptime(date_str, "%Y-%m-%d").year
            except:
                continue

            year_skill_map[year].extend(job.get("skills_required", []))

    # Count skills
    yearly_counts = {}
    global_counter = Counter()

    for year, skills in year_skill_map.items():
        counter = Counter(skills)
        yearly_counts[year] = counter
        global_counter.update(counter)

    # Top N skills overall
    top_skills = [s for s, _ in global_counter.most_common(top_n)]

    # Build final structure
    years_sorted = sorted(yearly_counts.keys())
    trend_data = {}

    for year in years_sorted:
        total = sum(yearly_counts[year].values()) or 1
        trend_data[year] = {
            skill: round((yearly_counts[year].get(skill, 0) / total) * 100, 2)
            for skill in top_skills
        }

    return years_sorted, top_skills, trend_data  





def compute_salary_trends(matched_jobs) -> tuple:
    """
    Given already-matched jobs (from analyze_job_title / find_similar_jobs),
    compute average salary midpoint per year.

    Returns:
        salary_years   - sorted list of years (int)
        salary_by_year - dict { year: rounded avg midpoint }
        current_salary - avg salary for the most recent year, or None
    """
    year_salaries = defaultdict(list)

    for job in matched_jobs:
        date_str = job.get("posting_date")
        if not date_str:
            continue
        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
        except Exception:
            continue

        midpoint = parse_salary(job.get("salary_range"))
        if midpoint is None:
            continue

        year_salaries[year].append(midpoint)

    if not year_salaries:
        return [], {}, None

    salary_years = sorted(year_salaries.keys())
    salary_by_year = {
        year: int(round(sum(vals) / len(vals)))
        for year, vals in year_salaries.items()
    }
    current_salary = salary_by_year[salary_years[-1]]

    return salary_years, salary_by_year, current_salary 