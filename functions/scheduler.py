# =============================================================================
# functions/scheduler.py
#
# Background scheduler that refreshes the live jobs cache every 6 hours.
# Uses APScheduler — install with:  pip install apscheduler
#
# Called once at app startup in app.py:
#   from functions.scheduler import start_scheduler
#   start_scheduler()
# =============================================================================

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from functions.live_jobs import refresh_cache

logger = logging.getLogger(__name__)

# Seed queries used to pre-populate the live jobs cache.
# Add or change these to match your target audience.
SEED_QUERIES = [
    "software engineer",
    "machine learning engineer",
    "data scientist",
    "backend developer",
    "frontend developer",
    "data engineer",
    "DevOps engineer",
    "product manager",
]

# Locations to fetch jobs for. Each query is fetched for every location,
# so 8 queries × 3 locations × 5 pages = 120 API calls per refresh.
# Adjust to stay within your API tier limits.
SEED_LOCATIONS = [
    "London, UK",
    "Remote",
    "United States",
]

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> None:
    """
    Start the background scheduler.
    - Runs an immediate refresh on startup so the cache is never empty.
    - Then repeats every 6 hours.
    """
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.warning("[scheduler] already running — skipping start.")
        return

    _scheduler = BackgroundScheduler(daemon=True)

    # Immediate first run
    _scheduler.add_job(
        _refresh,
        trigger="date",   # run once, right now
        id="live_jobs_startup",
    )

    # Recurring every 6 hours
    _scheduler.add_job(
        _refresh,
        trigger="interval",
        hours=6,
        id="live_jobs_refresh",
    )

    _scheduler.start()
    logger.info("[scheduler] started — live jobs cache will refresh every 6 hours.")


def stop_scheduler() -> None:
    """Gracefully stop the scheduler (called on app shutdown if needed)."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] stopped.")


def _refresh() -> None:
    """Wrapper so exceptions in the job don't crash the scheduler thread."""
    try:
        refresh_cache(SEED_QUERIES, num_pages=5, locations=SEED_LOCATIONS)
    except Exception as e:
        logger.error(f"[scheduler] refresh_cache failed: {e}")