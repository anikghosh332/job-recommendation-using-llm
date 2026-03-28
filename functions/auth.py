"""
functions/auth.py

Authentication helpers — password hashing, user CRUD, session helpers,
and password-reset token management.

Design principles:
  - No FastAPI imports — pure Python so every function is unit-testable.
  - No global state — all functions accept the data file path explicitly.
  - Users are stored in data/users.json (one JSON object keyed by email).
  - Passwords are hashed with bcrypt via passlib.
  - Reset tokens are signed, time-limited HMAC tokens (itsdangerous).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed."""
    return _pwd_context.verify(plain, hashed)


# ── Validation ────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Returns (True, "") if password is acceptable,
    or (False, reason) if not.
    Rules: min 8 chars, at least one digit, at least one letter.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter."
    return True, ""


# ── User store ────────────────────────────────────────────────────────────────
# Schema: { email: { "user_id": str, "name": str, "email": str,
#                    "password_hash": str, "created_at": str } }

def _load_users(users_path: str) -> dict:
    if not os.path.exists(users_path):
        return {}
    with open(users_path, "r") as f:
        return json.load(f)


def _save_users(users: dict, users_path: str) -> None:
    os.makedirs(os.path.dirname(users_path), exist_ok=True)
    with open(users_path, "w") as f:
        json.dump(users, f, indent=2)


def get_user_by_email(email: str, users_path: str) -> Optional[dict]:
    """Return user dict or None."""
    users = _load_users(users_path)
    return users.get(email.lower().strip())


def get_user_by_id(user_id: str, users_path: str) -> Optional[dict]:
    """Return user dict or None. O(n) — fine for small user sets."""
    users = _load_users(users_path)
    for user in users.values():
        if user["user_id"] == user_id:
            return user
    return None


def create_user(name: str, email: str, password: str, users_path: str) -> dict:
    """
    Create and persist a new user.
    Raises ValueError if email already exists or inputs are invalid.
    Returns the new user dict (without password_hash).
    """
    email = email.lower().strip()
    name  = name.strip()

    if not name:
        raise ValueError("Name cannot be empty.")
    if not is_valid_email(email):
        raise ValueError("Invalid email address.")
    ok, reason = is_strong_password(password)
    if not ok:
        raise ValueError(reason)

    users = _load_users(users_path)
    if email in users:
        raise ValueError("An account with this email already exists.")

    import uuid
    user_id = str(uuid.uuid4())
    user = {
        "user_id":       user_id,
        "name":          name,
        "email":         email,
        "password_hash": hash_password(password),
        "created_at":    datetime.utcnow().isoformat(),
    }
    users[email] = user
    _save_users(users, users_path)

    # Return public-safe dict (no hash)
    return {k: v for k, v in user.items() if k != "password_hash"}


def update_password(email: str, new_password: str, users_path: str) -> None:
    """
    Replace a user's password hash.
    Raises ValueError if user not found or password too weak.
    """
    email = email.lower().strip()
    ok, reason = is_strong_password(new_password)
    if not ok:
        raise ValueError(reason)

    users = _load_users(users_path)
    if email not in users:
        raise ValueError("User not found.")

    users[email]["password_hash"] = hash_password(new_password)
    _save_users(users, users_path)


def authenticate_user(email: str, password: str, users_path: str) -> Optional[dict]:
    """
    Verify email + password.
    Returns public user dict on success, None on failure.
    """
    user = get_user_by_email(email, users_path)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {k: v for k, v in user.items() if k != "password_hash"}


# ── Reset tokens ──────────────────────────────────────────────────────────────

def make_reset_token(email: str, secret_key: str, max_age: int = 3600) -> str:
    """
    Generate a signed, time-limited reset token embedding the email.
    max_age is in seconds (default 1 hour).
    """
    s = URLSafeTimedSerializer(secret_key, salt="password-reset")
    return s.dumps(email.lower().strip())


def verify_reset_token(token: str, secret_key: str, max_age: int = 3600) -> Optional[str]:
    """
    Verify and decode a reset token.
    Returns the email on success, None if expired or invalid.
    """
    s = URLSafeTimedSerializer(secret_key, salt="password-reset")
    try:
        email = s.loads(token, max_age=max_age)
        return email
    except (SignatureExpired, BadSignature):
        return None


# ── Session helpers ───────────────────────────────────────────────────────────
# The session is a plain dict stored in a signed cookie via SessionMiddleware.
# These helpers keep the key names consistent everywhere.

SESSION_KEY = "user_id"


def session_set_user(session: dict, user_id: str) -> None:
    session[SESSION_KEY] = user_id


def session_get_user_id(session: dict) -> Optional[str]:
    return session.get(SESSION_KEY)


def session_clear(session: dict) -> None:
    session.clear()


# ── User upload directory ─────────────────────────────────────────────────────

def user_upload_dir(base_upload_dir: str, user_id: str) -> str:
    """
    Return and create the per-user upload directory.
    e.g. static/uploads/<user_id>/
    """
    path = os.path.join(base_upload_dir, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def user_meta_path(base_upload_dir: str, user_id: str) -> str:
    """Return path to the per-user resumes_meta.json."""
    return os.path.join(user_upload_dir(base_upload_dir, user_id), "resumes_meta.json")