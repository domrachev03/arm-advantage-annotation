from __future__ import annotations

import hashlib
import hmac
import re
import time

from fastapi import Cookie, HTTPException, Response

from .config import COOKIE_SECURE, MOUNT_PATH, PASSWORD_SHA256, SESSION_SECRET, SESSION_TTL_S

COOKIE = "arm_annot_session"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,47}$")


def _mac(payload: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def sign(name: str, expires_at: int) -> str:
    payload = f"{name}|{expires_at}"
    return f"{payload}|{_mac(payload)}"


def clean_name(name: str) -> str:
    name = name.strip()
    if not NAME_RE.fullmatch(name):
        raise HTTPException(422, "name must be 1-48 ASCII letters, numbers, spaces, '.', '_' or '-'")
    return name


def password_matches(password: str) -> bool:
    if not PASSWORD_SHA256:
        return password == "arm"
    supplied = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(supplied, PASSWORD_SHA256.lower())


def login(response: Response, name: str) -> int:
    expires_at = int(time.time()) + SESSION_TTL_S
    response.set_cookie(
        COOKIE,
        sign(name, expires_at),
        max_age=SESSION_TTL_S,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path=MOUNT_PATH or "/",
    )
    return expires_at


def logout(response: Response) -> None:
    response.delete_cookie(COOKIE, path=MOUNT_PATH or "/", secure=COOKIE_SECURE, samesite="lax")


def optional_session(
    token: str | None = Cookie(default=None, alias=COOKIE),
) -> str | None:
    if not token or token.count("|") != 2:
        return None
    name, raw_exp, supplied_mac = token.split("|")
    try:
        expires_at = int(raw_exp)
    except ValueError:
        return None
    payload = f"{name}|{expires_at}"
    if expires_at < int(time.time()) or not hmac.compare_digest(_mac(payload), supplied_mac):
        return None
    try:
        return clean_name(name)
    except HTTPException:
        return None


def require_session(name: str | None = None) -> str:
    # FastAPI supplies the dependency result to `name`; calling directly remains useful in tests.
    if not name:
        raise HTTPException(401, "login required")
    return name
