import os
from pathlib import Path
from urllib.parse import urlencode

import requests

try:
    import streamlit as st
except Exception:
    st = None

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"

_LOCAL_SECRETS = None


def _load_local_secrets() -> dict:
    global _LOCAL_SECRETS
    if _LOCAL_SECRETS is not None:
        return _LOCAL_SECRETS

    if tomllib is None:
        _LOCAL_SECRETS = {}
        return _LOCAL_SECRETS

    candidates = [
        Path(__file__).resolve().parent / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                _LOCAL_SECRETS = tomllib.loads(candidate.read_text())
                return _LOCAL_SECRETS
            except Exception:
                continue

    _LOCAL_SECRETS = {}
    return _LOCAL_SECRETS


def _looks_placeholder(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return True

    placeholders = {
        "your_client_id",
        "your_client_secret",
        "your-app-name",
        "real_client_id",
        "real_client_secret",
        "https://your-app-name.streamlit.app",
    }

    return (
        normalized in placeholders
        or normalized.startswith("your_")
        or normalized.startswith("real_")
    )


def _get_secret(key: str, default: str = "") -> str:
    value = os.getenv(key, "")
    if value and not _looks_placeholder(value):
        return value

    if st is not None:
        try:
            secret_value = st.secrets.get(key, default)
            if secret_value and not _looks_placeholder(secret_value):
                return secret_value
        except Exception:
            pass

    local_secrets = _load_local_secrets()
    local_value = local_secrets.get(key, default)
    if local_value and not _looks_placeholder(local_value):
        return local_value

    return default


STRAVA_CLIENT_ID = _get_secret("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = _get_secret("STRAVA_CLIENT_SECRET")
STRAVA_REDIRECT_URI = _get_secret("STRAVA_REDIRECT_URI")


def _configured() -> bool:
    return all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REDIRECT_URI])


def get_auth_url() -> str | None:
    if not _configured():
        return None

    params = {
        "client_id": STRAVA_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": STRAVA_REDIRECT_URI,
        "approval_prompt": "force",
        "scope": "read,read_all,profile:read_all,activity:read_all",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    if not _configured():
        return {"error": "Strava credentials are not configured."}

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"Token exchange failed: {exc}"}


def refresh_access_token(refresh_token: str) -> dict:
    if not _configured():
        return {"error": "Strava credentials are not configured."}

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=20,
        )
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"Token refresh failed: {exc}"}


def get_activities(access_token: str, per_page: int = 10) -> list:
    try:
        response = requests.get(
            f"{API_BASE}/athlete/activities",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"per_page": per_page},
            timeout=20,
        )
        if response.status_code != 200:
            return []
        return response.json()
    except requests.RequestException:
        return []


def get_activity_streams(activity_id: int, access_token: str) -> dict:
    try:
        response = requests.get(
            f"{API_BASE}/activities/{activity_id}/streams",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "keys": "time,heartrate,cadence,altitude",
                "key_by_type": "true",
            },
            timeout=20,
        )
        if response.status_code != 200:
            return {}
        return response.json()
    except requests.RequestException:
        return {}
