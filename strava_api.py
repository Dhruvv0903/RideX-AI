import os
from typing import Any, Dict, List, Optional

import requests


STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI", "")

AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"


def _configured() -> bool:
    return all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REDIRECT_URI])


def get_auth_url() -> Optional[str]:
    if not _configured():
        return None
    return (
        f"{AUTH_URL}?client_id={STRAVA_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={STRAVA_REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope=read,activity:read_all"
    )


def exchange_code_for_token(code: str) -> Dict[str, Any]:
    if not _configured():
        return {"error": "Strava environment variables are not configured."}

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


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
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


def get_activities(access_token: str, per_page: int = 10) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{API_BASE}/athlete/activities",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"per_page": per_page},
        timeout=20,
    )
    if response.status_code != 200:
        return []
    return response.json()


def get_activity_streams(activity_id: int, access_token: str) -> Dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/activities/{activity_id}/streams",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": "time,heartrate,cadence,altitude", "key_by_type": "true"},
        timeout=20,
    )
    if response.status_code != 200:
        return {}
    return response.json()
