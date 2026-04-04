import webbrowser
import time
import requests

CLIENT_ID = "220471"
CLIENT_SECRET = "755c40c43a36c92ce4c207cfcf6f8724bb171003"
REDIRECT_URI = "https://ridex-ai.streamlit.app/"

# ==============================
# AUTH URL
# ==============================
def get_auth_url():
    return (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=auto"
        f"&scope=activity:read_all"
    )

# ==============================
# EXCHANGE CODE → TOKEN
# ==============================
def exchange_code_for_token(code):
    url = "https://www.strava.com/oauth/token"

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    }

    return requests.post(url, data=data).json()

# ==============================
# REFRESH TOKEN
# ==============================
def refresh_access_token(refresh_token):
    url = "https://www.strava.com/oauth/token"

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }

    return requests.post(url, data=data).json()

# ==============================
# GET ACTIVITIES
# ==============================
def get_activities(access_token):
    url = "https://www.strava.com/api/v3/athlete/activities"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return []

    return response.json()
