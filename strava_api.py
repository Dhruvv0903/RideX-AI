import requests
import webbrowser
import time

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REDIRECT_URI = "http://localhost:8501"

def get_auth_url():
    return f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&approval_prompt=auto&scope=activity:read_all"

def exchange_code_for_token(code):
    url = "https://www.strava.com/oauth/token"

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    }

    response = requests.post(url, data=data)
    return response.json()


def get_activities(access_token):
    url = "https://www.strava.com/api/v3/athlete/activities"

    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(url, headers=headers)
    return response.json()
