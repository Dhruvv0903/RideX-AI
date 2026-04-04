import pandas as pd
import numpy as np
from strava_api import get_activities

# ==============================
# STRAVA LOADER
# ==============================
def load_from_strava(access_token):
    activities = get_activities(access_token)

    if not activities or isinstance(activities, dict):
        return None

    data = []

    for act in activities[:10]:  # last 10 activities
        if act.get("type") != "Ride":
            continue

        hr = act.get("average_heartrate", None)

        # skip if no HR (important)
        if hr is None:
            continue

        data.append({
            "time": pd.to_datetime(act["start_date"]),
            "hr": float(hr)
        })

    if len(data) == 0:
        return None

    df = pd.DataFrame(data).sort_values("time")
    df["delta"] = 1

    return df


# ==============================
# MOCK CONNECTED DEVICE
# ==============================
def load_from_device():
    base = pd.Timestamp.now()

    times = pd.date_range(end=base, periods=300, freq="s")

    hr = 115 + 30 * np.sin(np.linspace(0, 8, 300)) + np.random.normal(0, 3, 300)

    df = pd.DataFrame({
        "time": times,
        "hr": hr.clip(100, 180)
    })

    df["delta"] = 1

    return df
