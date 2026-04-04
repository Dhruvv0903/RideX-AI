import pandas as pd
import numpy as np
from strava_api import get_activities
import pandas as pd
import numpy as np

def load_from_strava(access_token):
    activities = get_activities(access_token)

    if not activities:
        return None

    data = []

    for act in activities[:5]:  # last 5 rides
        if act["type"] != "Ride":
            continue

        hr = act.get("average_heartrate", 120)

        data.append({
            "time": pd.to_datetime(act["start_date"]),
            "hr": hr
        })

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.sort_values("time")
        df["delta"] = 1

    return df
# ==============================
# MOCK CONNECTED DEVICE (PHASE 1)
# ==============================
def load_from_device():
    """
    Simulates pulling data from Fitbit / Apple / Google.
    Replace later with real API calls.
    """

    base = pd.Timestamp.now()

    times = pd.date_range(end=base, periods=300, freq="s")

    hr = 115 + 30 * np.sin(np.linspace(0, 8, 300)) + np.random.normal(0, 3, 300)

    df = pd.DataFrame({
        "time": times,
        "hr": hr.clip(100, 180)
    })

    df["delta"] = 1

    return df
