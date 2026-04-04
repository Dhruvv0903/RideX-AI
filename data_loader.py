import pandas as pd
import numpy as np

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
