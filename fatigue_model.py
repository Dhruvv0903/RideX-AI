import pandas as pd
import numpy as np

# ==============================
# EXISTING FUNCTION (KEEP)
# ==============================
def calculate_fatigue(hr, cadence, slope, duration_min, elevation_m):
    hr_factor       = min(hr / 195, 1.0)
    slope_factor    = min(max(slope, 0) / 15, 1.0)
    duration_factor = min(duration_min / 120, 1.0)
    cadence_factor  = 0.3 if cadence < 70 else 0.15 if cadence < 80 else 0
    elev_factor     = min(elevation_m / 1000, 1.0)

    fatigue = (
        hr_factor       * 35 +
        slope_factor    * 15 +
        duration_factor * 25 +
        cadence_factor  *  8 +
        elev_factor     * 10
    )

    return round(min(fatigue, 100), 1)


# ==============================
# 🔥 UPGRADED PERFORMANCE MODEL
# ==============================
def compute_fatigue(df, resting_hr, max_hr):

    fatigue = 0
    fatigue_series = []
    zone_series = []

    for _, r in df.iterrows():

        hr = r["hr"]
        delta = r["delta"]

        # ----------------------
        # INTENSITY (normalized)
        # ----------------------
        intensity = max(0, (hr - resting_hr) / (max_hr - resting_hr))
        intensity = min(intensity, 1.2)  # allow slight overshoot

        # ----------------------
        # HR ZONES
        # ----------------------
        if intensity < 0.5:
            zone = "Z1"
            load_factor = 0.5
        elif intensity < 0.7:
            zone = "Z2"
            load_factor = 0.8
        elif intensity < 0.85:
            zone = "Z3"
            load_factor = 1.2
        elif intensity < 1.0:
            zone = "Z4"
            load_factor = 1.6
        else:
            zone = "Z5"
            load_factor = 2.2

        zone_series.append(zone)

        # ----------------------
        # FATIGUE BUILDUP
        # ----------------------
        fatigue += intensity * load_factor * delta * 0.035

        # ----------------------
        # RECOVERY DECAY
        # ----------------------
        fatigue -= 0.02 * delta

        # clamp
        fatigue = max(0, min(100, fatigue))

        fatigue_series.append(fatigue)

    df = df.copy()
    df["fatigue"] = fatigue_series
    df["zone"] = zone_series

    return df


# ==============================
# EXISTING INSIGHTS (KEEP)
# ==============================
def generate_insights(df):
    insights = []

    if "fatigue" in df:
        max_fatigue = df['fatigue'].max()
        avg_fatigue = df['fatigue'].mean()

        if max_fatigue > 80:
            insights.append("⚠️ Peak fatigue very high — risk of burnout.")

        if avg_fatigue > 60:
            insights.append("🔥 Sustained high intensity effort.")

    if "zone" in df:
        z5_time = (df["zone"] == "Z5").sum()
        if z5_time > len(df) * 0.2:
            insights.append("🚨 Too much time in max zone.")

    if len(insights) == 0:
        insights.append("✅ Effort distribution balanced.")

    return insights
