import pandas as pd

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
# 🔥 NEW FUNCTION (REQUIRED BY APP)
# ==============================
def compute_fatigue(df, resting_hr, max_hr):
    fatigue = 0
    out = []

    for _, r in df.iterrows():
        intensity = max(0, (r["hr"] - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * r["delta"] * 0.03
        fatigue -= 0.015 * r["delta"]

        fatigue = max(0, min(100, fatigue))
        out.append(fatigue)

    df = df.copy()
    df["fatigue"] = out
    return df


# ==============================
# EXISTING INSIGHTS (KEEP)
# ==============================
def generate_insights(df):
    insights = []

    max_fatigue = df['fatigue_score'].max()
    avg_fatigue = df['fatigue_score'].mean()

    if max_fatigue > 70:
        insights.append("⚠️ High fatigue reached — consider recovery.")

    if avg_fatigue > 50:
        insights.append("🔥 Overall ride intensity was high.")

    if (df['cadence'] < 70).sum() > len(df)*0.3:
        insights.append("🚴 Low cadence contributed to fatigue.")

    if len(insights) == 0:
        insights.append("✅ Ride intensity was well managed.")

    return insights
