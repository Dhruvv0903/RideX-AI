import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from fatigue_model import calculate_fatigue

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")

st.title("🚴 RideX AI — Fatigue Analyzer")
st.write("Understand how hard your ride actually was — not just distance or speed.")

# ---------------- HOW TO USE ----------------
st.subheader("📥 How to Use")

st.write("""
Upload a ride file from apps like Strava, Garmin, or Fitbit.

Supported formats:
- CSV
- TCX (Garmin / Fitbit)

Required:
- Heart rate data

Optional:
- Cadence, speed, elevation
""")

st.info("💡 RideX automatically adapts to different file formats.")

# ---------------- SAMPLE CSV ----------------
# ---------------- SAMPLE CSV (REALISTIC RIDE) ----------------
import numpy as np

np.random.seed(42)

n = 300  # ~5 min ride at 1Hz (you can scale later)

heart_rate = np.clip(
    90 + np.linspace(0, 60, n) + np.random.normal(0, 5, n),
    80, 190
)

cadence = np.clip(
    75 + np.random.normal(0, 10, n),
    50, 110
)

speed = np.clip(
    20 + np.sin(np.linspace(0, 10, n)) * 5 + np.random.normal(0, 1.5, n),
    5, 40
)

slope = np.clip(
    np.sin(np.linspace(0, 6, n)) * 6 + np.random.normal(0, 1, n),
    -5, 12
)

elevation = np.cumsum(np.maximum(slope, 0)) * 2

sample_df = pd.DataFrame({
    "heart_rate": heart_rate.round(0),
    "cadence": cadence.round(0),
    "speed": speed.round(1),
    "slope": slope.round(2),
    "elevation_m": elevation.round(1)
})

sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="📄 Download Realistic Sample Ride",
    data=sample_csv,
    file_name="ridex_sample_realistic.csv"
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload ride file", type=["csv", "tcx"])

# ---------------- TCX PARSER ----------------
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    data = []

    prev_time = None
    prev_dist = None

    for trackpoint in root.findall('.//ns:Trackpoint', ns):

        time_elem = trackpoint.find('.//ns:Time', ns)
        dist_elem = trackpoint.find('.//ns:DistanceMeters', ns)
        hr = trackpoint.find('.//ns:HeartRateBpm/ns:Value', ns)
        cadence = trackpoint.find('.//ns:Cadence', ns)
        altitude = trackpoint.find('.//ns:AltitudeMeters', ns)

        curr_time = pd.to_datetime(time_elem.text) if time_elem is not None else None
        curr_dist = float(dist_elem.text) if dist_elem is not None else None

        speed = 0

        if prev_time is not None and prev_dist is not None and curr_time is not None and curr_dist is not None:
            time_diff = (curr_time - prev_time).total_seconds()
            dist_diff = curr_dist - prev_dist

            if time_diff > 0 and dist_diff >= 0:
                speed = (dist_diff / time_diff) * 3.6

                # cap unrealistic spikes
                if speed > 80:
                    speed = 0

        data.append({
            'heart_rate': int(hr.text) if hr is not None else 0,
            'cadence': int(cadence.text) if cadence is not None else 0,
            'elevation_m': float(altitude.text) if altitude is not None else 0,
            'speed': speed
        })

        prev_time = curr_time
        prev_dist = curr_dist

    df = pd.DataFrame(data)

    # smooth speed
    if 'speed' in df.columns:
        df['speed'] = df['speed'].rolling(5, min_periods=1).mean()

    return df

# ---------------- NORMALIZATION ----------------
def normalize_data(df):
    df.columns = df.columns.str.lower().str.replace('-', '_')

    column_map = {
        'heart_rate': ['heart_rate', 'heartrate', 'heartratebpm', 'hr'],
        'cadence': ['cadence', 'rpm'],
        'speed': ['speed', 'velocity'],
        'elevation_m': ['elevation', 'altitude', 'altitudemeters'],
        'slope': ['slope', 'gradient']
    }

    for standard_col, variations in column_map.items():
        for col in variations:
            if col in df.columns:
                df[standard_col] = df[col]
                break

    for col in ['heart_rate', 'cadence', 'speed', 'slope', 'elevation_m']:
        if col not in df.columns:
            df[col] = 0

    return df

# ---------------- HELPERS ----------------
def fatigue_zone(score):
    if score < 40:
        return "🟢 Low Fatigue"
    elif score < 65:
        return "🟡 Moderate Fatigue"
    else:
        return "🔴 High Fatigue"

def generate_insights(df):
    insights = []

    max_fatigue = df['fatigue_score'].max()
    avg_fatigue = df['fatigue_score'].mean()

    valid_cadence = df[df['cadence'] > 0]

    if len(valid_cadence) > 0:
        low_cadence_pct = (valid_cadence['cadence'] < 70).sum() / len(valid_cadence) * 100
    else:
        low_cadence_pct = 0

    if max_fatigue > 70:
        insights.append("⚠️ You entered high fatigue levels — recovery ride recommended.")

    if avg_fatigue > 50:
        insights.append("🔥 Sustained effort was high — monitor recovery tomorrow.")

    if low_cadence_pct > 30:
        insights.append(
            f"🚴 You spent {round(low_cadence_pct,1)}% of the ride below optimal cadence (<70). This increases fatigue."
        )

    if max_fatigue < 40:
        insights.append("✅ Ride intensity was controlled — good endurance session.")

    if len(insights) == 0:
        insights.append("👍 Balanced ride — no major fatigue drivers detected.")

    return insights

# ---------------- MAIN ----------------
if uploaded_file:

    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".tcx"):
            df = parse_tcx(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        df = normalize_data(df)

        if df['heart_rate'].sum() == 0:
            st.error("No valid heart rate data found.")
            st.stop()

        # detect if speed existed originally
        original_speed_missing = df['speed'].sum() == 0

        df['duration_min'] = df.index / 60

        df['fatigue_score'] = df.apply(
            lambda row: calculate_fatigue(
                row['heart_rate'],
                row['cadence'],
                row['slope'],
                row['duration_min'],
                row['elevation_m']
            ),
            axis=1
        )

        latest = df['fatigue_score'].iloc[-1]

        st.subheader("📍 Current State")
        st.write(f"Fatigue Level: {fatigue_zone(latest)}")

        st.subheader("📈 Fatigue Curve")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['fatigue_score'], linewidth=2)

        ax.axhline(40, linestyle='--', alpha=0.7)
        ax.axhline(65, linestyle='--', alpha=0.7)

        ax.set_title("Fatigue Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Fatigue Score")

        st.pyplot(fig)

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Max Fatigue", df['fatigue_score'].max())
        col2.metric("Average Fatigue", round(df['fatigue_score'].mean(), 1))
        col3.metric("Avg Speed (computed)", round(df['speed'].mean(), 1))

        # explain speed logic
        st.caption("ℹ️ Speed is derived from distance and time when not directly available.")

        if original_speed_missing:
            st.info("Speed data was not present in the file — calculated using distance/time.")

        # ---------------- ZONES ----------------
        st.subheader("⏱ Time in Zones")

        total = len(df)
        low = (df['fatigue_score'] < 40).sum()
        moderate = ((df['fatigue_score'] >= 40) & (df['fatigue_score'] < 65)).sum()
        high = (df['fatigue_score'] >= 65).sum()

        st.write(f"🟢 Low: {round(low/total*100,1)}%")
        st.write(f"🟡 Moderate: {round(moderate/total*100,1)}%")
        st.write(f"🔴 High: {round(high/total*100,1)}%")

        # ---------------- SUMMARY ----------------
        st.subheader("📊 Ride Summary")
        st.write(f"Peak Fatigue: {df['fatigue_score'].max()}")
        st.write(f"Time in High Fatigue: {high} data points")

        # ---------------- VERDICT ----------------
        st.subheader("🏁 Final Verdict")

        if latest < 40:
            st.success("Endurance ride — you can train again tomorrow.")
        elif latest < 65:
            st.warning("Moderate strain — consider a light recovery ride next.")
        else:
            st.error("High fatigue — recovery strongly recommended.")

        # ---------------- INSIGHTS ----------------
        st.subheader("🧠 Insights")
        for insight in generate_insights(df):
            st.write("- " + insight)

        st.subheader("💡 Key Takeaway")
        st.write("Your fatigue was primarily influenced by intensity, cadence, and sustained effort.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
