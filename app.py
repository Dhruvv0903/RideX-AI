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
sample_csv = """heart_rate,cadence,slope,speed
80,70,2,15
120,85,5,22
140,78,8,20
160,65,10,18
155,60,11,16
"""

st.download_button(
    label="📄 Download Sample CSV",
    data=sample_csv,
    file_name="sample_ride.csv"
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload ride file", type=["csv", "tcx"])


# ---------------- TCX PARSER ----------------
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    data = []

    for trackpoint in root.findall('.//ns:Trackpoint', ns):
        hr = trackpoint.find('.//ns:HeartRateBpm/ns:Value', ns)
        cadence = trackpoint.find('.//ns:Cadence', ns)
        altitude = trackpoint.find('.//ns:AltitudeMeters', ns)

        data.append({
            'heart_rate': int(hr.text) if hr is not None else 0,
            'cadence': int(cadence.text) if cadence is not None else 0,
            'elevation_m': float(altitude.text) if altitude is not None else 0
        })

    return pd.DataFrame(data)


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

    low_cadence_pct = (df['cadence'] < 70).sum() / len(df) * 100

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


# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    try:
        file_name = uploaded_file.name.lower()

        # Detect file type
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".tcx"):
            df = parse_tcx(uploaded_file)

        else:
            st.error("Unsupported file format. Upload CSV or TCX.")
            st.stop()

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        # Normalize
        df = normalize_data(df)

        # Validate heart rate
        if df['heart_rate'].sum() == 0:
            st.error("No valid heart rate data found.")
            st.stop()

        # Duration approximation
        df['duration_min'] = df.index / 60

        # Calculate fatigue
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

        # ---------------- OUTPUT ----------------
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

        col1, col2, col3 = st.columns(3)
        col1.metric("Max Fatigue", df['fatigue_score'].max())
        col2.metric("Average Fatigue", round(df['fatigue_score'].mean(), 1))
        col3.metric("Avg Speed", round(df['speed'].mean(), 1))

        st.subheader("⏱ Time in Zones")
        total = len(df)
        low = (df['fatigue_score'] < 40).sum()
        moderate = ((df['fatigue_score'] >= 40) & (df['fatigue_score'] < 65)).sum()
        high = (df['fatigue_score'] >= 65).sum()

        st.write(f"🟢 Low: {round(low/total*100,1)}%")
        st.write(f"🟡 Moderate: {round(moderate/total*100,1)}%")
        st.write(f"🔴 High: {round(high/total*100,1)}%")

        st.subheader("📊 Ride Summary")
        st.write(f"Peak Fatigue: {df['fatigue_score'].max()}")
        st.write(f"Time in High Fatigue: {high} data points")

        st.subheader("🏁 Final Verdict")
        if latest < 40:
            st.success("Endurance ride — you can train again tomorrow.")
        elif latest < 65:
            st.warning("Moderate strain — consider a light recovery ride next.")
        else:
            st.error("High fatigue — recovery strongly recommended.")

        st.subheader("🧠 Insights")
        for insight in generate_insights(df):
            st.write("- " + insight)

        st.subheader("💡 Key Takeaway")
        st.write("Your fatigue was primarily influenced by intensity, cadence, and sustained effort.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
