import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_model import calculate_fatigue

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")

st.title("🚴 RideX AI — Fatigue Analyzer")
st.write("Understand how hard your ride actually was — not just distance or speed.")

# ---------------- HOW TO USE ----------------
st.subheader("📥 How to Use")

st.write("""
Upload a CSV file exported from fitness apps like Strava, Fitbit, Garmin, or Apple Health.

Required:
- heart rate data (any format)

Optional:
- cadence, speed, elevation
""")

st.info("💡 Tip: Different apps export different formats — RideX will automatically adapt your data.")

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
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


# ---------------- NORMALIZATION FUNCTION ----------------
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

    # Fill missing columns
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
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        # Normalize data
        df = normalize_data(df)

        # Validate core data
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

        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("Max Fatigue", df['fatigue_score'].max())
        col2.metric("Average Fatigue", round(df['fatigue_score'].mean(), 1))
        col3.metric("Avg Speed", round(df['speed'].mean(), 1))

        st.subheader("🧠 Insights")
        for insight in generate_insights(df):
            st.write("- " + insight)

    except Exception as e:
        st.error(f"Error processing file: {e}")
