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
Upload a CSV file with the following columns:

- heart_rate (required)
- cadence (optional)
- slope (optional)
- speed (optional)

Each row should represent a moment in your ride.
""")

# 🔥 NEW UX LINE (what your dad was missing)
st.info("💡 Tip: Export your ride data from apps like Strava, Garmin, or Fitbit as CSV and upload it here.")

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

    low_cadence_pct = (df['cadence'] < 70).sum() / len(df) * 100 if 'cadence' in df else 0

    if max_fatigue > 70:
        insights.append("⚠️ You entered high fatigue levels — recovery ride recommended.")

    if avg_fatigue > 50:
        insights.append("🔥 Sustained effort was high — monitor recovery tomorrow.")

    if 'cadence' in df and low_cadence_pct > 30:
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

        # Clean column names
        df.columns = df.columns.str.lower().str.replace('-', '_')

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        # Validate required columns
        if 'heart_rate' not in df.columns:
            st.error("Missing required column: heart_rate")
            st.stop()

        # Add missing optional columns
        if 'cadence' not in df:
            df['cadence'] = 0
        if 'slope' not in df:
            df['slope'] = 0
        if 'elevation_m' not in df:
            df['elevation_m'] = 0

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

        # ---------------- CURRENT STATE ----------------
        st.subheader("📍 Current State")
        st.write(f"Fatigue Level: {fatigue_zone(latest)}")

        # ---------------- GRAPH ----------------
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

        if 'speed' in df.columns:
            col3.metric("Avg Speed", round(df['speed'].mean(), 1))

        # ---------------- ZONES ----------------
        low = (df['fatigue_score'] < 40).sum()
        moderate = ((df['fatigue_score'] >= 40) & (df['fatigue_score'] < 65)).sum()
        high = (df['fatigue_score'] >= 65).sum()

        total = len(df)

        st.subheader("⏱ Time in Zones")
        st.write(f"🟢 Low: {round(low/total*100,1)}%")
        st.write(f"🟡 Moderate: {round(moderate/total*100,1)}%")
        st.write(f"🔴 High: {round(high/total*100,1)}%")

        # ---------------- SUMMARY ----------------
        st.subheader("📊 Ride Summary")
        st.write(f"Peak Fatigue: {df['fatigue_score'].max()}")
        st.write(f"Time in High Fatigue: {high} data points")

        # ---------------- FINAL VERDICT ----------------
        st.subheader("🏁 Final Verdict")

        if latest < 40:
            st.success("Endurance ride — you can train again tomorrow.")
        elif latest < 65:
            st.warning("Moderate strain — consider a light recovery ride next.")
        else:
            st.error("High fatigue — recovery strongly recommended.")

        # ---------------- INSIGHTS ----------------
        st.subheader("🧠 Insights")

        insights = generate_insights(df)
        for insight in insights:
            st.write("- " + insight)

        # ---------------- KEY TAKEAWAY ----------------
        st.subheader("💡 Key Takeaway")
        st.write("Your fatigue was primarily influenced by intensity, cadence, and sustained effort.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
