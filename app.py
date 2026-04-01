import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

st.set_page_config(layout="wide")

# ==============================
# USER PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 60)

max_hr = 220 - age

# HR zones (personalized)
optimal_low = 0.65 * max_hr
optimal_high = 0.75 * max_hr

# ==============================
# PARSE TCX
# ==============================
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    data = []

    for trackpoint in root.findall(".//ns:Trackpoint", ns):
        time_el = trackpoint.find("ns:Time", ns)
        hr_el = trackpoint.find(".//ns:HeartRateBpm/ns:Value", ns)

        if time_el is not None and hr_el is not None:
            data.append({
                "time": pd.to_datetime(time_el.text),
                "hr": float(hr_el.text)
            })

    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    df["delta"] = df["time"].diff().dt.total_seconds().fillna(0)

    return df


# ==============================
# FATIGUE MODEL (FIXED SCALE)
# ==============================
def compute_fatigue(df):
    fatigue = 0
    fatigue_list = []

    for _, row in df.iterrows():
        hr = row["hr"]
        delta = row["delta"]

        intensity = max(0, (hr - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * delta * 0.02   # controlled growth
        fatigue -= 0.015 * delta             # recovery

        fatigue = max(0, min(100, fatigue))  # clamp

        fatigue_list.append(fatigue)

    df["fatigue"] = fatigue_list
    return df


# ==============================
# LOAD FILE
# ==============================
st.title("🚴 RideX AI — Performance Engine")

uploaded_files = st.file_uploader("Upload TCX files", type=["tcx"], accept_multiple_files=True)

history = []

if uploaded_files:
    for file in uploaded_files:
        df = parse_tcx(file)
        if not df.empty:
            df = compute_fatigue(df)

            history.append({
                "date": df["time"].iloc[-1],
                "load": df["fatigue"].mean(),
                "avg_hr": df["hr"].mean()
            })

# ==============================
# HISTORY DISPLAY
# ==============================
if history:
    history_df = pd.DataFrame(history).sort_values("date")

    st.subheader("📂 Ride History")
    st.dataframe(history_df)

    # ==============================
    # TRAINING LOAD (SIMPLIFIED)
    # ==============================
    ATL = history_df["load"].ewm(span=7).mean().iloc[-1]
    CTL = history_df["load"].ewm(span=42).mean().iloc[-1]
    TSB = CTL - ATL

    st.subheader("📊 Training Load")

    col1, col2, col3 = st.columns(3)

    col1.metric("ATL (Fatigue)", f"{ATL:.1f}")
    col2.metric("CTL (Fitness)", f"{CTL:.1f}")
    col3.metric("TSB (Form)", f"{TSB:.1f}")

    # ==============================
    # TREND GRAPH
    # ==============================
    st.subheader("📈 Training Load Trend")

    history_df["ATL"] = history_df["load"].ewm(span=7).mean()
    history_df["CTL"] = history_df["load"].ewm(span=42).mean()
    history_df["TSB"] = history_df["CTL"] - history_df["ATL"]

    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==============================
    # READINESS
    # ==============================
    last_date = history_df["date"].iloc[-1]
    today = pd.Timestamp.now()

    days_since = (today - last_date).days

    st.write(f"📅 Days since last ride: {days_since}")

    if days_since > 7:
        readiness = "Fresh — long break recovered you"
    elif TSB > 10:
        readiness = "Fresh — ready for hard effort"
    elif TSB < -10:
        readiness = "Fatigued — recovery needed"
    else:
        readiness = "Moderate — train smart"

    st.subheader("🧠 Readiness")
    st.success(readiness)

    # ==============================
    # TOMORROW PREDICTION
    # ==============================
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = {
        "Rest": 0,
        "Light": 30,
        "Hard": 60
    }

    pred = []

    for name, load in scenarios.items():
        atl_next = ATL * 0.9 + load
        ctl_next = CTL * 0.98 + load
        tsb_next = ctl_next - atl_next

        pred.append({
            "Scenario": name,
            "ATL": round(atl_next, 1),
            "CTL": round(ctl_next, 1),
            "TSB": round(tsb_next, 1)
        })

    pred_df = pd.DataFrame(pred)
    st.dataframe(pred_df)

    best = pred_df.loc[pred_df["TSB"].idxmax()]["Scenario"]

    st.subheader("🧠 Recommendation")
    st.info(f"{best} ride recommended")

# ==============================
# LIVE RIDE MODE (FIXED LOGIC)
# ==============================
st.subheader("⚡ Live Ride Mode — Pacing Engine")

live_file = st.file_uploader("Upload for simulation", type=["tcx"], key="live")

if live_file:
    df_live = parse_tcx(live_file)

    if not df_live.empty:
        df_live = compute_fatigue(df_live)

        start = st.button("▶ Start Simulation")

        if start:
            placeholder = st.empty()

            for i in range(len(df_live)):
                row = df_live.iloc[i]
                hr = row["hr"]
                fatigue = row["fatigue"]

                # ==============================
                # HR ZONE BASED DECISION (FIXED)
                # ==============================
                zone1 = optimal_low * 0.9
                zone2 = optimal_low
                zone3 = optimal_high
                zone4 = optimal_high * 1.1

                if fatigue > 90:
                    decision = "🛑 STOP — exhaustion imminent"

                elif fatigue > 75:
                    if hr > optimal_high:
                        decision = "⚠️ Overreaching — reduce effort"
                    else:
                        decision = "⚠️ High fatigue — stay easy"

                elif hr < zone1:
                    decision = "🔥 Too easy — push more"

                elif zone1 <= hr < zone2:
                    decision = "🚴 Warm-up — increase slightly"

                elif zone2 <= hr <= zone3:
                    decision = "✅ Optimal zone — perfect pacing"

                elif zone3 < hr <= zone4:
                    decision = "⚖️ Hard effort — controlled"

                else:
                    decision = "🚨 Too intense — back off NOW"

                with placeholder.container():
                    st.metric("Heart Rate", f"{hr:.0f} bpm")
                    st.metric("Fatigue", f"{fatigue:.1f} / 100")

                    st.info(f"Optimal HR Zone: {int(optimal_low)} - {int(optimal_high)} bpm")

                    st.success(decision)

                time.sleep(0.5)  # slowed down
