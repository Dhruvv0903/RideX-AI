import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🚴 RideX AI — Performance Engine")

# ---------------- USER ----------------
st.sidebar.header("👤 Rider Profile")
age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 60)

max_hr = 220 - age

# ---------------- LOAD FILE ----------------
@st.cache_data
def load_file(file_bytes, name):
    file = BytesIO(file_bytes)

    try:
        if name.endswith(".tcx"):
            tree = ET.parse(file)
            root = tree.getroot()
            ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

            data = []
            for tp in root.findall('.//ns:Trackpoint', ns):
                hr = tp.find('.//ns:HeartRateBpm/ns:Value', ns)
                time_elem = tp.find('.//ns:Time', ns)

                data.append({
                    "hr": int(hr.text) if hr is not None else 0,
                    "time": pd.to_datetime(time_elem.text) if time_elem is not None else None
                })

            df = pd.DataFrame(data)

        else:
            df = pd.read_csv(file)
            df["time"] = pd.date_range(start="2025-01-01", periods=len(df), freq="s")
            df.rename(columns={"heart_rate": "hr"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        return df

    except:
        return pd.DataFrame()

# ---------------- LOAD MODEL ----------------
def compute_load(df):
    df = df.copy()
    df["intensity"] = df["hr"] / max_hr
    df["intensity"] = df["intensity"].clip(0, 1)
    df["load"] = (df["intensity"] ** 2) * 80
    return df

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader("Upload ride files", accept_multiple_files=True)

if uploaded:

    history = []

    for file in uploaded:
        df = load_file(file.read(), file.name)

        if df.empty:
            continue

        df = df.iloc[::5]
        df = compute_load(df)

        history.append({
            "date": df["time"].iloc[0],
            "load": df["load"].mean(),
            "avg_hr": df["hr"].mean()
        })

    history = pd.DataFrame(history).sort_values("date").reset_index(drop=True)

    st.subheader("📁 Processed Rides")
    st.dataframe(history)

    # ---------------- ATL / CTL ----------------
    atl, ctl = 0, 0
    atl_list, ctl_list = [], []

    prev_date = None

    for _, row in history.iterrows():
        d = row["date"]

        gap = max((d - prev_date).days, 1) if prev_date is not None else 1

        atl *= np.exp(-gap / 7)
        ctl *= np.exp(-gap / 42)

        atl += row["load"] * 0.7
        ctl += row["load"] * 0.3

        atl_list.append(atl)
        ctl_list.append(ctl)

        prev_date = d

    history["ATL"] = atl_list
    history["CTL"] = ctl_list
    history["TSB"] = history["CTL"] - history["ATL"]

    # ---------------- METRICS ----------------
    st.subheader("📊 Training Load")

    c1, c2, c3 = st.columns(3)
    c1.metric("ATL (Fatigue)", round(history["ATL"].iloc[-1], 1))
    c2.metric("CTL (Fitness)", round(history["CTL"].iloc[-1], 1))
    c3.metric("TSB (Form)", round(history["TSB"].iloc[-1], 1))

    st.info("""
    ATL = short-term fatigue  
    CTL = long-term fitness  
    TSB = readiness  
    """)

    # ---------------- GRAPH ----------------
    st.subheader("📈 Trend")

    fig, ax = plt.subplots()
    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------------- GAP ----------------
    today = pd.Timestamp.now()
    last_date = history["date"].iloc[-1]
    gap = (today - last_date).days

    st.write(f"📅 Days since last ride: {gap}")

    tsb = history["TSB"].iloc[-1]

    if gap > 7:
        st.success("Recovered (long gap)")
    elif tsb > 10:
        st.success("Fresh")
    elif tsb > -10:
        st.warning("Moderate")
    else:
        st.error("Fatigued")

# =========================
# ⚡ LIVE RIDE MODE (FIXED)
# =========================

st.divider()
st.header("⚡ Live Ride Mode — Pacing Engine")

live = st.file_uploader("Upload for simulation", key="live")

if live:

    df_live = load_file(live.read(), live.name)

    if not df_live.empty:

        df_live = df_live.iloc[::3]

        if st.button("▶ Start Ride"):

            placeholder = st.empty()
            fatigue = 0

            for i in range(len(df_live)):

                hr = df_live.iloc[i]["hr"]
                if hr == 0:
                    continue

                intensity = hr / max_hr

                # 🔥 FIXED FATIGUE MODEL (bounded)
                fatigue_rate = (intensity ** 3) * 25
                fatigue += fatigue_rate * 0.05
                fatigue = min(fatigue, 100)

                # ⏱ FIXED TTE (never negative)
                if fatigue >= 99:
                    tte = 0
                elif fatigue_rate > 0:
                    tte = int((100 - fatigue) / (fatigue_rate + 1e-6))
                else:
                    tte = 999

                # 🎯 HR ZONE
                optimal_low = int(max_hr * 0.65)
                optimal_high = int(max_hr * 0.75)

                # 🧠 DECISION ENGINE
                if fatigue > 85:
                    decision = "🛑 Near exhaustion — back off immediately"
                elif fatigue > 70:
                    decision = "⚠️ High fatigue — reduce effort"
                elif intensity > 0.85:
                    decision = "⚠️ Unsustainable — will burn out"
                elif intensity > 0.7:
                    decision = "⚖️ Strong pace — monitor fatigue"
                elif intensity > 0.6:
                    decision = "✅ Sustainable endurance pace"
                else:
                    decision = "🔥 You can push harder"

                with placeholder.container():

                    st.metric("Heart Rate", int(hr))
                    st.metric("Fatigue (0-100)", round(fatigue, 1))
                    st.metric("Time to Exhaustion (sec)", tte)

                    st.info(f"🎯 Optimal HR Zone: {optimal_low}-{optimal_high}")
                    st.subheader(decision)

                # 🐢 SLOWER FOR VISUALIZATION
                time.sleep(0.15)
