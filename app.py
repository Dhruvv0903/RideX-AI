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

# ---------------- CACHE ----------------
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
            df.rename(columns={"heart_rate":"hr"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

        return df

    except:
        return pd.DataFrame()

# ---------------- LOAD MODEL ----------------
def compute_load(df):
    df = df.copy()
    df["intensity"] = df["hr"] / max_hr
    df["intensity"] = df["intensity"].clip(0, 1)

    # slightly more realistic scaling
    df["load"] = (df["intensity"]**2) * 80
    return df

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader("Upload ride files", accept_multiple_files=True)

if uploaded:

    history = []

    for file in uploaded:
        df = load_file(file.read(), file.name)

        if df.empty:
            continue

        df = df.iloc[::5]  # speed boost
        df = compute_load(df)

        history.append({
            "date": df["time"].iloc[0],
            "load": df["load"].mean(),
            "avg_hr": df["hr"].mean()
        })

    history = pd.DataFrame(history)
    history = history.sort_values("date").reset_index(drop=True)

    st.subheader("📁 Processed Rides")
    st.dataframe(history)

    # ---------------- ATL / CTL ----------------
    atl, ctl = 0, 0
    atl_list, ctl_list = [], []

    prev_date = None

    for _, row in history.iterrows():
        d = row["date"]

        if prev_date is not None:
            gap = max((d - prev_date).days, 1)
        else:
            gap = 1

        # decay based on REAL time gap
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
    c1.metric("ATL (Fatigue)", round(history["ATL"].iloc[-1],1))
    c2.metric("CTL (Fitness)", round(history["CTL"].iloc[-1],1))
    c3.metric("TSB (Form)", round(history["TSB"].iloc[-1],1))

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
        readiness = "Recovered (long gap)"
    elif tsb > 10:
        readiness = "Fresh"
    elif tsb > -10:
        readiness = "Moderate"
    else:
        readiness = "Fatigued"

    st.subheader("🧠 Readiness")
    st.success(readiness)

    # ---------------- SMART TOMORROW ----------------
    st.subheader("🔮 Tomorrow Prediction")

    preds = []

    for name, load in {"Rest":0, "Light":30, "Hard":60}.items():

        atl_next = history["ATL"].iloc[-1]*np.exp(-1/7) + load
        ctl_next = history["CTL"].iloc[-1]*np.exp(-1/42) + load*0.5
        tsb_next = ctl_next - atl_next

        preds.append({
            "Scenario": name,
            "ATL": round(atl_next,1),
            "CTL": round(ctl_next,1),
            "TSB": round(tsb_next,1)
        })

    st.dataframe(pd.DataFrame(preds))

    if gap > 7:
        st.success("Ease back in (don't go hard)")
    elif tsb < -10:
        st.warning("Rest tomorrow")
    elif tsb > 10:
        st.success("Push hard tomorrow")
    else:
        st.info("Light ride recommended")

# ---------------- LIVE MODE (UPGRADED) ----------------
st.divider()
st.header("⚡ Live Ride Mode")

live = st.file_uploader("Upload for simulation", key="live")

if live:

    df_live = load_file(live.read(), live.name)

    if not df_live.empty:

        df_live = df_live.iloc[::3]

        if st.button("▶ Start Ride"):

            placeholder = st.empty()

            fatigue = 0
            fatigue_rate = 0

            for i in range(len(df_live)):

                hr = df_live.iloc[i]["hr"]

                if hr == 0:
                    continue

                intensity = hr / max_hr

                # fatigue accumulation
                fatigue_rate = intensity * 0.6
                fatigue += fatigue_rate

                # 🔥 FUTURE PREDICTION
                if fatigue_rate > 0:
                    minutes_to_exhaust = max(1, int((100 - fatigue) / (fatigue_rate * 10)))
                else:
                    minutes_to_exhaust = 999

                with placeholder.container():

                    st.metric("Heart Rate", int(hr))
                    st.metric("Fatigue", round(fatigue,1))
                    st.metric("Time to fatigue (min)", minutes_to_exhaust)

                    if intensity < 0.6:
                        st.success("Easy")
                    elif intensity < 0.8:
                        st.warning("Moderate")
                    else:
                        st.error("Too Hard — unsustainable")

                time.sleep(0.01)
