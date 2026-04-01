import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

# ---------------- USER INPUT ----------------
st.sidebar.header("👤 Rider Profile")
age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 60)

max_hr = 220 - age

# ---------------- SAFE TCX PARSER ----------------
def parse_tcx_safe(file):
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

        data = []

        for tp in root.findall('.//ns:Trackpoint', ns):
            hr = tp.find('.//ns:HeartRateBpm/ns:Value', ns)
            time_elem = tp.find('.//ns:Time', ns)

            data.append({
                "heart_rate": int(hr.text) if hr is not None else 0,
                "time": pd.to_datetime(time_elem.text) if time_elem is not None else None
            })

        df = pd.DataFrame(data)

        if df.empty:
            return df

        if df["time"].isnull().all():
            df["time"] = pd.date_range(start="2025-01-01", periods=len(df), freq="s")

        return df

    except Exception as e:
        st.error(f"TCX parsing failed: {e}")
        return pd.DataFrame()

# ---------------- LOAD FILE ----------------
def load_file(file):
    if file.name.endswith(".tcx"):
        return parse_tcx_safe(file)
    else:
        df = pd.read_csv(file)
        if "heart_rate" not in df.columns:
            st.error("CSV must have heart_rate column")
            return pd.DataFrame()
        df["time"] = pd.date_range(start="2025-01-01", periods=len(df), freq="s")
        return df

# ---------------- FATIGUE MODEL ----------------
def compute_load(df):
    df = df.copy()

    df["intensity"] = df["heart_rate"] / max_hr
    df["intensity"] = df["intensity"].clip(0, 1)

    # Smarter load (non-linear)
    df["load"] = df["intensity"] ** 2 * 100

    return df

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload ride files",
    type=["csv", "tcx"],
    accept_multiple_files=True
)

history = []

if uploaded_files:

    for file in uploaded_files:
        df = load_file(file)

        if df.empty:
            continue

        df = compute_load(df)

        avg_load = df["load"].mean()
        avg_hr = df["heart_rate"].mean()

        # extract date
        ride_date = df["time"].iloc[0]

        history.append({
            "date": ride_date,
            "load": avg_load,
            "avg_hr": avg_hr
        })

    history = pd.DataFrame(history)

    # ---------------- DATE FIX ----------------
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
    history = history.sort_values("date").reset_index(drop=True)

    st.subheader(f"📁 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD MODEL ----------------
    atl_list = []
    ctl_list = []

    atl = 0
    ctl = 0

    prev_date = None

    for i, row in history.iterrows():

        current_date = row["date"]

        if prev_date is not None:
            gap = (current_date - prev_date).days
        else:
            gap = 1

        gap = max(gap, 1)

        # decay over gap
        atl *= np.exp(-gap / 7)
        ctl *= np.exp(-gap / 42)

        # add load
        atl += row["load"] * (1 - np.exp(-1/7))
        ctl += row["load"] * (1 - np.exp(-1/42))

        atl_list.append(atl)
        ctl_list.append(ctl)

        prev_date = current_date

    history["ATL"] = atl_list
    history["CTL"] = ctl_list
    history["TSB"] = history["CTL"] - history["ATL"]

    # ---------------- METRICS ----------------
    st.subheader("📊 Training Load")

    col1, col2, col3 = st.columns(3)
    col1.metric("ATL (Fatigue)", round(history["ATL"].iloc[-1], 1))
    col2.metric("CTL (Fitness)", round(history["CTL"].iloc[-1], 1))
    col3.metric("TSB (Form)", round(history["TSB"].iloc[-1], 1))

    st.info("""
    ATL = short-term fatigue  
    CTL = long-term fitness  
    TSB = readiness (positive = fresh, negative = fatigued)
    """)

    # ---------------- GRAPH ----------------
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Load")

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------------- GAP AWARENESS ----------------
    today = pd.Timestamp.now().tz_localize(None)
    last_date = history["date"].iloc[-1]

    days_since = (today - last_date).days
    st.write(f"📅 Days since last ride: {days_since}")

    # ---------------- READINESS ----------------
    tsb_now = history["TSB"].iloc[-1]

    if days_since > 5:
        readiness = "Fresh — long break recovered you"
    elif tsb_now > 10:
        readiness = "Fresh — ready for hard effort"
    elif tsb_now > -10:
        readiness = "Moderate — train smart"
    else:
        readiness = "Fatigued — recovery needed"

    st.subheader("🧠 Readiness")
    st.success(readiness)

    # ---------------- TOMORROW PREDICTION ----------------
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = {
        "Rest": 0,
        "Light": 30,
        "Hard": 60
    }

    predictions = []

    for name, load in scenarios.items():

        atl_next = history["ATL"].iloc[-1] * np.exp(-1/7) + load
        ctl_next = history["CTL"].iloc[-1] * np.exp(-1/42) + load
        tsb_next = ctl_next - atl_next

        predictions.append({
            "Scenario": name,
            "ATL": round(atl_next,1),
            "CTL": round(ctl_next,1),
            "TSB": round(tsb_next,1)
        })

    pred_df = pd.DataFrame(predictions)
    st.dataframe(pred_df)

    # recommendation
    if tsb_now < -10:
        st.warning("Rest tomorrow")
    elif tsb_now > 10:
        st.success("Push hard tomorrow")
    else:
        st.info("Do a light ride tomorrow")

# ---------------- LIVE MODE ----------------
st.divider()
st.header("⚡ Live Ride Mode (Simulation)")

uploaded_live = st.file_uploader(
    "Upload file for live simulation",
    type=["csv", "tcx"],
    key="live"
)

if uploaded_live:

    df_live = load_file(uploaded_live)

    if not df_live.empty:

        if st.button("▶️ Start Simulation"):

            placeholder = st.empty()
            fatigue = 0

            for i in range(len(df_live)):

                hr = df_live.iloc[i]["heart_rate"]

                if hr == 0:
                    continue

                intensity = hr / max_hr
                fatigue += intensity * 0.4

                with placeholder.container():
                    st.subheader("🚴 Live Ride")

                    st.metric("Heart Rate", int(hr))
                    st.metric("Fatigue", round(fatigue,1))

                    if intensity < 0.6:
                        st.success("Easy")
                    elif intensity < 0.8:
                        st.warning("Moderate")
                    else:
                        st.error("Hard")

                    if fatigue > 70:
                        st.error("⚠️ Slow down")

                time.sleep(0.05)
