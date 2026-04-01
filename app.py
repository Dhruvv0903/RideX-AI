import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from io import BytesIO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

# ---------------- USER INPUT ----------------
st.sidebar.header("👤 Rider Profile")
age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 60)

max_hr = 220 - age

# ---------------- CACHE LOADER ----------------
@st.cache_data
def load_file_cached(file_bytes, filename):
    file = BytesIO(file_bytes)

    if filename.endswith(".tcx"):
        return parse_tcx_safe(file)
    else:
        df = pd.read_csv(file)

        if "heart_rate" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.date_range(start="2025-01-01", periods=len(df), freq="s")
        return df

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

    except Exception:
        return pd.DataFrame()

# ---------------- FATIGUE MODEL ----------------
def compute_load(df):
    df = df.copy()
    df["intensity"] = df["heart_rate"] / max_hr
    df["intensity"] = df["intensity"].clip(0, 1)
    df["load"] = df["intensity"] ** 2 * 100
    return df

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload ride files",
    type=["csv", "tcx"],
    accept_multiple_files=True
)

if uploaded_files:

    history = []

    for file in uploaded_files:
        df = load_file_cached(file.read(), file.name)

        if df.empty:
            continue

        # 🔥 DOWNSAMPLING (huge speed boost)
        df = df.iloc[::5]

        df = compute_load(df)

        history.append({
            "date": df["time"].iloc[0],
            "load": df["load"].mean(),
            "avg_hr": df["heart_rate"].mean()
        })

    history = pd.DataFrame(history)

    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
    history = history.sort_values("date").reset_index(drop=True)

    st.session_state.history = history

# ---------------- USE STORED HISTORY ----------------
history = st.session_state.history

if history is not None:

    st.subheader(f"📁 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD ----------------
    atl, ctl = 0, 0
    atl_list, ctl_list = [], []

    prev_date = None

    for _, row in history.iterrows():

        current_date = row["date"]

        if prev_date is not None:
            gap = max((current_date - prev_date).days, 1)
        else:
            gap = 1

        atl *= np.exp(-gap / 7)
        ctl *= np.exp(-gap / 42)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("ATL (Fatigue)", round(history["ATL"].iloc[-1], 1))
    c2.metric("CTL (Fitness)", round(history["CTL"].iloc[-1], 1))
    c3.metric("TSB (Form)", round(history["TSB"].iloc[-1], 1))

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

    # ---------------- TOMORROW ----------------
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = {"Rest":0, "Light":30, "Hard":60}
    preds = []

    for name, load in scenarios.items():
        atl_next = history["ATL"].iloc[-1]*np.exp(-1/7) + load
        ctl_next = history["CTL"].iloc[-1]*np.exp(-1/42) + load
        tsb_next = ctl_next - atl_next

        preds.append({
            "Scenario": name,
            "ATL": round(atl_next,1),
            "CTL": round(ctl_next,1),
            "TSB": round(tsb_next,1)
        })

    st.dataframe(pd.DataFrame(preds))

    if tsb_now < -10:
        st.warning("Rest tomorrow")
    elif tsb_now > 10:
        st.success("Push hard tomorrow")
    else:
        st.info("Light ride recommended")

# ---------------- LIVE MODE ----------------
st.divider()
st.header("⚡ Live Ride Mode")

uploaded_live = st.file_uploader("Upload for simulation", type=["csv","tcx"], key="live")

if uploaded_live:

    df_live = load_file_cached(uploaded_live.read(), uploaded_live.name)

    if not df_live.empty:

        df_live = df_live.iloc[::3]  # faster simulation

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
                    st.metric("Heart Rate", int(hr))
                    st.metric("Fatigue", round(fatigue,1))

                    if intensity < 0.6:
                        st.success("Easy")
                    elif intensity < 0.8:
                        st.warning("Moderate")
                    else:
                        st.error("Hard")

                time.sleep(0.01)
