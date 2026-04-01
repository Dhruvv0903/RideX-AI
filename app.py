import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

# ---------------- RIDER PROFILE ----------------
st.sidebar.header("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 60)

max_hr = 220 - age
st.sidebar.write(f"Estimated Max HR: {max_hr}")

# ---------------- LIVE HR FILE ----------------
HR_FILE = "hr_live.txt"

def read_live_hr():
    if os.path.exists(HR_FILE):
        with open(HR_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                try:
                    return int(lines[-1].strip())
                except:
                    return None
    return None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload ride files", type=["csv", "tcx"], accept_multiple_files=True
)

# ---------------- TCX PARSER ----------------
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    data = []
    prev_time = None
    prev_dist = None

    for tp in root.findall('.//ns:Trackpoint', ns):

        time_elem = tp.find('ns:Time', ns)
        dist_elem = tp.find('ns:DistanceMeters', ns)
        hr_elem = tp.find('ns:HeartRateBpm/ns:Value', ns)

        if time_elem is None:
            continue

        curr_time = pd.to_datetime(time_elem.text)
        curr_dist = float(dist_elem.text) if dist_elem is not None else None
        hr = int(hr_elem.text) if hr_elem is not None else 0

        speed = 0

        if prev_time is not None and prev_dist is not None and curr_dist is not None:
            time_diff = (curr_time - prev_time).total_seconds()
            dist_diff = curr_dist - prev_dist

            if time_diff > 0 and dist_diff >= 0:
                speed = (dist_diff / time_diff) * 3.6
                if speed > 80:
                    speed = 0

        data.append({
            "time": curr_time,
            "heart_rate": hr,
            "speed": speed
        })

        prev_time = curr_time
        prev_dist = curr_dist

    df = pd.DataFrame(data)

    if len(df) > 0:
        df["speed"] = df["speed"].rolling(5, min_periods=1).mean()

    return df

# ---------------- LOAD MODEL ----------------
def compute_load(df, max_hr):
    avg_hr = df["heart_rate"].mean()
    duration_hr = len(df) / 3600

    if avg_hr == 0 or max_hr == 0:
        return 0, 0, 0

    intensity = avg_hr / max_hr
    load = (intensity ** 2) * duration_hr * 100

    return load, avg_hr, df["speed"].mean()

# ---------------- MAIN ----------------
if uploaded_files:

    history = []

    for file in uploaded_files:

        if file.name.endswith(".tcx"):
            df = parse_tcx(file)
        else:
            df = pd.read_csv(file)

        if len(df) == 0:
            continue

        load, avg_hr, avg_speed = compute_load(df, max_hr)

        history.append({
            "date": df["time"].iloc[0],
            "load": load,
            "avg_hr": avg_hr,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(history)

    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    st.subheader(f"📁 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD ----------------
    ATL, CTL = 0, 0
    atl_list, ctl_list, tsb_list = [], [], []

    for i in range(len(history)):
        load = history.iloc[i]["load"]

        if i == 0:
            days_gap = 1
        else:
            prev = history.iloc[i-1]["date"]
            curr = history.iloc[i]["date"]
            days_gap = max((curr - prev).days, 1)

        atl_decay = 0.5 ** (days_gap / 7)
        ctl_decay = 0.5 ** (days_gap / 42)

        ATL = ATL * atl_decay + load
        CTL = CTL * ctl_decay + load
        TSB = CTL - ATL

        atl_list.append(ATL)
        ctl_list.append(CTL)
        tsb_list.append(TSB)

    history["ATL"] = atl_list
    history["CTL"] = ctl_list
    history["TSB"] = tsb_list

    current_atl = history["ATL"].iloc[-1]
    current_ctl = history["CTL"].iloc[-1]
    current_tsb = history["TSB"].iloc[-1]

    st.subheader("📊 Training Load")

    c1, c2, c3 = st.columns(3)
    c1.metric("ATL (Fatigue)", round(current_atl, 1))
    c2.metric("CTL (Fitness)", round(current_ctl, 1))
    c3.metric("TSB (Form)", round(current_tsb, 1))

    st.info("""
ATL = short-term fatigue  
CTL = long-term fitness  
TSB = readiness (positive = fresh, negative = fatigued)
""")

    # ---------------- DATE AWARENESS ----------------
    last_date = history["date"].iloc[-1]
    today = pd.Timestamp.now()

    days_since_last = (today - last_date).days
    st.write("📅 Days since last ride:", days_since_last)

    ATL_today = current_atl * (0.5 ** (days_since_last / 7))
    CTL_today = current_ctl * (0.5 ** (days_since_last / 42))
    TSB_today = CTL_today - ATL_today

    # ---------------- READINESS ----------------
    st.subheader("🧠 Readiness")

    if TSB_today > 10:
        st.success("Fresh — ready for hard effort")
    elif TSB_today > -10:
        st.warning("Moderate — train smart")
    else:
        st.error("Fatigued — recovery needed")

# ---------------- LIVE RIDE MODE ----------------
st.divider()
st.header("⚡ Live Ride Mode (REAL DATA)")

if st.button("Start Live Ride"):

    fatigue = 0
    placeholder = st.empty()

    for _ in range(1000):

        hr = read_live_hr()

        if hr is None:
            placeholder.warning("Waiting for HR signal...")
            time.sleep(1)
            continue

        intensity = hr / max_hr
        fatigue += intensity * 0.5

        with placeholder.container():
            st.write(f"❤️ HR: {hr}")
            st.write(f"⚡ Fatigue: {round(fatigue,1)}")

            if fatigue > 60:
                st.error("⚠️ Slow down")
            elif fatigue > 40:
                st.warning("Moderate effort")
            else:
                st.success("Easy pace")

        time.sleep(1)
