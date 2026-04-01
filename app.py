import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")

st.title("🚴 RideX AI — Performance Engine")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload ride files",
    type=["csv", "tcx"],
    accept_multiple_files=True
)

# ---------------- TCX PARSER ----------------
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    data = []
    prev_time, prev_dist = None, None

    for tp in root.findall('.//ns:Trackpoint', ns):

        time_elem = tp.find('.//ns:Time', ns)
        dist_elem = tp.find('.//ns:DistanceMeters', ns)
        hr = tp.find('.//ns:HeartRateBpm/ns:Value', ns)
        cadence = tp.find('.//ns:Cadence', ns)
        altitude = tp.find('.//ns:AltitudeMeters', ns)

        curr_time = pd.to_datetime(time_elem.text) if time_elem is not None else None
        curr_dist = float(dist_elem.text) if dist_elem is not None else None

        speed = 0.0

        if prev_time is not None and prev_dist is not None and curr_time is not None and curr_dist is not None:
            dt = (curr_time - prev_time).total_seconds()
            dd = curr_dist - prev_dist

            if dt > 0 and dd >= 0:
                speed = (dd / dt) * 3.6
                if speed > 80:
                    speed = 0.0

        data.append({
            "time": curr_time,
            "heart_rate": int(hr.text) if hr is not None else 0,
            "cadence": int(cadence.text) if cadence is not None else 0,
            "elevation_m": float(altitude.text) if altitude is not None else 0,
            "speed": float(speed)
        })

        prev_time, prev_dist = curr_time, curr_dist

    df = pd.DataFrame(data)

    if len(df) > 0:
        df["speed"] = df["speed"].rolling(5, min_periods=1).mean()

    return df


# ---------------- NORMALIZATION ----------------
def normalize(df):
    df.columns = df.columns.str.lower().str.replace("-", "_")
    for col in ["heart_rate", "cadence", "speed", "elevation_m"]:
        if col not in df.columns:
            df[col] = 0
    return df


# ---------------- PERSONALIZATION ----------------
def calibrate_hr(df):
    valid_hr = df[df["heart_rate"] > 0]["heart_rate"]

    if len(valid_hr) < 10:
        return 60, 180  # fallback

    resting = valid_hr.quantile(0.05)
    max_hr = valid_hr.quantile(0.95)

    return resting, max_hr


def compute_relative_effort(hr, resting, max_hr):
    if max_hr - resting <= 0:
        return 0

    effort = (hr - resting) / (max_hr - resting)
    return max(0, min(effort, 1))


def compute_personalized_fatigue(df, resting, max_hr):
    efforts = []

    for i, row in df.iterrows():
        effort = compute_relative_effort(row["heart_rate"], resting, max_hr)

        # fatigue = effort weighted by time + slight elevation influence
        fatigue = effort * (i / 60 + 1) + 0.0005 * row["elevation_m"]

        efforts.append(fatigue * 100)

    return efforts


# ---------------- MAIN ----------------
if uploaded_files:

    history = []

    for file in uploaded_files:

        name = file.name.lower()

        if name.endswith(".tcx"):
            df = parse_tcx(file)
        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            df["time"] = pd.date_range(start="2025-01-01", periods=len(df), freq="S")
        else:
            continue

        df = normalize(df)

        if df["heart_rate"].sum() == 0:
            continue

        # ---------------- HR PERSONALIZATION ----------------
        resting_hr, max_hr = calibrate_hr(df)

        df["fatigue_score"] = compute_personalized_fatigue(df, resting_hr, max_hr)

        avg_fatigue = np.mean(df["fatigue_score"])
        avg_hr = df["heart_rate"].mean()

        valid_speed = df[df["speed"] > 0]["speed"]
        avg_speed = valid_speed.mean() if len(valid_speed) > 0 else None

        ride_date = df["time"].iloc[0]

        history.append({
            "date": ride_date,
            "fatigue": avg_fatigue,
            "load": avg_fatigue * len(df) / 100,
            "avg_hr": avg_hr,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(history)

    if len(history) == 0:
        st.error("No valid rides.")
        st.stop()

    history = history.sort_values("date").reset_index(drop=True)

    st.subheader(f"📂 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD ----------------
    ATL, CTL = [], []
    atl, ctl = 0, 0
    last_date = None

    for _, row in history.iterrows():

        if last_date is None:
            days = 1
        else:
            days = (row["date"] - last_date).days
            if days <= 0:
                days = 1
            elif days > 30:
                atl *= 0.3
                ctl *= 0.7
                days = 1

        decay_atl = pow(0.5, days / 7)
        decay_ctl = pow(0.5, days / 42)

        atl = atl * decay_atl + row["load"]
        ctl = ctl * decay_ctl + row["load"]

        ATL.append(atl)
        CTL.append(ctl)

        last_date = row["date"]

    history["ATL"] = ATL
    history["CTL"] = CTL
    history["TSB"] = history["CTL"] - history["ATL"]

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

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))

    plt.xticks(rotation=30)

    st.pyplot(fig)

    # ---------------- READINESS ----------------
    st.subheader("🧠 Readiness")

    latest_tsb = history["TSB"].iloc[-1]

    if latest_tsb > 10:
        st.success("Fresh — ready for hard effort")
    elif latest_tsb > -10:
        st.warning("Balanced — moderate training recommended")
    else:
        st.error("Fatigued — recovery needed")

    # ---------------- PREDICTION ----------------
    st.subheader("🔮 Tomorrow Prediction")

    recent = history[
        history["date"] > (history["date"].max() - pd.Timedelta(days=14))
    ]

    if len(recent) > 0:
        atl_now = recent["ATL"].iloc[-1]
        ctl_now = recent["CTL"].iloc[-1]
    else:
        atl_now = history["ATL"].iloc[-1]
        ctl_now = history["CTL"].iloc[-1]

    scenarios = {
        "Rest": 0,
        "Light": 30,
        "Hard": 70
    }

    preds = []

    for name, load in scenarios.items():
        new_atl = atl_now * 0.7 + load
        new_ctl = ctl_now * 0.95 + load
        tsb = new_ctl - new_atl

        preds.append({
            "Scenario": name,
            "ATL": round(new_atl, 1),
            "CTL": round(new_ctl, 1),
            "TSB": round(tsb, 1)
        })

    st.dataframe(pd.DataFrame(preds))

    # ---------------- RECOMMENDATION ----------------
    st.subheader("🧠 Recommendation")

    if latest_tsb < -15:
        st.warning("You should rest tomorrow.")
    elif latest_tsb < 5:
        st.info("Go for a light ride.")
    else:
        st.success("You can push hard tomorrow.")
