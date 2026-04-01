import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.dates as mdates

st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

uploaded_files = st.file_uploader(
    "Upload ride files",
    type=["tcx"],
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

        curr_time = pd.to_datetime(time_elem.text) if time_elem is not None else None
        curr_dist = float(dist_elem.text) if dist_elem is not None else None

        speed = 0.0

        if prev_time is not None and prev_dist is not None and curr_time is not None and curr_dist is not None:
            dt = (curr_time - prev_time).total_seconds()
            dd = curr_dist - prev_dist

            if dt > 0 and dd >= 0:
                speed = (dd / dt) * 3.6
                if speed > 80:
                    speed = 0

        data.append({
            "time": curr_time,
            "heart_rate": int(hr.text) if hr is not None else 0,
            "speed": speed
        })

        prev_time, prev_dist = curr_time, curr_dist

    df = pd.DataFrame(data)

    if len(df) > 0:
        df["speed"] = df["speed"].rolling(5, min_periods=1).mean()

    return df


# ---------------- HR CALIBRATION ----------------
def calibrate_hr(df):
    valid = df[df["heart_rate"] > 0]["heart_rate"]

    if len(valid) < 10:
        return 60, 180

    return valid.quantile(0.05), valid.quantile(0.95)


def relative_effort(hr, rest, max_hr):
    if max_hr - rest <= 0:
        return 0
    return max(0, min((hr - rest) / (max_hr - rest), 1))


# ---------------- RECOMMENDATION ENGINE ----------------
def generate_workout(tsb):
    if tsb > 10:
        return "🔥 Hard Ride: 60 min | High intensity intervals | Cadence 85+"
    elif tsb > -10:
        return "⚡ Moderate Ride: 45 min | Steady pace | Cadence 80–90"
    else:
        return "🛌 Recovery Ride: 30 min | Very easy | Cadence 85+ OR Rest"


# ---------------- MAIN ----------------
if uploaded_files:

    history = []

    for file in uploaded_files:

        df = parse_tcx(file)

        if df["heart_rate"].sum() == 0:
            continue

        rest, max_hr = calibrate_hr(df)

        efforts = df["heart_rate"].apply(lambda x: relative_effort(x, rest, max_hr))
        df["effort"] = efforts

        avg_effort = efforts.mean()
        duration_min = len(df) / 60

        avg_fatigue = avg_effort * 100
        load = avg_fatigue * (duration_min / 60)

        avg_speed = df[df["speed"] > 0]["speed"].mean()

        history.append({
            "date": df["time"].iloc[0],
            "fatigue": avg_fatigue,
            "load": load,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(history).sort_values("date").reset_index(drop=True)

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
            days = max(1, days)

        decay_atl = np.exp(-days / 7)
        decay_ctl = np.exp(-days / 42)

        atl = atl * decay_atl + row["load"] * 0.1
        ctl = ctl * decay_ctl + row["load"] * 0.05

        ATL.append(atl)
        CTL.append(ctl)

        last_date = row["date"]

    history["ATL"] = ATL
    history["CTL"] = CTL
    history["TSB"] = history["CTL"] - history["ATL"]

    # ---------------- REAL CURRENT STATE ----------------
    today = pd.Timestamp.now()
    days_since_last = (today - history["date"].iloc[-1]).days

    atl = atl * np.exp(-days_since_last / 7)
    ctl = ctl * np.exp(-days_since_last / 42)
    tsb = ctl - atl

    # ---------------- METRICS ----------------
    st.subheader("📊 Training Load")

    col1, col2, col3 = st.columns(3)
    col1.metric("ATL (Fatigue)", round(atl, 1))
    col2.metric("CTL (Fitness)", round(ctl, 1))
    col3.metric("TSB (Form)", round(tsb, 1))

    st.info("""
ATL = short-term fatigue  
CTL = long-term fitness  
TSB = readiness
""")

    # ---------------- GRAPH ----------------
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots()

    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    plt.xticks(rotation=30)

    st.pyplot(fig)

    # ---------------- READINESS ----------------
    st.subheader("🧠 Readiness")

    if tsb > 10:
        st.success("Fresh — ready for hard effort")
    elif tsb > -10:
        st.warning("Moderate — train smart")
    else:
        st.error("Fatigued — recovery needed")

    # ---------------- TOMORROW ----------------
    st.subheader("🔮 Tomorrow Prediction")

    atl_now, ctl_now = atl, ctl

    scenarios = {
        "Rest": 0,
        "Light": 20,
        "Hard": 50
    }

    preds = []

    for name, load_val in scenarios.items():

        new_atl = atl_now * 0.7 + load_val
        new_ctl = ctl_now * 0.95 + load_val * 0.5
        new_tsb = new_ctl - new_atl

        preds.append({
            "Scenario": name,
            "ATL": round(new_atl, 1),
            "CTL": round(new_ctl, 1),
            "TSB": round(new_tsb, 1)
        })

    st.dataframe(pd.DataFrame(preds))

    # ---------------- RECOMMENDATION ----------------
    st.subheader("🧠 Recommendation")

    workout = generate_workout(tsb)
    st.success(workout)
