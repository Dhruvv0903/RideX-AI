import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

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


# ---------------- ADAPTIVE MODEL ----------------
def get_adaptive_params(history):
    if len(history) < 3:
        return 1.0, 1.0  # default

    fatigue_trend = history["fatigue"].mean()

    # if user fatigues easily → reduce sensitivity
    fatigue_multiplier = 1.0 + (fatigue_trend / 100)

    # recovery sensitivity
    recovery_rate = max(0.8, 1.2 - (fatigue_trend / 200))

    return fatigue_multiplier, recovery_rate


# ---------------- MAIN ----------------
if uploaded_files:

    history = []

    for file in uploaded_files:

        if file.name.endswith(".tcx"):
            df = parse_tcx(file)
        else:
            continue

        if df["heart_rate"].sum() == 0:
            continue

        rest, max_hr = calibrate_hr(df)

        efforts = []
        for i, row in df.iterrows():
            e = relative_effort(row["heart_rate"], rest, max_hr)
            efforts.append(e)

        df["effort"] = efforts

        avg_effort = np.mean(efforts)

        duration_min = len(df) / 60

        # ---------------- SCALING FIX ----------------
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

    # ---------------- ADAPTIVE ----------------
    fatigue_mult, recovery_rate = get_adaptive_params(history)

    # ---------------- TRAINING LOAD ----------------
    ATL, CTL = [], []
    atl, ctl = 0, 0
    last_date = None

    for _, row in history.iterrows():

        if last_date is None:
            days = 1
        else:
            days = (row["date"] - last_date).days
            days = max(1, min(days, 30))

        decay_atl = pow(0.5, days / 7)
        decay_ctl = pow(0.5, days / 42)

        atl = atl * decay_atl + row["load"] * 0.1 * fatigue_mult
        ctl = ctl * decay_ctl + row["load"] * 0.05 * recovery_rate

        ATL.append(atl)
        CTL.append(ctl)

        last_date = row["date"]

    history["ATL"] = ATL
    history["CTL"] = CTL
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

    tsb = history["TSB"].iloc[-1]

    if tsb > 10:
        st.success("Fresh — push hard")
    elif tsb > -10:
        st.warning("Moderate — train smart")
    else:
        st.error("Fatigued — recover")

    # ---------------- PREDICTION ----------------
    st.subheader("🔮 Tomorrow Prediction")

    atl_now = history["ATL"].iloc[-1]
    ctl_now = history["CTL"].iloc[-1]

    scenarios = {
        "Rest": 0,
        "Light": 20,
        "Hard": 50
    }

    preds = []

    for name, load in scenarios.items():

        new_atl = atl_now * 0.7 + load
        new_ctl = ctl_now * 0.95 + load * 0.5
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

    if tsb < -15:
        st.warning("Rest tomorrow")
    elif tsb < 5:
        st.info("Light ride recommended")
    else:
        st.success("Push hard tomorrow")
