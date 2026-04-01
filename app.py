import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")

st.title("🚴 RideX AI — Performance Engine")

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


# ---------------- LOAD PER RIDE ----------------
def compute_load(df):
    avg_hr = df["heart_rate"].mean()
    duration_hr = len(df) / 3600  # seconds → hours

    # normalized load (kept realistic)
    load = avg_hr * duration_hr

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

        load, avg_hr, avg_speed = compute_load(df)

        history.append({
            "date": df["time"].iloc[0],
            "load": load,
            "avg_hr": avg_hr,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(history)

    # 🔥 CRITICAL FIX: datetime conversion
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    st.subheader(f"📁 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD MODEL ----------------
    ATL = 0
    CTL = 0

    atl_list = []
    ctl_list = []
    tsb_list = []

    for i in range(len(history)):
        load = history.iloc[i]["load"]

        if i == 0:
            days_gap = 1
        else:
            prev_date = history.iloc[i-1]["date"]
            curr_date = history.iloc[i]["date"]
            days_gap = max((curr_date - prev_date).days, 1)

        # decay factors
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

    # ---------------- CURRENT STATE ----------------
    current_atl = history["ATL"].iloc[-1]
    current_ctl = history["CTL"].iloc[-1]
    current_tsb = history["TSB"].iloc[-1]

    st.subheader("📊 Training Load")

    col1, col2, col3 = st.columns(3)
    col1.metric("ATL (Fatigue)", round(current_atl, 1))
    col2.metric("CTL (Fitness)", round(current_ctl, 1))
    col3.metric("TSB (Form)", round(current_tsb, 1))

    st.info("""
ATL = short-term fatigue  
CTL = long-term fitness  
TSB = readiness (positive = fresh, negative = fatigued)
""")

    # ---------------- PLOT ----------------
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots()

    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.set_xlabel("Date")
    ax.set_ylabel("Load")
    ax.legend()

    plt.xticks(rotation=30)

    st.pyplot(fig)

    # ---------------- REAL DATE AWARENESS ----------------
    last_date = history["date"].iloc[-1]

    if last_date.tzinfo is not None:
        today = pd.Timestamp.now(tz=last_date.tzinfo)
    else:
        today = pd.Timestamp.now()

    days_since_last = (today - last_date).days

    st.write("📅 Days since last ride:", days_since_last)

    # decay to today
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

    # ---------------- TOMORROW PREDICTION ----------------
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = {
        "Rest": 0,
        "Light": 30,
        "Hard": 60
    }

    pred = []

    for name, load in scenarios.items():
        new_atl = ATL_today * 0.5 + load
        new_ctl = CTL_today * 0.9 + load
        new_tsb = new_ctl - new_atl

        pred.append({
            "Scenario": name,
            "ATL": round(new_atl, 1),
            "CTL": round(new_ctl, 1),
            "TSB": round(new_tsb, 1)
        })

    pred_df = pd.DataFrame(pred)
    st.dataframe(pred_df)

    # ---------------- RECOMMENDATION ----------------
    st.subheader("🧠 Recommendation")

    if TSB_today < -15:
        st.warning("Rest tomorrow")
    elif TSB_today < 5:
        st.info("Light ride recommended")
    else:
        st.success("You can push hard tomorrow")
