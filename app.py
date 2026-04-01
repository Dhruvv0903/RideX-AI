import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates
from fatigue_model import calculate_fatigue

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

    prev_time = None
    prev_dist = None

    for trackpoint in root.findall('.//ns:Trackpoint', ns):

        time_elem = trackpoint.find('.//ns:Time', ns)
        dist_elem = trackpoint.find('.//ns:DistanceMeters', ns)
        hr = trackpoint.find('.//ns:HeartRateBpm/ns:Value', ns)
        cadence = trackpoint.find('.//ns:Cadence', ns)
        altitude = trackpoint.find('.//ns:AltitudeMeters', ns)

        curr_time = pd.to_datetime(time_elem.text) if time_elem is not None else None
        curr_dist = float(dist_elem.text) if dist_elem is not None else None

        speed = 0.0

        if prev_time is not None and prev_dist is not None and curr_time is not None and curr_dist is not None:
            time_diff = (curr_time - prev_time).total_seconds()
            dist_diff = curr_dist - prev_dist

            if time_diff > 0 and dist_diff >= 0:
                speed = float((dist_diff / time_diff) * 3.6)

                if speed > 80:
                    speed = 0.0

        data.append({
            'time': curr_time,
            'heart_rate': int(hr.text) if hr is not None else 0,
            'cadence': int(cadence.text) if cadence is not None else 0,
            'elevation_m': float(altitude.text) if altitude is not None else 0.0,
            'speed': float(speed)
        })

        prev_time = curr_time
        prev_dist = curr_dist

    df = pd.DataFrame(data)

    if len(df) > 0:
        df['speed'] = df['speed'].rolling(5, min_periods=1).mean()

    return df


# ---------------- NORMALIZATION ----------------
def normalize_data(df):
    df.columns = df.columns.str.lower().str.replace('-', '_')

    for col in ['heart_rate', 'cadence', 'speed', 'elevation_m']:
        if col not in df.columns:
            df[col] = 0

    return df


# ---------------- MAIN ----------------
if uploaded_files:

    history = []

    for file in uploaded_files:

        file_name = file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(file)
            df['time'] = pd.date_range(start='2025-01-01', periods=len(df), freq='S')

        elif file_name.endswith(".tcx"):
            df = parse_tcx(file)

        else:
            continue

        df = normalize_data(df)

        if df['heart_rate'].sum() == 0:
            continue

        df['duration_min'] = df.index / 60

        df['fatigue_score'] = df.apply(
            lambda row: calculate_fatigue(
                row['heart_rate'],
                row['cadence'],
                0,
                row['duration_min'],
                row['elevation_m']
            ),
            axis=1
        )

        avg_fatigue = df['fatigue_score'].mean()
        avg_hr = df['heart_rate'].mean()

        valid_speed = df[df['speed'] > 0]['speed']
        avg_speed = valid_speed.mean() if len(valid_speed) > 0 else None

        ride_date = df['time'].iloc[0]

        history.append({
            "date": ride_date,
            "fatigue": avg_fatigue,
            "load": avg_fatigue * len(df) / 100,
            "avg_hr": avg_hr,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(history)

    if len(history) == 0:
        st.error("No valid rides processed.")
        st.stop()

    # 🧠 CRITICAL: sort by real time
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
                # reset if long break
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
    col1.metric("Fatigue (ATL)", round(history["ATL"].iloc[-1], 1))
    col2.metric("Fitness (CTL)", round(history["CTL"].iloc[-1], 1))
    col3.metric("Form (TSB)", round(history["TSB"].iloc[-1], 1))

    st.info("""
**What do these mean?**
- **ATL (Acute Training Load)** → short-term fatigue  
- **CTL (Chronic Training Load)** → long-term fitness  
- **TSB (Training Stress Balance)** → readiness (fresh vs fatigued)
""")

    # ---------------- GRAPH ----------------
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")

    ax.legend()

    # ✅ FIXED DATE AXIS (with year)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))

    plt.xticks(rotation=30)

    ax.set_xlabel("Date")
    ax.set_ylabel("Load")

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

    # ---------------- SMART PREDICTION ----------------
    st.subheader("🔮 Tomorrow Prediction")

    # ✅ FIX: only recent data (last 14 days)
    recent_history = history[
        history["date"] > (history["date"].max() - pd.Timedelta(days=14))
    ]

    if len(recent_history) > 0:
        current_atl = recent_history["ATL"].iloc[-1]
        current_ctl = recent_history["CTL"].iloc[-1]
    else:
        current_atl = history["ATL"].iloc[-1]
        current_ctl = history["CTL"].iloc[-1]

    scenarios = {
        "Rest Day": 0,
        "Light Ride": 30,
        "Hard Ride": 70
    }

    predictions = []

    for name, load in scenarios.items():
        new_atl = current_atl * 0.7 + load
        new_ctl = current_ctl * 0.95 + load
        tsb = new_ctl - new_atl

        predictions.append({
            "Scenario": name,
            "ATL": round(new_atl, 1),
            "CTL": round(new_ctl, 1),
            "TSB": round(tsb, 1)
        })

    pred_df = pd.DataFrame(predictions)
    st.dataframe(pred_df)

    # ---------------- RECOMMENDATION ----------------
    st.subheader("🧠 Recommendation")

    if latest_tsb < -15:
        st.warning("You should rest tomorrow.")
    elif latest_tsb < 5:
        st.info("Go for a light ride.")
    else:
        st.success("You can push hard tomorrow.")
