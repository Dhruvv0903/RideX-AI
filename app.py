import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")
st.title("🚴 RideX AI — Performance Engine")

# ---------------- FILE UPLOAD ----------------
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

    times, distances, heart_rates, cadences = [], [], [], []

    for tp in root.findall('.//ns:Trackpoint', ns):
        time_elem = tp.find('ns:Time', ns)
        dist_elem = tp.find('ns:DistanceMeters', ns)
        hr_elem = tp.find('.//ns:HeartRateBpm/ns:Value', ns)
        cad_elem = tp.find('ns:Cadence', ns)

        if time_elem is not None:
            times.append(pd.to_datetime(time_elem.text))

        distances.append(float(dist_elem.text) if dist_elem is not None else None)
        heart_rates.append(int(hr_elem.text) if hr_elem is not None else 0)
        cadences.append(int(cad_elem.text) if cad_elem is not None else 0)

    df = pd.DataFrame({
        "time": times,
        "distance": distances,
        "heart_rate": heart_rates,
        "cadence": cadences
    })

    df = df.dropna(subset=["time"])

    # SPEED (vectorized)
    df["time_diff"] = df["time"].diff().dt.total_seconds()
    df["dist_diff"] = df["distance"].diff()

    df["speed"] = (df["dist_diff"] / df["time_diff"]) * 3.6
    df["speed"] = df["speed"].clip(lower=0, upper=80).fillna(0)
    df["speed"] = df["speed"].rolling(5, min_periods=1).mean()

    return df


# ---------------- FATIGUE ----------------
def calculate_fatigue(hr, cadence, duration_min):
    hr_factor = min(hr / 195, 1)
    duration_factor = min(duration_min / 120, 1)

    cadence_penalty = 0.3 if cadence < 70 else 0.1 if cadence < 80 else 0

    fatigue = (
        hr_factor * 50 +
        duration_factor * 40 +
        cadence_penalty * 10
    )

    return min(fatigue, 100)


# ---------------- MAIN ----------------
if uploaded_files:

    rides = []

    for file in uploaded_files:

        df = parse_tcx(file)

        if df.empty:
            continue

        duration_sec = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
        duration_min = duration_sec / 60

        avg_hr = df["heart_rate"].mean()
        avg_cad = df["cadence"].mean()
        avg_speed = df["speed"].mean()

        fatigue = calculate_fatigue(avg_hr, avg_cad, duration_min)

        load = fatigue * duration_min / 60

        rides.append({
            "date": df["time"].iloc[0],
            "fatigue": fatigue,
            "load": load,
            "avg_hr": avg_hr,
            "avg_speed": avg_speed
        })

    history = pd.DataFrame(rides)

    # 🚨 SAFETY CHECK
    if history.empty:
        st.warning("No valid ride data found.")
        st.stop()

    history = history.sort_values("date")

    st.subheader(f"📂 {len(history)} ride(s) processed")
    st.dataframe(history)

    # ---------------- TRAINING LOAD ----------------
    ATL, CTL = [], []
    atl, ctl = 0, 0
    last_date = None

    for _, row in history.iterrows():

        days = 1 if last_date is None else max((row["date"] - last_date).days, 1)

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

    # ---------------- METRICS ----------------
    st.subheader("📊 Training Load")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fatigue (ATL)", round(history["ATL"].iloc[-1], 1))
    col2.metric("Fitness (CTL)", round(history["CTL"].iloc[-1], 1))
    col3.metric("Form (TSB)", round(history["TSB"].iloc[-1], 1))

    # ---------------- GRAPH ----------------
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots()
    ax.plot(history["date"], history["ATL"], label="ATL")
    ax.plot(history["date"], history["CTL"], label="CTL")
    ax.plot(history["date"], history["TSB"], label="TSB")
    ax.legend()
    st.pyplot(fig)

    # ---------------- PREDICTION (SAFE) ----------------
    st.subheader("🔮 Tomorrow Prediction")

    if "ATL" in history.columns and len(history) > 0:

        current_atl = history["ATL"].iloc[-1]
        current_ctl = history["CTL"].iloc[-1]

        decay_atl = pow(0.5, 1 / 7)
        decay_ctl = pow(0.5, 1 / 42)

        scenarios = {
            "Rest Day": 0,
            "Light Ride": 30,
            "Hard Ride": 70
        }

        predictions = []

        for name, load in scenarios.items():
            next_atl = current_atl * decay_atl + load
            next_ctl = current_ctl * decay_ctl + load
            next_tsb = next_ctl - next_atl

            predictions.append({
                "Scenario": name,
                "Predicted ATL": round(next_atl, 1),
                "Predicted CTL": round(next_ctl, 1),
                "Predicted TSB": round(next_tsb, 1)
            })

        pred_df = pd.DataFrame(predictions)
        st.dataframe(pred_df)

    else:
        st.info("Upload ride data to enable predictions.")
