import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from fatigue_model import calculate_fatigue

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RideX AI", layout="wide")

st.title("🚴 RideX AI — Performance Engine")

# ---------------- SAMPLE FILE ----------------
try:
    with open("exercise_tcx_file.tcx", "rb") as f:
        tcx_data = f.read()

    st.download_button(
        label="📄 Download Sample Ride",
        data=tcx_data,
        file_name="ridex_sample.tcx",
        mime="application/xml"
    )
except:
    st.warning("Sample file not found")

uploaded_file = st.file_uploader("Upload ride file", type=["csv", "tcx"])

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

        speed = 0
        if prev_time is not None and prev_dist is not None and curr_time is not None and curr_dist is not None:
            t = (curr_time - prev_time).total_seconds()
            d = curr_dist - prev_dist
            if t > 0 and d >= 0:
                speed = (d / t) * 3.6
                if speed > 80:
                    speed = 0

        data.append({
            "heart_rate": int(hr.text) if hr is not None else 0,
            "speed": speed
        })

        prev_time, prev_dist = curr_time, curr_dist

    df = pd.DataFrame(data)
    df["speed"] = df["speed"].rolling(5, min_periods=1).mean()
    return df

# ---------------- MAIN ----------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        if uploaded_file.name.endswith(".tcx"):
            df = parse_tcx(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df["duration_min"] = df.index / 60

        df["fatigue_score"] = df.apply(
            lambda row: calculate_fatigue(
                row.get("heart_rate", 0),
                0,
                0,
                row["duration_min"],
                0
            ), axis=1
        )

        # ---------------- SAVE HISTORY ----------------
        history_file = "ride_history.csv"

        new_entry = {
            "date": datetime.now(),
            "avg_fatigue": df["fatigue_score"].mean(),
            "avg_hr": df["heart_rate"].mean(),
            "avg_speed": df["speed"].mean() if "speed" in df.columns else 0
        }

        new_df = pd.DataFrame([new_entry])

        if os.path.exists(history_file):
            hist = pd.read_csv(history_file)
            hist = pd.concat([hist, new_df], ignore_index=True)
        else:
            hist = new_df

        hist.to_csv(history_file, index=False)

    # Load updated history ONCE after all uploads
    hist = pd.read_csv("ride_history.csv")

    # ---------------- DEBUG (REMOVE LATER) ----------------
    st.write("History length:", len(hist))
    st.dataframe(hist.tail())

    # ---------------- TRAINING LOAD ----------------
    st.subheader("📊 Training Load (Advanced)")

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")

    atl = []
    ctl = []

    atl_value = 0
    ctl_value = 0

    for _, row in hist.iterrows():
        load = row["avg_fatigue"]

        atl_value = atl_value + (load - atl_value) * (1/7)
        ctl_value = ctl_value + (load - ctl_value) * (1/30)

        atl.append(atl_value)
        ctl.append(ctl_value)

    hist["ATL"] = atl
    hist["CTL"] = ctl
    hist["TSB"] = hist["CTL"] - hist["ATL"]

    latest_row = hist.iloc[-1]

    acute = latest_row["ATL"]
    chronic = latest_row["CTL"]
    balance = latest_row["TSB"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Fatigue (ATL)", round(acute,1))
    col2.metric("Fitness (CTL)", round(chronic,1))
    col3.metric("Form (TSB)", round(balance,1))

    # ---------------- READINESS ----------------
    st.subheader("🧠 Readiness")

    if balance > 10:
        st.success("Fresh — ready for high intensity")
    elif balance > -5:
        st.info("Balanced — maintain training")
    else:
        st.warning("Fatigued — recovery needed")

    # ---------------- GRAPH FIX ----------------
    st.subheader("📈 Training Load Trend")

    if len(hist) > 1:
        fig, ax = plt.subplots()

        ax.plot(hist["ATL"], label="Fatigue (ATL)")
        ax.plot(hist["CTL"], label="Fitness (CTL)")
        ax.plot(hist["TSB"], label="Form (TSB)")

        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Upload more rides to see training trends.")

    # ---------------- CURRENT ----------------
    st.subheader("📍 Current Ride")
    st.write(f"Fatigue Score: {round(latest,1)}")
