import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time
import numpy as np

st.set_page_config(layout="wide")

# ==============================
# PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

max_hr = 220 - age

# ==============================
# PARSERS
# ==============================
@st.cache_data
def parse_tcx(file_bytes):
    root = ET.fromstring(file_bytes)
    ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    data = []
    for tp in root.findall(".//ns:Trackpoint", ns):
        t = tp.find("ns:Time", ns)
        hr = tp.find(".//ns:HeartRateBpm/ns:Value", ns)

        if t is not None and hr is not None:
            data.append({
                "time": pd.to_datetime(t.text),
                "hr": float(hr.text)
            })

    df = pd.DataFrame(data)

    if not df.empty:
        df["delta"] = df["time"].diff().dt.total_seconds().fillna(1)

    return df


@st.cache_data
def parse_csv(file_bytes):
    from io import StringIO
    df = pd.read_csv(StringIO(file_bytes.decode()))
    df["time"] = pd.to_datetime(df["time"])
    df["delta"] = df["time"].diff().dt.total_seconds().fillna(1)
    return df


@st.cache_data
def compute_fatigue(df, resting_hr, max_hr):
    fatigue = 0
    out = []

    for _, r in df.iterrows():
        intensity = max(0, (r["hr"] - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * r["delta"] * 0.03
        fatigue -= 0.015 * r["delta"]

        fatigue = max(0, min(100, fatigue))
        out.append(fatigue)

    df = df.copy()
    df["fatigue"] = out
    return df


# ==============================
# SAMPLE DATA
# ==============================
def generate_sample_rides():
    rides = []
    base_date = pd.Timestamp.now()

    for d in range(5):
        times = pd.date_range(end=base_date - pd.Timedelta(days=d),
                              periods=200, freq="s")

        hr = 120 + 25 * np.sin(np.linspace(0, 6, 200)) + np.random.normal(0, 3, 200)

        df = pd.DataFrame({
            "time": times,
            "hr": hr.clip(100, 175)
        })

        df["delta"] = 1
        rides.append(df)

    return rides


# ==============================
# MAIN
# ==============================
st.title("🚴 RideX AI — Performance Engine")

mode = st.radio("Mode", ["Upload Files", "Sample Data"])

history = []
all_hr = []

if mode == "Upload Files":
    files = st.file_uploader("Upload TCX/CSV", type=["tcx", "csv"], accept_multiple_files=True)

    if files:
        for f in files:
            b = f.read()
            df = parse_tcx(b) if f.name.endswith(".tcx") else parse_csv(b)

            if not df.empty:
                df = compute_fatigue(df, resting_hr, max_hr)

                all_hr.extend(df["hr"])
                history.append({
                    "date": df["time"].iloc[-1],
                    "load": df["fatigue"].mean(),
                    "avg_hr": df["hr"].mean()
                })

else:
    rides = generate_sample_rides()

    for df in rides:
        df = compute_fatigue(df, resting_hr, max_hr)

        all_hr.extend(df["hr"])
        history.append({
            "date": df["time"].iloc[-1],
            "load": df["fatigue"].mean(),
            "avg_hr": df["hr"].mean()
        })


# ==============================
# ADAPTIVE ZONES
# ==============================
if len(all_hr) > 20:
    s = pd.Series(all_hr)
    zone_low = s.mean() - 0.5 * s.std()
    zone_high = s.mean() + 0.5 * s.std()
else:
    zone_low = 0.65 * max_hr
    zone_high = 0.75 * max_hr


# ==============================
# HISTORY
# ==============================
if history:
    history_df = pd.DataFrame(history).sort_values("date")

    st.subheader("📂 Ride History")
    st.dataframe(history_df)

    history_df["ATL"] = history_df["load"].ewm(span=7).mean()
    history_df["CTL"] = history_df["load"].ewm(span=42).mean()
    history_df["TSB"] = history_df["CTL"] - history_df["ATL"]

    ATL = history_df["ATL"].iloc[-1]
    CTL = history_df["CTL"].iloc[-1]
    TSB = history_df["TSB"].iloc[-1]

    st.subheader("📊 Training Load")

    c1, c2, c3 = st.columns(3)
    c1.metric("ATL", f"{ATL:.1f}")
    c2.metric("CTL", f"{CTL:.1f}")
    c3.metric("TSB", f"{TSB:.1f}")

    st.info("""
    ATL = Acute Training Load (fatigue)  
    CTL = Chronic Training Load (fitness)  
    TSB = Training Stress Balance (readiness)
    """)

    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    # ==============================
    # TOMORROW PREDICTION (FIXED)
    # ==============================
    st.subheader("🔮 Tomorrow Prediction")

    rows = []
    scenarios = ["Rest", "Light", "Hard"]

    for i, load in enumerate([0, 30, 60]):
        atl_n = ATL + (load - ATL) / 7
        ctl_n = CTL + (load - CTL) / 42
        tsb_n = ctl_n - atl_n

        rows.append({
            "Scenario": scenarios[i],
            "ATL": round(atl_n, 1),
            "CTL": round(ctl_n, 1),
            "TSB": round(tsb_n, 1)
        })

    pred_df = pd.DataFrame(rows)
    st.dataframe(pred_df)

    # 🔥 BEST DECISION (THIS WAS MISSING)
    best_row = pred_df.loc[pred_df["TSB"].idxmax()]
    best_scenario = best_row["Scenario"]

    if best_scenario == "Rest":
        st.success("🛑 Recommended: Rest Day")
    elif best_scenario == "Light":
        st.info("🚴 Recommended: Light Ride")
    else:
        st.success("🔥 Recommended: Hard Training Day")


# ==============================
# LIVE MODE
# ==============================
st.subheader("⚡ Live Ride Mode")

live_mode = st.radio("Live Source", ["Upload", "Sample"])

if live_mode == "Upload":
    lf = st.file_uploader("Upload file", type=["tcx", "csv"], key="live")

    if lf:
        b = lf.read()
        df_live = parse_tcx(b) if lf.name.endswith(".tcx") else parse_csv(b)
else:
    df_live = generate_sample_rides()[0]

if "df_live" in locals() and not df_live.empty:
    df_live = compute_fatigue(df_live, resting_hr, max_hr)

    if st.button("▶ Start Simulation"):

        placeholder = st.empty()
        smoothed_hr = None

        for i in range(0, len(df_live), 6):

            row = df_live.iloc[i]
            hr = row["hr"]
            fatigue = row["fatigue"]

            smoothed_hr = hr if smoothed_hr is None else 0.85 * smoothed_hr + 0.15 * hr

            if smoothed_hr < zone_low:
                decision = "🔥 Push more"
            elif zone_low <= smoothed_hr <= zone_high:
                decision = "✅ Perfect pacing"
            elif smoothed_hr <= zone_high + 10:
                decision = "⚖️ Strong effort"
            else:
                decision = "🚨 Too intense"

            if fatigue > 85:
                decision = "🛑 STOP"
            elif fatigue > 70:
                decision = "⚠️ Reduce effort"

            with placeholder.container():
                st.metric("Heart Rate", f"{int(smoothed_hr)} bpm")
                st.metric("Fatigue", f"{int(fatigue)} / 100")
                st.info(f"Adaptive Zone: {int(zone_low)} - {int(zone_high)} bpm")
                st.success(decision)

            time.sleep(0.08)
