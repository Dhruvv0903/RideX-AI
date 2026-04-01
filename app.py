import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ==============================
# USER PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

max_hr = 220 - age

# ==============================
# PARSE TCX
# ==============================
def parse_tcx(file):
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    data = []

    for tp in root.findall(".//ns:Trackpoint", ns):
        time_el = tp.find("ns:Time", ns)
        hr_el = tp.find(".//ns:HeartRateBpm/ns:Value", ns)

        if time_el is not None and hr_el is not None:
            data.append({
                "time": pd.to_datetime(time_el.text),
                "hr": float(hr_el.text)
            })

    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["delta"] = df["time"].diff().dt.total_seconds().fillna(1)

    return df

# ==============================
# FATIGUE MODEL
# ==============================
def compute_fatigue(df):
    fatigue = 0
    fatigue_list = []

    for _, row in df.iterrows():
        hr = row["hr"]
        delta = row["delta"]

        intensity = max(0, (hr - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * delta * 0.03
        fatigue -= 0.015 * delta

        fatigue = max(0, min(100, fatigue))

        fatigue_list.append(fatigue)

    df["fatigue"] = fatigue_list
    return df

# ==============================
# MAIN
# ==============================
st.title("🚴 RideX AI — Performance Engine")

uploaded_files = st.file_uploader("Upload TCX files", type=["tcx"], accept_multiple_files=True)

history = []
all_hr = []

if uploaded_files:
    for file in uploaded_files:
        df = parse_tcx(file)

        if not df.empty:
            df = compute_fatigue(df)

            all_hr.extend(df["hr"].tolist())

            history.append({
                "date": df["time"].iloc[-1],
                "load": df["fatigue"].mean(),
                "avg_hr": df["hr"].mean()
            })

# ==============================
# ADAPTIVE ZONES
# ==============================
if len(all_hr) > 20:
    hr_series = pd.Series(all_hr)

    mean_hr = hr_series.mean()
    std_hr = hr_series.std()

    zone_low = mean_hr - 0.5 * std_hr
    zone_high = mean_hr + 0.5 * std_hr

else:
    zone_low = 0.65 * max_hr
    zone_high = 0.75 * max_hr

# ==============================
# HISTORY + TRAINING LOAD
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

    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==============================
    # DATE AWARENESS FIX
    # ==============================
    last_date = pd.to_datetime(history_df["date"].iloc[-1]).tz_localize(None)
    today = pd.Timestamp.now().tz_localize(None)

    days_since = (today - last_date).days
    st.write(f"📅 Days since last ride: {days_since}")

    # ==============================
    # READINESS
    # ==============================
    if days_since > 7:
        readiness = "Fresh — long break recovered you"
    elif TSB > 10:
        readiness = "Fresh — push hard"
    elif TSB < -10:
        readiness = "Fatigued — recover"
    else:
        readiness = "Moderate — train smart"

    st.subheader("🧠 Readiness")
    st.success(readiness)

    # ==============================
    # 🔮 TOMORROW PREDICTION (RESTORED)
    # ==============================
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = []

    for load in [0, 30, 60]:
        atl_next = ATL + (load - ATL) / 7
        ctl_next = CTL + (load - CTL) / 42
        tsb_next = ctl_next - atl_next

        scenarios.append({
            "Scenario": ["Rest", "Light", "Hard"][len(scenarios)],
            "ATL": round(atl_next, 1),
            "CTL": round(ctl_next, 1),
            "TSB": round(tsb_next, 1)
        })

    pred_df = pd.DataFrame(scenarios)
    st.dataframe(pred_df)

    # ==============================
    # RECOMMENDATION
    # ==============================
    if TSB < -10:
        rec = "Rest tomorrow"
    elif TSB < 0:
        rec = "Light ride recommended"
    else:
        rec = "You can push hard tomorrow"

    st.subheader("🧠 Recommendation")
    st.info(rec)

# ==============================
# LIVE MODE (SMART PACING)
# ==============================
st.subheader("⚡ Live Ride Mode — Smart Pacing")

live_file = st.file_uploader("Upload for simulation", type=["tcx"], key="live")

if live_file:
    df_live = parse_tcx(live_file)

    if not df_live.empty:
        df_live = compute_fatigue(df_live)

        if st.button("▶ Start Simulation"):

            placeholder = st.empty()
            smoothed_hr = None

            for i in range(0, len(df_live), 5):
                row = df_live.iloc[i]

                hr = row["hr"]
                fatigue = row["fatigue"]

                if smoothed_hr is None:
                    smoothed_hr = hr
                else:
                    smoothed_hr = 0.85 * smoothed_hr + 0.15 * hr

                # ==========================
                # HR-DRIVEN DECISION (ADAPTIVE)
                # ==========================
                if smoothed_hr < zone_low:
                    decision = "🔥 Push more"

                elif zone_low <= smoothed_hr <= zone_high:
                    decision = "✅ Perfect pacing"

                elif zone_high < smoothed_hr <= zone_high + 10:
                    decision = "⚖️ Strong effort"

                else:
                    decision = "🚨 Too intense"

                # ==========================
                # FATIGUE LIMITER
                # ==========================
                if fatigue > 85:
                    decision = "🛑 STOP — exhaustion imminent"
                elif fatigue > 70:
                    decision = "⚠️ High fatigue — reduce effort"

                # ==========================
                # UI
                # ==========================
                with placeholder.container():
                    st.metric("Heart Rate", f"{int(smoothed_hr)} bpm")
                    st.metric("Fatigue", f"{int(fatigue)} / 100")

                    st.info(f"Adaptive Zone: {int(zone_low)} - {int(zone_high)} bpm")

                    st.success(decision)

                time.sleep(0.08)
