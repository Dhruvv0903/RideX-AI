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

# Personalized HR zones
optimal_low = 0.65 * max_hr
optimal_high = 0.75 * max_hr

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
# FIXED FATIGUE MODEL
# ==============================
def compute_fatigue(df):
    fatigue = 0
    fatigue_list = []

    for _, row in df.iterrows():
        hr = row["hr"]
        delta = row["delta"]

        intensity = max(0, (hr - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * delta * 0.05   # stronger signal
        fatigue -= 0.01 * delta               # controlled recovery

        fatigue = max(0, min(100, fatigue))

        fatigue_list.append(fatigue)

    df["fatigue"] = fatigue_list
    return df


# ==============================
# MAIN APP
# ==============================
st.title("🚴 RideX AI — Performance Engine")

uploaded_files = st.file_uploader("Upload TCX files", type=["tcx"], accept_multiple_files=True)

history = []

if uploaded_files:
    for file in uploaded_files:
        df = parse_tcx(file)

        if not df.empty:
            df = compute_fatigue(df)

            history.append({
                "date": df["time"].iloc[-1],
                "load": df["fatigue"].mean(),
                "avg_hr": df["hr"].mean()
            })

# ==============================
# HISTORY
# ==============================
if history:
    history_df = pd.DataFrame(history).sort_values("date")

    st.subheader("📂 Ride History")
    st.dataframe(history_df)

    # ==============================
    # TRAINING LOAD
    # ==============================
    history_df["ATL"] = history_df["load"].ewm(span=7).mean()
    history_df["CTL"] = history_df["load"].ewm(span=42).mean()
    history_df["TSB"] = history_df["CTL"] - history_df["ATL"]

    ATL = history_df["ATL"].iloc[-1]
    CTL = history_df["CTL"].iloc[-1]
    TSB = history_df["TSB"].iloc[-1]

    st.subheader("📊 Training Load")

    c1, c2, c3 = st.columns(3)
    c1.metric("ATL (Fatigue)", f"{ATL:.1f}")
    c2.metric("CTL (Fitness)", f"{CTL:.1f}")
    c3.metric("TSB (Form)", f"{TSB:.1f}")

    # ==============================
    # GRAPH
    # ==============================
    st.subheader("📈 Training Load Trend")

    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==============================
    # FIXED DATE HANDLING
    # ==============================
    last_date = pd.to_datetime(history_df["date"].iloc[-1])
    last_date = last_date.tz_localize(None) if last_date.tzinfo else last_date
    today = pd.Timestamp.now().tz_localize(None)

    days_since = (today - last_date).days

    st.write(f"📅 Days since last ride: {days_since}")

    # ==============================
    # READINESS
    # ==============================
    if days_since > 7:
        readiness = "Fresh — long break recovered you"
    elif TSB > 10:
        readiness = "Fresh — ready for hard effort"
    elif TSB < -10:
        readiness = "Fatigued — recovery needed"
    else:
        readiness = "Moderate — train smart"

    st.subheader("🧠 Readiness")
    st.success(readiness)

    # ==============================
    # PREDICTION
    # ==============================
    st.subheader("🔮 Tomorrow Prediction")

    scenarios = {"Rest": 0, "Light": 30, "Hard": 60}
    pred = []

    for name, load in scenarios.items():
        atl_next = ATL * 0.9 + load
        ctl_next = CTL * 0.98 + load
        tsb_next = ctl_next - atl_next

        pred.append({
            "Scenario": name,
            "ATL": round(atl_next, 1),
            "CTL": round(ctl_next, 1),
            "TSB": round(tsb_next, 1)
        })

    pred_df = pd.DataFrame(pred)
    st.dataframe(pred_df)

    best = pred_df.loc[pred_df["TSB"].idxmax()]["Scenario"]

    st.subheader("🧠 Recommendation")
    st.info(f"{best} ride recommended")

# ==============================
# LIVE MODE (FIXED LOGIC)
# ==============================
st.subheader("⚡ Live Ride Mode — Pacing Engine")

live_file = st.file_uploader("Upload for simulation", type=["tcx"], key="live")

if live_file:
    df_live = parse_tcx(live_file)

    if not df_live.empty:
        df_live = compute_fatigue(df_live)

        if st.button("▶ Start Simulation"):
            placeholder = st.empty()

            for _, row in df_live.iterrows():
                hr = row["hr"]
                fatigue = row["fatigue"]

                # Zones
                zone1 = optimal_low * 0.9
                zone2 = optimal_low
                zone3 = optimal_high
                zone4 = optimal_high * 1.1

                # FINAL FIXED DECISION ENGINE
                if fatigue > 90:
                    decision = "🛑 STOP — exhaustion imminent"

                elif fatigue > 75:
                    if hr > optimal_high:
                        decision = "⚠️ Overreaching — reduce effort"
                    else:
                        decision = "⚠️ High fatigue — stay easy"

                elif hr < zone1:
                    decision = "🔥 Too easy — push more"

                elif zone1 <= hr < zone2:
                    decision = "🚴 Warm-up — increase slightly"

                elif zone2 <= hr <= zone3:
                    decision = "✅ Optimal zone — perfect pacing"

                elif zone3 < hr <= zone4:
                    decision = "⚖️ Hard effort — controlled"

                else:
                    decision = "🚨 Too intense — back off NOW"

                with placeholder.container():
                    st.metric("Heart Rate", f"{hr:.0f} bpm")
                    st.metric("Fatigue", f"{fatigue:.1f} / 100")

                    st.info(f"Optimal HR Zone: {int(optimal_low)} - {int(optimal_high)} bpm")

                    st.success(decision)

                time.sleep(0.7)  # slower, readable
