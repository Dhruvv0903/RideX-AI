import streamlit as st
import pandas as pd
import numpy as np
import time

from fatigue_model import compute_fatigue
from strava_api import get_auth_url, exchange_code_for_token, get_activities
from data_loader import load_from_device

st.set_page_config(layout="wide")

# ==============================
# PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

training_goal = st.sidebar.selectbox(
    "Training Goal",
    ["Build Fitness", "Maintain", "Recover"]
)

max_hr = 220 - age

# ==============================
# STRAVA CONNECT
# ==============================
st.sidebar.subheader("🔗 Connect")

if st.sidebar.button("Connect Strava"):
    st.markdown(f"[Click here to authorize]({get_auth_url()})")

code = st.sidebar.text_input("Paste Strava Code")

history = []
all_hr = []

# ==============================
# LOAD FROM STRAVA
# ==============================
if code:
    token_data = exchange_code_for_token(code)

    # 🔴 DEBUG (DO NOT REMOVE YET)
    st.write(token_data)

    if "access_token" in token_data:
        access_token = token_data["access_token"]
        activities = get_activities(access_token)

        for act in activities[:5]:
            if act["type"] != "Ride":
                continue

            hr = act.get("average_heartrate", 120)

            history.append({
                "date": pd.to_datetime(act["start_date"]),
                "load": hr * 0.6,
                "avg_hr": hr
            })

            all_hr.append(hr)

    else:
        st.error("❌ Strava connection failed")

# ==============================
# FALLBACK SAMPLE DATA
# ==============================
if not history:
    st.warning("Using sample data")

    for i in range(5):
        hr = 120 + np.random.randint(-10, 20)

        history.append({
            "date": pd.Timestamp.now() - pd.Timedelta(days=i),
            "load": hr * 0.5,
            "avg_hr": hr
        })

        all_hr.append(hr)

# ==============================
# ADAPTIVE ZONES
# ==============================
s = pd.Series(all_hr)

zone_low = s.mean() - 0.5 * s.std()
zone_high = s.mean() + 0.5 * s.std()

# ==============================
# HISTORY + LOAD
# ==============================
history_df = pd.DataFrame(history).sort_values("date")

st.subheader("📂 Ride History")
st.dataframe(history_df)

history_df["ATL"] = history_df["load"].ewm(span=7).mean()
history_df["CTL"] = history_df["load"].ewm(span=42).mean()
history_df["TSB"] = history_df["CTL"] - history_df["ATL"]

ATL = history_df["ATL"].iloc[-1]
CTL = history_df["CTL"].iloc[-1]
TSB = history_df["TSB"].iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("ATL", f"{ATL:.1f}")
c2.metric("CTL", f"{CTL:.1f}")
c3.metric("TSB", f"{TSB:.1f}")

st.info("""
ATL = Acute Training Load (short-term fatigue)  
CTL = Chronic Training Load (long-term fitness)  
TSB = Training Stress Balance (readiness)
""")

# ==============================
# DAYS SINCE LAST RIDE
# ==============================
last_date = pd.to_datetime(history_df["date"].iloc[-1]).tz_localize(None)
today = pd.Timestamp.now().tz_localize(None)
gap = (today - last_date).days

st.write(f"📅 Days since last ride: {gap}")

# ==============================
# READINESS
# ==============================
if gap > 7:
    readiness = "Fresh — recovered"
elif TSB > 10:
    readiness = "Fresh — push hard"
elif TSB < -10:
    readiness = "Fatigued — recover"
else:
    readiness = "Moderate — train smart"

st.subheader("🧠 Readiness")
st.success(readiness)

# ==============================
# TOMORROW PREDICTION
# ==============================
st.subheader("🔮 Tomorrow Prediction")

scenarios = ["Rest", "Light", "Hard"]
loads = [0, 30, 60]
rows = []

for i in range(3):
    load = loads[i]

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

best = pred_df.loc[pred_df["TSB"].idxmax()]["Scenario"]

if gap > 5:
    final = "Light"
else:
    final = best

if TSB < -10:
    final = "Rest"

st.subheader("🧠 Tomorrow Recommendation")

if final == "Rest":
    st.warning("🛑 Rest Day Recommended")
elif final == "Light":
    st.info("🚴 Light Ride Recommended")
else:
    st.success("🔥 Hard Training Day Recommended")

# ==============================
# 3 DAY PLAN
# ==============================
st.subheader("🚀 3-Day Training Plan")

plan = ["Rest", "Light", "Hard"]

c1, c2, c3 = st.columns(3)
c1.metric("Day 1", plan[0])
c2.metric("Day 2", plan[1])
c3.metric("Day 3", plan[2])

# ==============================
# ⚡ LIVE MODE (FIXED)
# ==============================
st.subheader("⚡ Live Ride Mode")

live_mode = st.radio("Live Source", ["Upload", "Sample"])

if live_mode == "Upload":
    lf = st.file_uploader("Upload file", type=["tcx", "csv"], key="live")

    if lf:
        df_live = load_from_device()
else:
    df_live = load_from_device()

# ==============================
# RUN SIMULATION
# ==============================
if df_live is not None and not df_live.empty:

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
