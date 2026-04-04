import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from fatigue_model import compute_fatigue
from strava_api import (
    get_auth_url,
    exchange_code_for_token,
    get_activities,
    refresh_access_token
)
from data_loader import load_from_device

st.set_page_config(layout="wide")

# ==============================
# TITLE
# ==============================
st.title("🚴 RideX AI — Performance Engine")

# ==============================
# PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

max_hr = 220 - age

# ==============================
# STRAVA CONNECT
# ==============================
st.sidebar.subheader("🔗 Connect")

if st.sidebar.button("Connect Strava"):
    st.markdown(f"[Authorize Strava]({get_auth_url()})")

# ==============================
# HANDLE AUTH CODE (FIXED LOOP)
# ==============================
query_params = st.query_params
code = query_params.get("code", None)

if code and "access_token" not in st.session_state:
    token_data = exchange_code_for_token(code)

    if "access_token" in token_data:
        st.session_state["access_token"] = token_data["access_token"]
        st.session_state["refresh_token"] = token_data["refresh_token"]
        st.session_state["expires_at"] = token_data["expires_at"]

        # 🔥 CRITICAL FIX — removes infinite loop
        st.query_params.clear()

        st.success("✅ Strava Connected")

    else:
        st.error("❌ Strava connection failed")
        st.write(token_data)

# ==============================
# AUTO REFRESH TOKEN
# ==============================
if "access_token" in st.session_state:
    if time.time() > st.session_state["expires_at"]:
        new_tokens = refresh_access_token(st.session_state["refresh_token"])

        st.session_state["access_token"] = new_tokens["access_token"]
        st.session_state["refresh_token"] = new_tokens["refresh_token"]
        st.session_state["expires_at"] = new_tokens["expires_at"]

# ==============================
# 📥 UPLOAD DATA (RESTORED)
# ==============================
st.subheader("📥 Upload Ride Files")

uploaded_files = st.file_uploader(
    "Upload TCX/CSV files",
    type=["tcx", "csv"],
    accept_multiple_files=True
)

# ==============================
# LOAD DATA (PRIORITY SYSTEM)
# ==============================
history = []
all_hr = []

# 🔥 PRIORITY 1: UPLOAD
if uploaded_files:

    for file in uploaded_files:
        df = load_from_device()  # placeholder for now

        avg_hr = df["hr"].mean()

        history.append({
            "date": pd.Timestamp.now(),
            "load": avg_hr * 0.6,
            "avg_hr": avg_hr
        })

        all_hr.extend(df["hr"].tolist())

# 🔥 PRIORITY 2: STRAVA
elif "access_token" in st.session_state:

    activities = get_activities(st.session_state["access_token"])

    for act in activities[:5]:
        if act.get("type") != "Ride":
            continue

        hr = act.get("average_heartrate", None)

        if hr is None:
            continue

        history.append({
            "date": pd.to_datetime(act["start_date"]),
            "load": hr * 0.6,
            "avg_hr": hr
        })

        all_hr.append(hr)

# 🔥 PRIORITY 3: SAMPLE
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
# HISTORY TABLE
# ==============================
history_df = pd.DataFrame(history).sort_values("date")

st.subheader("📂 Ride History")
st.dataframe(history_df)

# ==============================
# LOAD METRICS
# ==============================
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
# GRAPH (RESTORED)
# ==============================
fig, ax = plt.subplots()

ax.plot(history_df["date"], history_df["ATL"], label="ATL")
ax.plot(history_df["date"], history_df["CTL"], label="CTL")
ax.plot(history_df["date"], history_df["TSB"], label="TSB")

ax.legend()
ax.set_title("Training Load Trends")

st.pyplot(fig)

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
# LIVE MODE (FIXED)
# ==============================
st.subheader("⚡ Live Ride Mode")

df_live = None

live_mode = st.radio("Live Source", ["Upload", "Sample"])

if live_mode == "Upload":
    lf = st.file_uploader("Upload live file", type=["tcx", "csv"])

    if lf is not None:
        df_live = load_from_device()

else:
    df_live = load_from_device()

# ==============================
# LIVE SIMULATION
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
