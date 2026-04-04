import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time

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

# ✅ RESTORED TRAINING GOAL
training_goal = st.sidebar.selectbox(
    "Training Goal",
    ["Build Fitness", "Maintain", "Recover"]
)

max_hr = 220 - age

# ==============================
# ✅ DATA SOURCE SELECTOR (ADDED)
# ==============================
data_mode = st.sidebar.radio(
    "Data Source",
    ["Upload Files", "Strava", "Sample Data"]
)

# ==============================
# TCX PARSER
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

# ==============================
# CSV PARSER
# ==============================
@st.cache_data
def parse_csv(file_bytes):
    from io import StringIO
    df = pd.read_csv(StringIO(file_bytes.decode()))
    df["time"] = pd.to_datetime(df["time"])
    df["delta"] = df["time"].diff().dt.total_seconds().fillna(1)
    return df

# ==============================
# STRAVA CONNECT
# ==============================
st.sidebar.subheader("🔗 Connect")

if st.sidebar.button("Connect Strava"):
    st.markdown(f"[Authorize Strava]({get_auth_url()})")

query_params = st.query_params
code = query_params.get("code", None)

if code and "access_token" not in st.session_state:
    token_data = exchange_code_for_token(code)

    if "access_token" in token_data:
        st.session_state["access_token"] = token_data["access_token"]
        st.session_state["refresh_token"] = token_data["refresh_token"]
        st.session_state["expires_at"] = token_data["expires_at"]

        st.query_params.clear()
        st.success("✅ Strava Connected")

    else:
        st.error("❌ Strava connection failed")
        st.write(token_data)

# ==============================
# TOKEN REFRESH
# ==============================
if "access_token" in st.session_state:
    if time.time() > st.session_state["expires_at"]:
        new_tokens = refresh_access_token(st.session_state["refresh_token"])

        st.session_state["access_token"] = new_tokens["access_token"]
        st.session_state["refresh_token"] = new_tokens["refresh_token"]
        st.session_state["expires_at"] = new_tokens["expires_at"]

# ==============================
# DATA PIPELINE (CONTROLLED)
# ==============================
history = []
all_hr = []

# 🔵 UPLOAD MODE
if data_mode == "Upload Files":

    st.subheader("📥 Upload Ride Files")

    uploaded_files = st.file_uploader(
        "Upload TCX/CSV",
        type=["tcx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
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
        st.warning("Upload files to proceed")

# 🟢 STRAVA MODE
elif data_mode == "Strava":

    if "access_token" in st.session_state:

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

    else:
        st.warning("Connect Strava to load data")

# 🟡 SAMPLE MODE
elif data_mode == "Sample Data":

    st.info("Using sample data")

    for i in range(5):
        hr = 120 + np.random.randint(-10, 20)

        history.append({
            "date": pd.Timestamp.now() - pd.Timedelta(days=i),
            "load": hr * 0.5,
            "avg_hr": hr
        })

        all_hr.append(hr)

# ==============================
# SAFETY STOP (ADDED)
# ==============================
if not history:
    st.stop()

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

# ==============================
# GRAPH
# ==============================
fig, ax = plt.subplots()

ax.plot(history_df["date"], history_df["ATL"], label="ATL")
ax.plot(history_df["date"], history_df["CTL"], label="CTL")
ax.plot(history_df["date"], history_df["TSB"], label="TSB")

ax.legend()
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
# 🔮 TOMORROW PREDICTION (RESTORED)
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

# ==============================
# 🧠 DECISION ENGINE (RESTORED)
# ==============================
best_row = pred_df.loc[pred_df["TSB"].idxmax()]
best = best_row["Scenario"]

# gap correction
if gap > 5:
    final = "Light"
else:
    final = best

# fatigue override
if TSB < -10:
    final = "Rest"

# ==============================
# FINAL OUTPUT
# ==============================
st.subheader("🧠 Tomorrow Recommendation")

if final == "Rest":
    st.warning("🛑 Rest Day Recommended")
elif final == "Light":
    st.info("🚴 Light Ride Recommended")
else:
    st.success("🔥 Hard Training Day Recommended")


# ==============================
# 🚀 3-DAY PLAN (RESTORED)
# ==============================
st.subheader("🚀 3-Day Training Plan")

def simulate_3_day_plan(ATL, CTL):
    scenarios = ["Rest", "Light", "Hard"]
    loads = {"Rest": 0, "Light": 30, "Hard": 60}

    best_plan = None
    best_score = -999

    for d1 in scenarios:
        for d2 in scenarios:
            for d3 in scenarios:

                atl = ATL
                ctl = CTL
                tsb_list = []

                for d in [d1, d2, d3]:
                    load = loads[d]

                    atl = atl + (load - atl) / 7
                    ctl = ctl + (load - ctl) / 42
                    tsb = ctl - atl

                    tsb_list.append(tsb)

                min_tsb = min(tsb_list)
                variance = np.var(tsb_list)

                if min_tsb < -15:
                    score = -100
                else:
                    score = min_tsb - variance

                if score > best_score:
                    best_score = score
                    best_plan = {
                        "Day 1": d1,
                        "Day 2": d2,
                        "Day 3": d3,
                        "TSB Trend": [round(x, 1) for x in tsb_list]
                    }

    return best_plan


plan = simulate_3_day_plan(ATL, CTL)

c1, c2, c3 = st.columns(3)
c1.metric("Day 1", plan["Day 1"])
c2.metric("Day 2", plan["Day 2"])
c3.metric("Day 3", plan["Day 3"])

st.write(f"TSB Trend: {plan['TSB Trend']}")
# ==============================
# LIVE MODE (UNCHANGED)
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
