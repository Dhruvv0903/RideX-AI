import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time
import json
import matplotlib.dates as mdates

from strava_api import get_activity_streams
from fatigue_model import compute_fatigue
from strava_api import (
    get_auth_url,
    exchange_code_for_token,
    get_activities,
    refresh_access_token
)
from data_loader import load_from_device
import pickle
import os

@st.cache_data
def load_tokens():
    if os.path.exists("tokens.pkl"):
        with open("tokens.pkl", "rb") as f:
            return pickle.load(f)
    return {}

@st.cache_data
def save_tokens(tokens):
    with open("tokens.pkl", "wb") as f:
        pickle.dump(tokens, f)

# In auth, after setting st.session_state:
save_tokens({"access_token": st.session_state["access_token"], ...})
# On load: st.session_state.update(load_tokens())
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

training_goal = st.sidebar.selectbox(
    "Training Goal",
    ["Build Fitness", "Maintain", "Recover"]
)

max_hr = 220 - age

# ==============================
# DATA SOURCE
# ==============================
data_mode = st.sidebar.radio(
    "Data Source",
    ["Upload Files", "Strava", "Sample Data"]
)

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

# ==============================
# STRAVA AUTH (Patched Original)
# ==============================
st.sidebar.subheader("🔗 Connect")

if st.sidebar.button("Connect Strava"):
    st.markdown(f"[Authorize Strava]({get_auth_url()})")

# Load token from URL params (fixed list/JSON handling)
params = st.query_params
if "token" in params and "access_token" not in st.session_state:
    try:
        token_str = params["token"]
        if isinstance(token_str, list):
            token_str = token_str[0]
        token_data = json.loads(token_str)
        st.session_state["access_token"] = token_data["access_token"]
        st.session_state["refresh_token"] = token_data["refresh_token"]
        st.session_state["expires_at"] = token_data["expires_at"]
        st.rerun()  # Refresh to clear params if desired
    except Exception as e:
        st.sidebar.error(f"Token load failed: {e}")

# Exchange code (use query_params directly, no deprecated set)
code = st.query_params.get("code", None)
if code and "access_token" not in st.session_state:
    try:
        token_data = exchange_code_for_token(code)
        if "access_token" in token_data:
            st.session_state.update(token_data)
            # Clear code param, set token briefly then clear for security
            new_params = st.query_params.to_dict()
            new_params.pop("code", None)
            new_params["token"] = json.dumps(token_data)  # Temp for persistence
            st.query_params.clear()
            st.query_params.update(new_params)
            st.success("✅ Strava Connected")
            st.rerun()
        else:
            st.error("❌ Strava connection failed")
    except Exception as e:
        st.error(f"Exchange failed: {e}")

# Refresh token (unchanged)
if "access_token" in st.session_state and time.time() > st.session_state["expires_at"]:
    try:
        new_tokens = refresh_access_token(st.session_state["refresh_token"])
        st.session_state.update(new_tokens)
    except:
        pass

# ==============================
# TOKEN REFRESH
# ==============================
if "access_token" in st.session_state:
    if time.time() > st.session_state["expires_at"]:
        new_tokens = refresh_access_token(st.session_state["refresh_token"])
        st.session_state.update(new_tokens)

# ==============================
# DATA PIPELINE
# ==============================
history = []
all_hr = []

# UPLOAD
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
                    "load": df["fatigue"].mean() * 2.5,
                    "avg_hr": df["hr"].mean()
                })
    else:
        st.warning("Upload files to proceed")

# STRAVA
elif data_mode == "Strava":
    if "access_token" in st.session_state:

        activities = get_activities(st.session_state["access_token"])

        for act in activities[:5]:
            if act.get("type") != "Ride":
                continue

            streams = get_activity_streams(
                act["id"],
                st.session_state["access_token"]
            )

            if not streams or "heartrate" not in streams:
                continue

            hr_data = streams["heartrate"]["data"]
            time_data = streams["time"]["data"]

            if len(hr_data) == 0:
                continue

            df_stream = pd.DataFrame({
                "time": pd.to_datetime(act["start_date"]) + pd.to_timedelta(time_data, unit="s"),
                "hr": hr_data
            })

            df_stream["delta"] = df_stream["time"].diff().dt.total_seconds().fillna(1)

            df_stream = compute_fatigue(df_stream, resting_hr, max_hr)

            all_hr.extend(df_stream["hr"])

            history.append({
                "date": df_stream["time"].iloc[-1],
                "load": df_stream["fatigue"].sum() / 10,
                "avg_hr": df_stream["hr"].mean()
            })

    else:
        st.warning("Connect Strava to load data")

# SAMPLE
elif data_mode == "Sample Data":
    st.info("Using sample data")

    for i in range(5):
        hr = 120 + np.random.randint(-10, 20)
        history.append({
            "date": pd.Timestamp.now() - pd.Timedelta(days=i),
            "load": hr * 1.0,
            "avg_hr": hr
        })
        all_hr.append(hr)

# ==============================
# STOP IF EMPTY
# ==============================
if not history:
    st.stop()

# ==============================
# METRICS
# ==============================
history_df = pd.DataFrame(history).sort_values("date")

history_df["ATL"] = history_df["load"].ewm(span=7).mean()
history_df["CTL"] = history_df["load"].ewm(span=60).mean()
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

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
ax.legend()

st.pyplot(fig)

# ==============================
# READINESS
# ==============================
last_date = pd.to_datetime(history_df["date"].iloc[-1]).tz_localize(None)
gap = (pd.Timestamp.now().tz_localize(None) - last_date).days

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
# LIVE MODE (UNCHANGED)
# ==============================
st.subheader("⚡ Live Ride Mode")

df_live = load_from_device()

if df_live is not None and not df_live.empty:
    df_live = compute_fatigue(df_live, resting_hr, max_hr)

    if st.button("▶ Start Simulation"):
        placeholder = st.empty()

        for i in range(0, len(df_live), 6):
            row = df_live.iloc[i]

            with placeholder.container():
                st.metric("Heart Rate", int(row["hr"]))
                st.metric("Fatigue", int(row["fatigue"]))

            time.sleep(0.08)
