import json
import time
import xml.etree.ElementTree as ET
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load_from_device
from fatigue_model import compute_fatigue, compute_training_load
from strava_api import (
    exchange_code_for_token,
    get_activities,
    get_activity_streams,
    get_auth_url,
    refresh_access_token,
)

st.set_page_config(page_title="RideX AI", page_icon="🚴", layout="wide")


@st.cache_data
def parse_tcx(file_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(file_bytes)
    ns = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    rows = []
    for tp in root.findall(".//ns:Trackpoint", ns):
        time_node = tp.find("ns:Time", ns)
        hr_node = tp.find(".//ns:HeartRateBpm/ns:Value", ns)
        cadence_node = tp.find("ns:Cadence", ns)
        altitude_node = tp.find("ns:AltitudeMeters", ns)

        if time_node is None or hr_node is None:
            continue

        rows.append(
            {
                "time": pd.to_datetime(time_node.text),
                "hr": float(hr_node.text),
                "cadence": float(cadence_node.text) if cadence_node is not None else np.nan,
                "elevation": float(altitude_node.text) if altitude_node is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    return finalize_stream_df(df)


@st.cache_data
def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))
    column_map = {c.lower().strip(): c for c in df.columns}

    if "time" not in column_map:
        raise ValueError("CSV must include a 'time' column.")
    if "hr" not in column_map and "heart_rate" not in column_map:
        raise ValueError("CSV must include 'hr' or 'heart_rate'.")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[column_map["time"]])
    hr_col = column_map["hr"] if "hr" in column_map else column_map["heart_rate"]
    out["hr"] = pd.to_numeric(df[hr_col], errors="coerce")

    if "cadence" in column_map:
        out["cadence"] = pd.to_numeric(df[column_map["cadence"]], errors="coerce")
    if "slope" in column_map:
        out["slope"] = pd.to_numeric(df[column_map["slope"]], errors="coerce")
    if "elevation" in column_map:
        out["elevation"] = pd.to_numeric(df[column_map["elevation"]], errors="coerce")

    return finalize_stream_df(out)


def finalize_stream_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "hr"]).copy()

    df["delta"] = df["time"].diff().dt.total_seconds().fillna(1).clip(lower=1, upper=30)

    if "cadence" not in df.columns:
        df["cadence"] = np.nan
    if "elevation" not in df.columns:
        df["elevation"] = np.nan
    if "slope" not in df.columns:
        elev_delta = df["elevation"].diff().fillna(0)
        df["slope"] = elev_delta.clip(-12, 12)

    return df


def build_history_row(df: pd.DataFrame) -> dict:
    load_score = compute_training_load(df)
    return {
        "date": pd.to_datetime(df["time"].iloc[-1], utc=True).tz_convert(None),
        "load": float(load_score),
        "avg_hr": float(df["hr"].mean()),
        "peak_fatigue": float(df["fatigue"].max()),
        "avg_fatigue": float(df["fatigue"].mean()),
        "duration_min": float(df["delta"].sum() / 60.0),
    }


def compute_fitness_metrics(history_df: pd.DataFrame) -> pd.DataFrame:
    history_df = history_df.sort_values("date").copy()
    history_df["ATL"] = history_df["load"].ewm(span=7, adjust=False).mean()
    history_df["CTL"] = history_df["load"].ewm(span=42, adjust=False).mean()
    history_df["TSB"] = history_df["CTL"] - history_df["ATL"]
    history_df["load_7d"] = history_df["load"].rolling(7, min_periods=1).sum()
    history_df["load_3d"] = history_df["load"].rolling(3, min_periods=1).sum()
    return history_df


def compute_data_confidence(history_df: pd.DataFrame) -> str:
    rides = len(history_df)
    if rides >= 8:
        return "High"
    if rides >= 4:
        return "Medium"
    return "Low"


def get_today_plan(
    tsb: float,
    atl: float,
    ctl: float,
    gap_days: int,
    training_goal: str,
    confidence: str,
) -> dict:
    recommendation = "Light Ride"
    explanation = (
        "You are in a usable training state today, but not so fresh that it makes sense to force a big effort."
    )
    instruction = "Ride 35-50 min in Zone 2. Keep the pace controlled and finish feeling like you could do more."
    fallback = "If the warm-up feels heavier than expected, cut it to 25-30 min and keep it fully easy."

    if gap_days >= 7:
        recommendation = "Return Ride"
        explanation = (
            "You have been away from the bike for a bit, so the priority today is to re-open the legs without shocking the system."
        )
        instruction = "Ride 30-40 min in Zone 1-2. Smooth cadence, no hard surges, no chasing numbers."
        fallback = "If you still feel stale after 10-15 min, turn it into a short recovery spin or take the day off."
    elif tsb <= -15:
        recommendation = "Rest Day"
        explanation = (
            "You are carrying enough recent fatigue that more intensity is unlikely to help. Today should absorb training, not add stress."
        )
        instruction = "Take a full rest day. If you need movement, do 20-30 min very easy in Zone 1 only."
        fallback = "If your legs feel dead or heart rate is unusually high at easy effort, skip the spin and recover fully."
    elif tsb <= -7:
        recommendation = "Recovery Ride"
        explanation = (
            "You are not buried, but the fatigue is real. This is a better day for circulation and recovery than for quality work."
        )
        instruction = "Ride 30-45 min in Zone 1-2. Keep cadence relaxed and avoid climbs or hard pulls."
        fallback = "If energy feels poor in the first 10 min, stop early and bank the recovery."
    elif tsb >= 10 and training_goal == "Build Fitness":
        recommendation = "Hard Intervals"
        explanation = (
            "You look fresh enough to handle quality today. This is a good window to push and actually absorb the work."
        )
        instruction = "Ride 50-65 min total with 4 x 5 min in Zone 4, separated by easy spinning."
        fallback = "If you do not feel sharp after the warm-up, switch to a steady 45-min endurance ride instead."
    elif tsb >= 8 and training_goal == "Maintain":
        recommendation = "Tempo Ride"
        explanation = (
            "You have enough freshness for a controlled quality day, but the goal is to keep fitness ticking rather than dig deep."
        )
        instruction = "Ride 45-60 min with 2 x 10 min in upper Zone 2 to low Zone 3."
        fallback = "If the legs feel flat, keep the whole ride in Zone 2 and skip the tempo blocks."
    elif training_goal == "Recover":
        recommendation = "Recovery Spin"
        explanation = (
            "Your current goal is recovery, so even on a decent day the right move is to stay gentle and let the body settle."
        )
        instruction = "Ride 25-40 min in Zone 1-2 only. Keep it easy enough to chat throughout."
        fallback = "If recovery still feels incomplete, replace the ride with full rest."
    elif training_goal == "Build Fitness":
        recommendation = "Endurance Ride"
        explanation = (
            "You are in a solid middle ground today: stable enough to train, but not so fresh that you need to force intensity."
        )
        instruction = "Ride 45-60 min in Zone 2. Smooth pressure on the pedals and no random surges."
        fallback = "If the ride feels unusually good, add 5-10 min near high Zone 2, but keep it controlled."
    elif training_goal == "Maintain":
        recommendation = "Steady Ride"
        explanation = (
            "You are in a balanced state. Today is a good day to keep consistency without trying to prove anything."
        )
        instruction = "Ride 40-55 min in Zone 2 with one or two short cadence pickups."
        fallback = "If time is tight, do 25-30 min easy and keep the routine alive."

    if confidence == "Low":
        explanation += " Data is limited, so this recommendation stays a little conservative."
        fallback = "When in doubt, choose the easier option today and let a few more rides sharpen the model."

    return {
        "recommendation": recommendation,
        "explanation": explanation,
        "instruction": instruction,
        "fallback": fallback,
        "confidence": confidence,
    }


def render_today_plan(plan: dict) -> None:
    st.subheader("Today's Plan")
    a, b = st.columns([2, 1])
    a.info(plan["recommendation"])
    b.metric("Confidence", plan["confidence"])

    st.markdown("**Explanation**")
    st.write(plan["explanation"])

    st.markdown("**Structured Instruction**")
    st.write(plan["instruction"])

    st.markdown("**Optional Fallback**")
    st.write(plan["fallback"])


def render_scenario_tests() -> None:
    scenarios = [
        {
            "label": "Overtrained",
            "tsb": -18.0,
            "atl": 92.0,
            "ctl": 68.0,
            "gap_days": 1,
            "goal": "Build Fitness",
            "confidence": "High",
        },
        {
            "label": "Balanced",
            "tsb": 1.5,
            "atl": 52.0,
            "ctl": 54.0,
            "gap_days": 1,
            "goal": "Maintain",
            "confidence": "Medium",
        },
        {
            "label": "Fresh",
            "tsb": 14.0,
            "atl": 38.0,
            "ctl": 57.0,
            "gap_days": 2,
            "goal": "Build Fitness",
            "confidence": "High",
        },
    ]

    for scenario in scenarios:
        with st.expander(scenario["label"], expanded=False):
            plan = get_today_plan(
                tsb=scenario["tsb"],
                atl=scenario["atl"],
                ctl=scenario["ctl"],
                gap_days=scenario["gap_days"],
                training_goal=scenario["goal"],
                confidence=scenario["confidence"],
            )
            st.write(f"**Recommendation:** {plan['recommendation']}")
            st.write(f"**Explanation:** {plan['explanation']}")
            st.write(f"**Instruction:** {plan['instruction']}")
            st.write(f"**Fallback:** {plan['fallback']}")
            st.write(f"**Confidence:** {plan['confidence']}")


def render_coach_logic_state(is_connected: bool) -> None:
    st.subheader("Coach Logic Check")
    if not is_connected:
        st.info("Connect Strava first to unlock coach logic checks based on your ride history.")
        return
    render_scenario_tests()


st.title("RideX AI - Performance Engine")
st.caption(
    "A lightweight system for turning cycling data into fatigue signals, readiness insight, and a daily coaching decision."
)

st.sidebar.title("Rider Profile")
age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)
training_goal = st.sidebar.selectbox(
    "Training Goal",
    ["Build Fitness", "Maintain", "Recover"],
)
max_hr = 220 - age

data_mode = st.sidebar.radio(
    "Data Source",
    ["Upload Files", "Strava"],
)

# ==============================
# STRAVA AUTH
# ==============================
st.sidebar.subheader("Connect Strava")

if "access_token" in st.session_state:
    st.sidebar.success("Strava connected")

if st.sidebar.button("Connect Strava"):
    auth_url = get_auth_url()
    if auth_url:
        st.sidebar.markdown(f"[Authorize Strava]({auth_url})")
    else:
        st.sidebar.error(
            "Strava is not configured yet. Add real STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, and STRAVA_REDIRECT_URI values."
        )

query_params = st.query_params
code = query_params.get("code", None)
if isinstance(code, list):
    code = code[0]

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
    except Exception:
        pass

if code and "access_token" not in st.session_state:
    token_data = exchange_code_for_token(code)

    if "access_token" in token_data:
        st.session_state.update(token_data)
        st.query_params.clear()
        st.query_params["token"] = json.dumps(token_data)
        st.success("Strava connected")
        st.rerun()
    else:
        st.error("Strava connection failed")

if "access_token" in st.session_state:
    try:
        if time.time() > st.session_state["expires_at"]:
            new_tokens = refresh_access_token(st.session_state["refresh_token"])
            if "access_token" in new_tokens:
                st.session_state.update(new_tokens)
    except Exception:
        st.sidebar.warning("Could not refresh Strava token. Try reconnecting.")

# ==============================
# DATA PIPELINE
# ==============================
history = []
latest_stream = None
all_hr = []

if data_mode == "Upload Files":
    st.caption("Upload your own TCX or CSV ride files to analyze how you performed and how much fatigue the ride created.")
    uploaded_files = st.file_uploader(
        "Upload TCX or CSV files",
        type=["tcx", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            b = f.read()
            df = parse_tcx(b) if f.name.lower().endswith(".tcx") else parse_csv(b)

            if not df.empty:
                df = compute_fatigue(df, resting_hr, max_hr)
                all_hr.extend(df["hr"].tolist())
                history.append(build_history_row(df))
                latest_stream = df

elif data_mode == "Strava":
    if "access_token" in st.session_state:
        activities = get_activities(st.session_state["access_token"])

        for act in activities[:5]:
            if act.get("type") != "Ride":
                continue

            streams = get_activity_streams(act["id"], st.session_state["access_token"])

            if not streams or "heartrate" not in streams or "time" not in streams:
                continue

            hr_data = streams["heartrate"]["data"]
            time_data = streams["time"]["data"]

            df_stream = pd.DataFrame(
                {
                    "time": pd.to_datetime(act["start_date"], utc=True) + pd.to_timedelta(time_data, unit="s"),
                    "hr": hr_data,
                }
            )

            df_stream["delta"] = df_stream["time"].diff().dt.total_seconds().fillna(1)
            df_stream = compute_fatigue(df_stream, resting_hr, max_hr)

            all_hr.extend(df_stream["hr"].tolist())
            history.append(
                {
                    "date": df_stream["time"].iloc[-1].tz_convert(None),
                    "load": float(df_stream["fatigue"].sum() / 10),
                    "avg_hr": float(df_stream["hr"].mean()),
                    "peak_fatigue": float(df_stream["fatigue"].max()),
                    "avg_fatigue": float(df_stream["fatigue"].mean()),
                    "duration_min": float(df_stream["delta"].sum() / 60.0),
                }
            )
            latest_stream = df_stream
    else:
        st.info("Connect Strava first to load your ride history.")

if not history:
    if data_mode == "Strava":
        st.warning("No Strava ride history is loaded yet.")
        render_coach_logic_state(is_connected=False)
    else:
        st.warning("No uploaded ride history yet. Add your files to see how you did and what the coach recommends next.")
    st.stop()

# ==============================
# METRICS
# ==============================
history_df = pd.DataFrame(history).sort_values("date")
history_df = compute_fitness_metrics(history_df)

atl = history_df["ATL"].iloc[-1]
ctl = history_df["CTL"].iloc[-1]
tsb = history_df["TSB"].iloc[-1]
confidence = compute_data_confidence(history_df)

c1, c2, c3 = st.columns(3)
c1.metric("ATL", f"{atl:.1f}")
c2.metric("CTL", f"{ctl:.1f}")
c3.metric("TSB", f"{tsb:.1f}")

# ==============================
# GRAPH
# ==============================
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(history_df["date"], history_df["ATL"], label="ATL", linewidth=2)
ax.plot(history_df["date"], history_df["CTL"], label="CTL", linewidth=2)
ax.plot(history_df["date"], history_df["TSB"], label="TSB", linewidth=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=30)
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)

# ==============================
# READINESS
# ==============================
last_date = pd.to_datetime(history_df["date"].iloc[-1]).tz_localize(None)
gap = (pd.Timestamp.now().tz_localize(None) - last_date).days

if gap > 7:
    readiness = "Fresh - recovered"
elif tsb > 10:
    readiness = "Fresh - push hard"
elif tsb < -10:
    readiness = "Fatigued - recover"
else:
    readiness = "Moderate - train smart"

st.subheader("Readiness")
st.success(readiness)

# ==============================
# COACH LAYER
# ==============================
plan = get_today_plan(tsb, atl, ctl, gap, training_goal, confidence)
render_today_plan(plan)

# ==============================
# RIDE SUMMARY
# ==============================
st.subheader("Ride Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Rides Loaded", len(history_df))
s2.metric("Last Avg HR", f"{history_df['avg_hr'].iloc[-1]:.0f} bpm")
s3.metric("Last Avg Fatigue", f"{history_df['avg_fatigue'].iloc[-1]:.1f}")
s4.metric("Days Since Ride", gap)

# ==============================
# LIVE MODE
# ==============================
st.subheader("Live Ride Mode")
df_live = load_from_device()

if df_live is not None and not df_live.empty:
    df_live = compute_fatigue(finalize_stream_df(df_live), resting_hr, max_hr)

    if st.button("Start Simulation"):
        placeholder = st.empty()

        for i in range(0, len(df_live), 6):
            row = df_live.iloc[i]

            with placeholder.container():
                m1, m2 = st.columns(2)
                m1.metric("Heart Rate", int(row["hr"]))
                m2.metric("Fatigue", int(row["fatigue"]))

            time.sleep(0.08)
else:
    st.caption("No live device feed is available right now.")

render_coach_logic_state(is_connected="access_token" in st.session_state)
