import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import time

st.set_page_config(layout="wide")

# ==============================
# PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

goal = st.sidebar.selectbox(
    "Training Goal",
    ["Build Fitness", "Maintain", "Recover"]
)

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
    base = pd.Timestamp.now()

    for d in range(5):
        times = pd.date_range(end=base - pd.Timedelta(days=d), periods=200, freq="s")

        hr = 120 + 25 * np.sin(np.linspace(0, 6, 200)) + np.random.normal(0, 3, 200)

        df = pd.DataFrame({
            "time": times,
            "hr": hr.clip(100, 175)
        })

        df["delta"] = 1
        rides.append(df)

    return rides


# ==============================
# 🚀 IMPROVED 3-DAY PLANNER
# ==============================
def simulate_3_day_plan(ATL, CTL, TSB, goal):

    # HARD safety gate
    if TSB < -10:
        return {
            "Day 1": "Rest",
            "Day 2": "Light",
            "Day 3": "Light",
            "TSB Trend": ["Recovery Mode"]
        }

    scenarios = ["Rest", "Light", "Hard"]
    loads = {"Rest": 0, "Light": 30, "Hard": 60}

    best_plan = None
    best_score = -9999

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
                avg_tsb = np.mean(tsb_list)
                variance = np.var(tsb_list)

                # ❌ reject dangerous plans
                if min_tsb < -12:
                    continue

                # 📉 trend penalty
                trend_penalty = 0
                for i in range(1, len(tsb_list)):
                    if tsb_list[i] < tsb_list[i-1]:
                        trend_penalty += 2

                load_score = sum([loads[d] for d in [d1, d2, d3]])

                # 🎯 GOAL-AWARE WEIGHTING
                if goal == "Build Fitness":
                    score = (
                        avg_tsb * 0.5 +
                        load_score * 0.4 -
                        trend_penalty * 2 -
                        variance * 0.2
                    )

                elif goal == "Recover":
                    score = (
                        avg_tsb * 0.9 -
                        load_score * 0.2 -
                        trend_penalty * 3 -
                        variance * 0.2
                    )

                else:  # Maintain
                    score = (
                        avg_tsb * 0.7 +
                        load_score * 0.2 -
                        trend_penalty * 2 -
                        variance * 0.2
                    )

                if score > best_score:
                    best_score = score
                    best_plan = {
                        "Day 1": d1,
                        "Day 2": d2,
                        "Day 3": d3,
                        "TSB Trend": [round(x, 1) for x in tsb_list]
                    }

    return best_plan


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
# HISTORY + LOAD
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
    # GRAPH
    # ==============================
    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    # ==============================
    # 🚀 3-DAY PLAN
    # ==============================
    st.subheader("🚀 3-Day Training Plan")

    plan = simulate_3_day_plan(ATL, CTL, TSB, goal)

    c1, c2, c3 = st.columns(3)
    c1.metric("Day 1", plan["Day 1"])
    c2.metric("Day 2", plan["Day 2"])
    c3.metric("Day 3", plan["Day 3"])

    st.write(f"TSB Trend: {plan['TSB Trend']}")
