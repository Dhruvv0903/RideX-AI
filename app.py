import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ==============================
# PROFILE
# ==============================
st.sidebar.title("👤 Rider Profile")

age = st.sidebar.number_input("Age", 10, 80, 18)
resting_hr = st.sidebar.number_input("Resting HR", 40, 100, 70)

max_hr = 220 - age

# ==============================
# CACHED PARSERS
# ==============================
@st.cache_data
def parse_tcx(file_bytes):
    root = ET.fromstring(file_bytes)

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
def compute_fatigue_cached(df, resting_hr, max_hr):
    fatigue = 0
    fatigue_list = []

    for _, row in df.iterrows():
        intensity = max(0, (row["hr"] - resting_hr) / (max_hr - resting_hr))

        fatigue += intensity * row["delta"] * 0.03
        fatigue -= 0.015 * row["delta"]

        fatigue = max(0, min(100, fatigue))
        fatigue_list.append(fatigue)

    df = df.copy()
    df["fatigue"] = fatigue_list
    return df


# ==============================
# SAMPLE DATA
# ==============================
@st.cache_data
def generate_sample_data():
    times = pd.date_range(end=pd.Timestamp.now(), periods=300, freq="S")
    hr = 120 + 20 * pd.Series(range(300)).apply(lambda x: (x % 50) / 2)

    df = pd.DataFrame({"time": times, "hr": hr})
    df["delta"] = 1
    return df


# ==============================
# MAIN
# ==============================
st.title("🚴 RideX AI — Performance Engine")

mode = st.radio("Mode", ["Upload Files", "Sample Data"])

history = []
all_hr = []

# ==============================
# LOAD DATA
# ==============================
if mode == "Upload Files":
    files = st.file_uploader("Upload TCX/CSV", type=["tcx", "csv"], accept_multiple_files=True)

    if files:
        for f in files:
            bytes_data = f.read()

            if f.name.endswith(".tcx"):
                df = parse_tcx(bytes_data)
            else:
                df = parse_csv(bytes_data)

            if not df.empty:
                df = compute_fatigue_cached(df, resting_hr, max_hr)

                all_hr.extend(df["hr"].tolist())

                history.append({
                    "date": df["time"].iloc[-1],
                    "load": df["fatigue"].mean(),
                    "avg_hr": df["hr"].mean()
                })

else:
    df = generate_sample_data()
    df = compute_fatigue_cached(df, resting_hr, max_hr)

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
    zone_low = hr_series.mean() - 0.5 * hr_series.std()
    zone_high = hr_series.mean() + 0.5 * hr_series.std()
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
    ATL = Acute Training Load (short-term fatigue)  
    CTL = Chronic Training Load (long-term fitness)  
    TSB = Training Stress Balance (readiness)
    """)

    # Chart
    fig, ax = plt.subplots()
    ax.plot(history_df["date"], history_df["ATL"], label="ATL")
    ax.plot(history_df["date"], history_df["CTL"], label="CTL")
    ax.plot(history_df["date"], history_df["TSB"], label="TSB")
    ax.legend()
    st.pyplot(fig)

    # ==============================
    # DATE + READINESS
    # ==============================
    last_date = pd.to_datetime(history_df["date"].iloc[-1]).tz_localize(None)
    today = pd.Timestamp.now().tz_localize(None)

    days_since = (today - last_date).days
    st.write(f"📅 Days since last ride: {days_since}")

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
    # TOMORROW
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

    st.dataframe(pd.DataFrame(scenarios))

    if TSB < -10:
        rec = "Rest tomorrow"
    elif TSB < 0:
        rec = "Light ride recommended"
    else:
        rec = "Push hard tomorrow"

    st.subheader("🧠 Recommendation")
    st.info(rec)


# ==============================
# LIVE MODE (FAST)
# ==============================
st.subheader("⚡ Live Ride Mode — Smart Pacing")

live_mode = st.radio("Live Source", ["Upload", "Sample"])

if live_mode == "Upload":
    live_file = st.file_uploader("Upload TCX/CSV", type=["tcx", "csv"], key="live")

    if live_file:
        bytes_data = live_file.read()
        df_live = parse_tcx(bytes_data) if live_file.name.endswith(".tcx") else parse_csv(bytes_data)
else:
    df_live = generate_sample_data()

if "df_live" in locals() and not df_live.empty:
    df_live = compute_fatigue_cached(df_live, resting_hr, max_hr)

    if st.button("▶ Start Simulation"):

        placeholder = st.empty()
        smoothed_hr = None

        for i in range(0, len(df_live), 8):  # faster skip

            row = df_live.iloc[i]
            hr = row["hr"]
            fatigue = row["fatigue"]

            smoothed_hr = hr if smoothed_hr is None else 0.85 * smoothed_hr + 0.15 * hr

            # HR logic
            if smoothed_hr < zone_low:
                decision = "🔥 Push more"
            elif zone_low <= smoothed_hr <= zone_high:
                decision = "✅ Perfect pacing"
            elif smoothed_hr <= zone_high + 10:
                decision = "⚖️ Strong effort"
            else:
                decision = "🚨 Too intense"

            # fatigue limiter
            if fatigue > 85:
                decision = "🛑 STOP — exhaustion imminent"
            elif fatigue > 70:
                decision = "⚠️ Reduce effort"

            with placeholder.container():
                st.metric("Heart Rate", f"{int(smoothed_hr)} bpm")
                st.metric("Fatigue", f"{int(fatigue)} / 100")
                st.info(f"Adaptive Zone: {int(zone_low)} - {int(zone_high)} bpm")
                st.success(decision)

            time.sleep(0.04)  # smooth + faster
