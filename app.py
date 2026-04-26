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
