import numpy as np
import pandas as pd


def compute_fatigue(df: pd.DataFrame, resting_hr: int, max_hr: int) -> pd.DataFrame:
    out = df.copy()

    hr_reserve = max(max_hr - resting_hr, 1)
    out["hr_ratio"] = ((out["hr"] - resting_hr) / hr_reserve).clip(0, 1.25)
    out["cadence"] = out["cadence"].fillna(85)
    out["slope"] = out["slope"].fillna(0).clip(-12, 12)
    out["elevation"] = out["elevation"].ffill().fillna(0)


    elapsed_min = out["delta"].cumsum() / 60.0
    duration_factor = (elapsed_min / max(elapsed_min.max(), 1)).clip(0, 1)
    cadence_factor = (np.abs(out["cadence"] - 88) / 28).clip(0, 1)
    slope_factor = ((out["slope"] + 12) / 24).clip(0, 1)
    elevation_gain = out["elevation"].diff().clip(lower=0).fillna(0)
    elevation_factor = (elevation_gain.rolling(20, min_periods=1).sum() / 40).clip(0, 1)

    fatigue_raw = (
        out["hr_ratio"] * 0.48
        + duration_factor * 0.20
        + cadence_factor * 0.10
        + slope_factor * 0.12
        + elevation_factor * 0.10
    )

    out["fatigue"] = (fatigue_raw * 100).clip(0, 100)
    out["fatigue_zone"] = pd.cut(
        out["fatigue"],
        bins=[-1, 40, 65, 100],
        labels=["Low", "Moderate", "High"],
    )
    return out


def compute_training_load(df: pd.DataFrame) -> float:
    duration_hours = max(df["delta"].sum() / 3600.0, 0.1)
    avg_fatigue = float(df["fatigue"].mean())
    peak_fatigue = float(df["fatigue"].max())
    return round(duration_hours * (avg_fatigue * 0.8 + peak_fatigue * 0.2), 2)
