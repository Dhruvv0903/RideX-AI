import numpy as np
import pandas as pd


def load_from_device() -> pd.DataFrame | None:
    rng = np.random.default_rng(21)
    points = 240
    hr = np.clip(122 + np.sin(np.linspace(0, 5, points)) * 18 + rng.normal(0, 4, points), 96, 182)
    cadence = np.clip(86 + rng.normal(0, 4, points), 70, 102)
    slope = np.clip(np.sin(np.linspace(0, 4, points)) * 4 + rng.normal(0, 0.8, points), -6, 8)
    elevation = np.cumsum(np.maximum(slope, -1) * 0.35)
    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=points, freq="5s")

    return pd.DataFrame(
        {
            "time": timestamps,
            "hr": hr,
            "cadence": cadence,
            "slope": slope,
            "elevation": elevation,
        }
    )
