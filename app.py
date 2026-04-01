# ---------------- SAMPLE CSV (REALISTIC RIDE) ----------------
import numpy as np

np.random.seed(42)

n = 300  # ~5 min ride at 1Hz (you can scale later)

heart_rate = np.clip(
    90 + np.linspace(0, 60, n) + np.random.normal(0, 5, n),
    80, 190
)

cadence = np.clip(
    75 + np.random.normal(0, 10, n),
    50, 110
)

speed = np.clip(
    20 + np.sin(np.linspace(0, 10, n)) * 5 + np.random.normal(0, 1.5, n),
    5, 40
)

slope = np.clip(
    np.sin(np.linspace(0, 6, n)) * 6 + np.random.normal(0, 1, n),
    -5, 12
)

elevation = np.cumsum(np.maximum(slope, 0)) * 2

sample_df = pd.DataFrame({
    "heart_rate": heart_rate.round(0),
    "cadence": cadence.round(0),
    "speed": speed.round(1),
    "slope": slope.round(2),
    "elevation_m": elevation.round(1)
})

sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="📄 Download Realistic Sample Ride",
    data=sample_csv,
    file_name="ridex_sample_realistic.csv"
)
