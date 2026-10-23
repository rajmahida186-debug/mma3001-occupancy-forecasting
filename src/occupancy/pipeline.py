"""Rebuild every processed data file from the raw Monash files.

Run from the repository root after placing the raw files in ``data/raw/``::

    python -m occupancy.pipeline

(after ``pip install -r requirements.txt``, which installs this package).
Takes about a minute (about 40 s for the 1.6 GB occupancy file).
"""

from __future__ import annotations

import time

import pandas as pd

from . import audit, clean
from .config import PROCESSED_DIR, RAW_DIR, RAW_ENV_FILE, RAW_OCCUPANCY_FILE

#: Window and sensors shown in Figure 3.
FIG3_WORKING, FIG3_FROZEN = "6012002000869", "6012002000227"
FIG3_START, FIG3_END = "2024-04-30 12:00", "2024-05-13 00:00"


def main() -> None:
    """Run the cleaning and audit steps and write results to ``data/processed``."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    hourly = clean.build_hourly_occupancy(RAW_DIR / RAW_OCCUPANCY_FILE)
    clean.save_hourly(hourly, PROCESSED_DIR / "occupancy_hourly.csv")
    print(f"Hourly occupancy: {len(hourly):,} rows in {time.perf_counter() - t:.0f} s")

    t = time.perf_counter()
    env = audit.load_env_co2(RAW_DIR / RAW_ENV_FILE)
    audit.sensor_health(env).to_csv(PROCESSED_DIR / "env_sensor_health_monthly.csv", index=False)
    audit.valid_periods(env).to_csv(PROCESSED_DIR / "env_sensor_valid_periods.csv", index=False)
    # Small extract for Figure 3, so the notebook can redraw it without the raw file.
    tz = env["arrived_at"].dt.tz
    window = env["arrived_at"].between(*(pd.Timestamp(x, tz=tz) for x in (FIG3_START, FIG3_END)))
    fig3 = env[window & env["sensorid"].isin([FIG3_WORKING, FIG3_FROZEN])]
    fig3[["sensorid", "arrived_at", "measured_at", "co2_ppm"]].to_csv(
        PROCESSED_DIR / "env_fig3_extract.csv", index=False)
    print(f"Environmental audit: {len(env):,} records in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
