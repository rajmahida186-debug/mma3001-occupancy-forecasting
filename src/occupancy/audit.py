"""Audit the environmental sensor data and detect frozen sensors.

Each environmental record holds two timestamps:

* ``createdate`` – when the record arrived in Monash's database, and
* the sensor's own measurement time, stored inside the ``jsondata`` text.

For a working sensor both advance together. A *frozen* sensor keeps sending
records (``createdate`` advances) but its own clock and its readings stop
changing, which is physically implausible for indoor air. This module unpacks
the JSON, then measures how often each sensor's clock actually moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import LOCAL_TZ

#: Columns of the raw environmental file used by the audit.
ENV_COLUMNS = ["sensorid", "jsondata", "createdate"]


def parse_co2(jsondata: str) -> tuple[float | None, int | None]:
    """Extract the CO₂ reading and the sensor's own timestamp from one record.

    Args:
        jsondata: The ``jsondata`` text of one record: a JSON list of
            ``{"values": [{"time": ..., "value": ...}], "variable": {...}}``.

    Returns:
        ``(co2_ppm, measured_unix_seconds)``. Either is ``None`` if the record
        has no CO₂ variable or the JSON cannot be parsed.
    """
    try:
        variables = json.loads(jsondata)
    except (TypeError, ValueError):
        return None, None
    for var in variables:
        if var.get("variable", {}).get("name") == "Carbon dioxide":
            first = (var.get("values") or [{}])[0]
            return first.get("value"), first.get("time")
    return None, None


def load_env_co2(raw_csv: str | Path, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Load the environmental file and keep one CO₂ reading per record.

    Args:
        raw_csv: Path to ``5EnvSensor_MayToDec2024_180kRows.csv``.
        tz: Local time zone for the output timestamps.

    Returns:
        DataFrame with ``sensorid``, ``co2_ppm``, ``measured_at`` (sensor clock)
        and ``arrived_at`` (database time), both local and time-zone aware.

    Raises:
        FileNotFoundError: If ``raw_csv`` does not exist.
    """
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw environmental file not found: {raw_csv}")
    raw = pd.read_csv(raw_csv, usecols=ENV_COLUMNS, dtype=str)
    parsed = raw["jsondata"].map(parse_co2)
    out = pd.DataFrame({
        "sensorid": raw["sensorid"],
        "co2_ppm": pd.to_numeric(parsed.str[0], errors="coerce"),
        "measured_unix": pd.to_numeric(parsed.str[1], errors="coerce"),
        "arrived_at": pd.to_datetime(raw["createdate"], utc=True, format="mixed").dt.tz_convert(tz),
    })
    out["measured_at"] = pd.to_datetime(out["measured_unix"], unit="s", utc=True).dt.tz_convert(tz)
    return out


def sensor_health(env: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Summarise, per sensor and period, whether the sensor's clock was moving.

    A period is classed as ``"frozen"`` when records arrived but the sensor
    reported at most one distinct measurement time in it.

    Args:
        env: Output of :func:`load_env_co2`.
        freq: Pandas period frequency for grouping arrivals (``"MS"`` = monthly).

    Returns:
        One row per sensor per period with ``records``, ``distinct_times``,
        ``distinct_co2`` and ``status`` (``"working"`` or ``"frozen"``).
    """
    period = env["arrived_at"].dt.tz_localize(None).dt.to_period(freq[0])
    g = env.assign(period=period).groupby(["sensorid", "period"])
    out = g.agg(records=("co2_ppm", "size"),
                distinct_times=("measured_unix", "nunique"),
                distinct_co2=("co2_ppm", "nunique")).reset_index()
    out["status"] = out["distinct_times"].le(1).map({True: "frozen", False: "working"})
    return out


def valid_periods(env: pd.DataFrame) -> pd.DataFrame:
    """Find the time span over which each sensor produced changing readings.

    Args:
        env: Output of :func:`load_env_co2`.

    Returns:
        One row per sensor with ``records``, ``distinct_times`` and the first and
        last *measurement* times that were not stuck repeats, or NaT if the
        sensor never produced more than two distinct measurement times.
    """
    rows = []
    for sid, g in env.groupby("sensorid"):
        counts = g["measured_unix"].value_counts()
        # A timestamp repeated many times is a stuck reading; keep times seen
        # at most a few times (a working sensor repeats a record occasionally).
        fresh = g[g["measured_unix"].map(counts) <= 5]
        n_distinct = g["measured_unix"].nunique()
        rows.append({
            "sensorid": sid,
            "records": len(g),
            "distinct_times": n_distinct,
            "valid_from": fresh["measured_at"].min() if n_distinct > 2 else pd.NaT,
            "valid_to": fresh["measured_at"].max() if n_distinct > 2 else pd.NaT,
        })
    return pd.DataFrame(rows)
