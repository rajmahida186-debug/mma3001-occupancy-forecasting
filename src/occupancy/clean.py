"""Clean the raw occupancy events and aggregate them to an hourly table.

The raw occupancy file is an *event log*: a new row is written whenever a
sensor's reading changes (roughly every 20 s while a space is in use), giving
about 9.1 million rows. Models need a regular time step, so this module turns
the events into one row per space per hour.

Main steps (see :func:`build_hourly_occupancy`):

1. Read the 1.6 GB CSV in chunks, so it fits in memory on a laptop.
2. Remove exact duplicate events.
3. Assign each event to its local clock hour, safely across daylight saving.
4. Average the headcounts within each hour.

Hours with no events are left out (treated as *missing*), not filled with zero:
an offline sensor cannot be distinguished from an empty room.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EVENT_HEADCOUNT_THRESHOLD, LOCAL_TZ, SPACE_NAMES

#: Columns of the raw occupancy file that the pipeline needs.
OCCUPANCY_COLUMNS = ["floorspaceid", "occupancystatus", "headcount", "collecteddate"]

#: Columns that together identify a unique event (the ``id`` column is
#: excluded because duplicated events were given different ids).
DUPLICATE_KEY = ["floorspaceid", "occupancystatus", "headcount", "collecteddate"]


def to_local_hour(timestamps: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    """Round timestamps down to the start of their local clock hour.

    The raw timestamps carry a UTC offset (e.g. ``"2024-05-14 13:02:10+10"``).
    Rounding *after* converting to local time fails on daylight-saving days,
    because some local hours are repeated (e.g. 2:00 am on 6 Apr 2025) and
    pandas cannot tell which one is meant. Rounding in UTC first and then
    converting avoids the ambiguity, since Melbourne's offsets are whole hours.

    Args:
        timestamps: Timestamp strings or datetimes, each with a UTC offset.
        tz: IANA time-zone name to convert to.

    Returns:
        Time-zone-aware timestamps (local time) at the start of each hour.

    Raises:
        ValueError: If any timestamp cannot be parsed.
    """
    try:
        utc = parse_utc_fast(timestamps)
    except (ValueError, TypeError, AttributeError):
        utc = parse_utc_general(timestamps)   # any other format: slower but flexible
    return utc.dt.floor("h").dt.tz_convert(tz)


def parse_utc_general(timestamps: pd.Series) -> pd.Series:
    """Parse timestamps of any format to UTC (the original, general method).

    pandas has to work out each value's format and offset separately, which is
    slow on millions of rows: profiling showed it took about half of the total
    cleaning time.

    Raises:
        ValueError: If any timestamp cannot be parsed.
    """
    return pd.to_datetime(timestamps, utc=True, format="mixed")


def parse_utc_fast(timestamps: pd.Series) -> pd.Series:
    """Parse ``"YYYY-MM-DD HH:MM:SS+HH"`` strings to UTC quickly.

    The raw file always uses this layout. The first 19 characters (local date
    and time) are parsed with a fixed format, which pandas does in fast
    compiled code; the last characters are the whole-hour UTC offset (``+10`` or
    ``+11``), which is subtracted. About 8x faster than
    :func:`parse_utc_general` with identical results.

    Raises:
        ValueError: If any value does not have exactly this layout, so the
            caller can fall back to :func:`parse_utc_general`.
    """
    local = pd.to_datetime(timestamps.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    offset_hours = timestamps.str.slice(19).astype("int64")   # fails on "+10:30", ".5+10", etc.
    return (local - pd.to_timedelta(offset_hours, unit="h")).dt.tz_localize("UTC")


def drop_duplicate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Remove events that repeat an earlier event exactly.

    Args:
        events: Raw occupancy events containing the :data:`DUPLICATE_KEY` columns.

    Returns:
        The events with exact duplicates removed (first occurrence kept).
    """
    return events.drop_duplicates(subset=DUPLICATE_KEY, keep="first")


def aggregate_hourly(events: pd.DataFrame) -> pd.DataFrame:
    """Summarise events per space per hour as partial sums.

    Partial sums (rather than means) are returned so that results from
    separate file chunks can be added together exactly afterwards.

    Args:
        events: Occupancy events with ``floorspaceid``, ``headcount`` and
            ``collecteddate`` columns.

    Returns:
        DataFrame indexed by (``floorspaceid``, ``hour``) with columns
        ``sum_headcount``, ``n_readings`` and ``max_headcount``.
    """
    df = events.assign(hour=to_local_hour(events["collecteddate"]))
    grouped = df.groupby(["floorspaceid", "hour"])["headcount"]
    return pd.DataFrame(
        {
            "sum_headcount": grouped.sum(),
            "n_readings": grouped.count(),
            "max_headcount": grouped.max(),
        }
    )


def combine_partial_hourly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge partial hourly sums from several chunks into final hourly values.

    An hour can be split across two chunks of the file, so sums and counts
    are added and maxima are re-maximised before the mean is calculated.

    Args:
        parts: Outputs of :func:`aggregate_hourly` for each chunk.

    Returns:
        One row per space per hour with ``mean_headcount`` (people),
        ``max_headcount`` (people) and ``n_readings``.
    """
    combined = pd.concat(parts).groupby(level=[0, 1]).agg(
        {"sum_headcount": "sum", "n_readings": "sum", "max_headcount": "max"}
    )
    combined["mean_headcount"] = combined["sum_headcount"] / combined["n_readings"]
    return combined.drop(columns="sum_headcount").reset_index()


def add_calendar_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add readable date, hour, weekday and space-name columns.

    Args:
        hourly: Output of :func:`combine_partial_hourly` with a local ``hour``
            timestamp and a ``floorspaceid`` column.

    Returns:
        A copy with ``space``, ``date``, ``hour_of_day``, ``weekday`` and
        ``day_type`` (``"Weekday"``/``"Weekend"``) columns added.
    """
    out = hourly.copy()
    out["space"] = out["floorspaceid"].map(SPACE_NAMES).fillna("Unknown")
    out["date"] = out["hour"].dt.strftime("%Y-%m-%d")
    out["hour_of_day"] = out["hour"].dt.hour
    out["weekday"] = out["hour"].dt.day_name()
    out["day_type"] = (out["hour"].dt.dayofweek < 5).map({True: "Weekday", False: "Weekend"})
    return out


def flag_events(hourly: pd.DataFrame, threshold: int = EVENT_HEADCOUNT_THRESHOLD) -> pd.DataFrame:
    """Flag hours with unusually high headcount as special events.

    Flagged hours are kept (they are genuine, e.g. Monash Open Day) so that
    later stages can decide explicitly whether to include them.

    Args:
        hourly: Hourly table with a ``max_headcount`` column.
        threshold: Peak headcount at or above which an hour is flagged.

    Returns:
        A copy with a boolean ``is_event`` column.
    """
    out = hourly.copy()
    out["is_event"] = out["max_headcount"] >= threshold
    return out


def build_hourly_occupancy(raw_csv: str | Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw occupancy file.

    Args:
        raw_csv: Path to ``5occupancySensor_MayToDec2024_9MRows.csv``.
        chunksize: Number of rows read at a time. 500,000 keeps peak memory
            near 0.5 GB with no loss of speed (2 million rows used 1.4 GB, the
            whole file at once 2.9 GB; see notebook 06).

    Returns:
        Hourly table with one row per space per hour that had at least one
        reading, sorted by space then time.

    Raises:
        FileNotFoundError: If ``raw_csv`` does not exist.
    """
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw occupancy file not found: {raw_csv}")

    parts = []
    seen_keys = None  # duplicates can span chunk boundaries, so keep the last chunk's keys
    for chunk in pd.read_csv(raw_csv, usecols=OCCUPANCY_COLUMNS, chunksize=chunksize,
                             dtype={"headcount": "float32"}):
        chunk = drop_duplicate_events(chunk)
        if seen_keys is not None:
            # Drop rows that duplicate an event already seen at the end of the previous chunk.
            merged = chunk.merge(seen_keys, on=DUPLICATE_KEY, how="left", indicator=True)
            chunk = chunk[(merged["_merge"] == "left_only").to_numpy()]
        seen_keys = chunk[DUPLICATE_KEY].tail(10_000)
        parts.append(aggregate_hourly(chunk))

    hourly = add_calendar_columns(combine_partial_hourly(parts))
    hourly = flag_events(hourly)
    order = {name: i for i, name in enumerate(SPACE_NAMES.values())}
    hourly = hourly.sort_values(["space", "hour"], key=lambda s: s.map(order) if s.name == "space" else s)
    return hourly.reset_index(drop=True)


def daily_coverage(hourly: pd.DataFrame) -> pd.DataFrame:
    """Count, per day and space, how many hours have at least one reading.

    Every calendar day between the first and last reading in the whole table
    is included, so days with no data appear as 0 (this is what Figure 2 shows).

    Args:
        hourly: Hourly table with ``date`` and ``space`` columns.

    Returns:
        DataFrame indexed by date (one row per day) with one column per space.
    """
    counts = hourly.groupby(["date", "space"]).size().unstack(fill_value=0)
    all_days = pd.date_range(counts.index.min(), counts.index.max(), freq="D").strftime("%Y-%m-%d")
    return counts.reindex(all_days, fill_value=0)


def save_hourly(hourly: pd.DataFrame, path: str | Path) -> None:
    """Save the hourly table as CSV with readable local timestamps.

    Args:
        hourly: Output of :func:`build_hourly_occupancy`.
        path: Destination CSV file.
    """
    out = hourly.copy()
    # Keep the UTC offset (e.g. +1000 / +1100) so the repeated hour when
    # daylight saving ends can be told apart when the file is read back.
    out["hour_start_local"] = out["hour"].dt.strftime("%Y-%m-%d %H:%M%z")
    cols = ["hour_start_local", "date", "hour_of_day", "weekday", "day_type", "space",
            "floorspaceid", "mean_headcount", "max_headcount", "n_readings", "is_event"]
    out["mean_headcount"] = out["mean_headcount"].round(3)
    out["max_headcount"] = out["max_headcount"].astype(int)
    out[cols].to_csv(path, index=False)


def load_hourly(path: str | Path, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Load a saved hourly CSV and restore time-zone-aware timestamps.

    Args:
        path: CSV written by :func:`save_hourly`.
        tz: Local time zone of the timestamps.

    Returns:
        Hourly table with an ``hour`` column of local, time-zone-aware timestamps.
    """
    df = pd.read_csv(path)
    df["hour"] = pd.to_datetime(df["hour_start_local"], utc=True).dt.tz_convert(tz)
    return df
