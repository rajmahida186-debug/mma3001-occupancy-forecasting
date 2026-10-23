"""Tests for the occupancy cleaning pipeline (``occupancy.clean``).

Each test uses a tiny hand-made table where the correct answer can be worked
out by hand, so a failure points directly at the broken step.
"""

import pandas as pd
import pytest

from occupancy import clean

SPACE = "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b"  # G.20 Breakout


def make_events(rows):
    """Build a small events table from (timestamp, headcount) pairs for one space."""
    return pd.DataFrame({
        "floorspaceid": SPACE,
        "occupancystatus": "CurrentlyOccupied",
        "headcount": [h for _, h in rows],
        "collecteddate": [t for t, _ in rows],
    })


def test_local_hour_rounds_down_to_start_of_hour():
    """13:02 and 13:59 local time both belong to the 13:00 hour."""
    hours = clean.to_local_hour(pd.Series(["2024-05-14 13:02:10+10", "2024-05-14 13:59:59+10"]))
    assert list(hours.dt.hour) == [13, 13]


def test_local_hour_survives_daylight_saving_end():
    """When clocks go back (6 Apr 2025), 2 am happens twice; both must be kept apart."""
    stamps = pd.Series(["2025-04-06 02:30:00+11", "2025-04-06 02:30:00+10"])
    hours = clean.to_local_hour(stamps)
    assert hours.iloc[0] != hours.iloc[1]          # two different real hours
    assert list(hours.dt.hour) == [2, 2]           # both labelled 2 am locally


def test_local_hour_survives_daylight_saving_start():
    """When clocks go forward (5 Oct 2025), 1:59 am is followed by 3:00 am."""
    stamps = pd.Series(["2025-10-05 01:59:00+10", "2025-10-05 03:01:00+11"])
    assert list(clean.to_local_hour(stamps).dt.hour) == [1, 3]


def test_unparseable_timestamp_raises():
    """Invalid timestamps must fail loudly rather than become silent gaps."""
    with pytest.raises((ValueError, TypeError)):
        clean.to_local_hour(pd.Series(["not a date"]))


def test_exact_duplicates_are_removed():
    events = make_events([("2024-05-14 13:00:00+10", 2), ("2024-05-14 13:00:00+10", 2),
                          ("2024-05-14 13:05:00+10", 3)])
    assert len(clean.drop_duplicate_events(events)) == 2


def test_hourly_mean_max_and_count():
    """Readings of 1, 2 and 6 people in one hour give mean 3, max 6, count 3."""
    events = make_events([("2024-05-14 13:00:00+10", 1), ("2024-05-14 13:20:00+10", 2),
                          ("2024-05-14 13:40:00+10", 6)])
    hourly = clean.combine_partial_hourly([clean.aggregate_hourly(events)])
    row = hourly.iloc[0]
    assert row["mean_headcount"] == pytest.approx(3.0)
    assert row["max_headcount"] == 6
    assert row["n_readings"] == 3


def test_hour_split_across_chunks_is_combined_exactly():
    """An hour split between two file chunks must give the same mean as one chunk."""
    a = make_events([("2024-05-14 13:00:00+10", 1), ("2024-05-14 13:20:00+10", 2)])
    b = make_events([("2024-05-14 13:40:00+10", 6)])
    hourly = clean.combine_partial_hourly([clean.aggregate_hourly(a), clean.aggregate_hourly(b)])
    assert len(hourly) == 1
    assert hourly.iloc[0]["mean_headcount"] == pytest.approx(3.0)


def test_hours_without_readings_are_missing_not_zero():
    """A gap between 13:00 and 16:00 must not create zero-occupancy rows."""
    events = make_events([("2024-05-14 13:10:00+10", 2), ("2024-05-14 16:10:00+10", 1)])
    hourly = clean.combine_partial_hourly([clean.aggregate_hourly(events)])
    assert len(hourly) == 2
    assert (hourly["mean_headcount"] > 0).all()


def test_calendar_columns_and_space_names():
    events = make_events([("2024-08-04 11:20:00+10", 5)])  # a Sunday
    hourly = clean.add_calendar_columns(clean.combine_partial_hourly([clean.aggregate_hourly(events)]))
    row = hourly.iloc[0]
    assert row["space"] == "G.20 Breakout"
    assert row["weekday"] == "Sunday" and row["day_type"] == "Weekend"
    assert row["hour_of_day"] == 11


def test_event_flag_marks_open_day_peak():
    hourly = pd.DataFrame({"max_headcount": [3, 72]})
    assert list(clean.flag_events(hourly, threshold=30)["is_event"]) == [False, True]


def test_daily_coverage_counts_hours_and_fills_empty_days():
    hourly = pd.DataFrame({"date": ["2024-05-01", "2024-05-01", "2024-05-03"],
                           "space": ["Space C"] * 3})
    cov = clean.daily_coverage(hourly)
    assert list(cov.index) == ["2024-05-01", "2024-05-02", "2024-05-03"]
    assert list(cov["Space C"]) == [2, 0, 1]


def test_save_and_load_round_trip(tmp_path):
    """Saving and reloading keeps both repeated 2 am hours distinct."""
    events = make_events([("2025-04-06 02:30:00+11", 1), ("2025-04-06 02:30:00+10", 4)])
    hourly = clean.flag_events(clean.add_calendar_columns(
        clean.combine_partial_hourly([clean.aggregate_hourly(events)])))
    path = tmp_path / "hourly.csv"
    clean.save_hourly(hourly, path)
    back = clean.load_hourly(path)
    assert back["hour"].nunique() == 2
    assert list(back["mean_headcount"]) == [1.0, 4.0]


def test_missing_raw_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean.build_hourly_occupancy(tmp_path / "does_not_exist.csv")


def test_fast_and_general_timestamp_parsing_agree():
    """The fast parser must give exactly the same instants as the general one."""
    ts = pd.Series(["2024-05-14 13:02:10+10", "2025-04-06 02:30:00+11", "2025-04-06 02:30:00+10",
                    "2024-10-06 03:15:59+11", "2023-11-24 00:36:52+11"])
    assert clean.parse_utc_fast(ts).equals(clean.parse_utc_general(ts))


def test_other_timestamp_layouts_fall_back_to_the_general_parser():
    """Fractional seconds or a ``+10:00`` offset are not the fast layout but still work."""
    ts = pd.Series(["2024-05-14 13:02:10.5+10", "2024-05-14 13:59:59+10:00"])
    with pytest.raises(ValueError):
        clean.parse_utc_fast(ts)
    out = clean.to_local_hour(ts)
    assert (out.dt.hour == 13).all()
