"""Tests for frozen-sensor detection (``occupancy.audit``)."""

import json

import pandas as pd
import pytest

from occupancy import audit


def record(co2, unix_time):
    """Build one ``jsondata`` string in the same format as the raw file."""
    return json.dumps([
        {"values": [{"time": unix_time, "value": 4.18}], "variable": {"name": "Battery", "unit": "V"}},
        {"values": [{"time": unix_time, "value": co2}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
    ])


def test_parse_co2_reads_value_and_sensor_time():
    assert audit.parse_co2(record(603, 1702273965)) == (603, 1702273965)


def test_parse_co2_handles_bad_or_missing_data():
    assert audit.parse_co2("not json") == (None, None)
    assert audit.parse_co2(json.dumps([{"values": [{"time": 1, "value": 20}],
                                        "variable": {"name": "Temperature"}}])) == (None, None)


def make_env(sensor, times, co2s, arrivals):
    df = pd.DataFrame({"sensorid": sensor, "measured_unix": times, "co2_ppm": co2s,
                       "arrived_at": pd.to_datetime(arrivals).tz_localize("Australia/Melbourne")})
    df["measured_at"] = pd.to_datetime(df["measured_unix"], unit="s", utc=True)
    return df


def test_frozen_sensor_is_detected():
    """Many records arriving with one unchanging sensor time = frozen."""
    env = make_env("A", [100] * 5, [666] * 5,
                   ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"])
    assert list(audit.sensor_health(env)["status"]) == ["frozen"]


def test_working_sensor_is_not_flagged():
    env = make_env("B", [100, 700, 1300], [450, 470, 520],
                   ["2024-05-01 00:00", "2024-05-01 00:10", "2024-05-01 00:20"])
    assert list(audit.sensor_health(env)["status"]) == ["working"]


def test_valid_period_excludes_stuck_repeats():
    """A sensor that works, then sticks, is valid only up to the last fresh reading."""
    times = [1_000, 2_000, 3_000] + [3_000] * 20          # works, then repeats 3_000
    env = make_env("C", times, [450] * 23, pd.date_range("2024-05-01", periods=23, freq="h"))
    row = audit.valid_periods(env).iloc[0]
    assert row["valid_to"] == pd.Timestamp(2_000, unit="s", tz="UTC")


def test_never_working_sensor_has_no_valid_period():
    env = make_env("D", [100] * 10, [666] * 10, pd.date_range("2024-05-01", periods=10, freq="h"))
    assert pd.isna(audit.valid_periods(env).iloc[0]["valid_from"])


def test_missing_env_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit.load_env_co2(tmp_path / "missing.csv")
