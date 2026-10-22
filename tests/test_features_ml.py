"""Tests for the day-ahead inputs and the gradient-boosting model.

The most important property tested is *no leakage*: an input for day D must
never depend on anything measured on day D itself.
"""

import numpy as np
import pandas as pd
import pytest

from occupancy import features
from occupancy.models import GradientBoostingForecaster


def make_hourly(days=35, spaces=("Space A",), value=None, start="2024-03-04"):
    """Complete hourly table for consecutive days. Default value encodes day*100 + hour."""
    rows = []
    for s in spaces:
        for d, date in enumerate(pd.date_range(start, periods=days, freq="D")):
            for h in range(24):
                v = value(d, h) if value else d * 100 + h
                rows.append((s, date.strftime("%Y-%m-%d"), h, date.day_name(),
                             "Weekend" if date.weekday() >= 5 else "Weekday", float(v)))
    return pd.DataFrame(rows, columns=["space", "date", "hour_of_day", "weekday", "day_type",
                                       "mean_headcount"])


def row(df, date, hour, space="Space A"):
    return df[(df.space == space) & (df.date == date) & (df.hour_of_day == hour)].iloc[0]


def test_lag_1d_and_7d_are_same_hour_on_earlier_days():
    f = features.add_features(make_hourly())
    r = row(f, "2024-03-14", 9)                        # day index 10
    assert r.lag_1d == 909 and r.lag_7d == 309          # day 9 and day 3, hour 9


def test_inputs_for_a_day_do_not_use_that_day():
    """Changing every reading on day D leaves day D's inputs unchanged (no leakage)."""
    base = make_hourly()
    changed = base.copy()
    changed.loc[changed.date == "2024-03-14", "mean_headcount"] = 999.0
    cols = ["lag_1d", "lag_7d", "mean_4w", "yesterday_mean"]
    a = features.add_features(base)
    b = features.add_features(changed)
    same_day = a.date == "2024-03-14"
    pd.testing.assert_frame_equal(a.loc[same_day, cols], b.loc[same_day, cols])
    # ...but the next day's "yesterday" inputs do change, as they should.
    assert row(b, "2024-03-15", 9).lag_1d == 999.0


def test_missing_yesterday_gives_missing_lag_not_zero():
    h = make_hourly()
    h = h[h.date != "2024-03-13"]
    assert np.isnan(row(features.add_features(h), "2024-03-14", 9).lag_1d)


def test_lags_never_cross_spaces():
    h = make_hourly(spaces=("Space A", "Space B"))
    h.loc[h.space == "Space B", "mean_headcount"] += 10_000
    f = features.add_features(h)
    assert row(f, "2024-03-14", 9, "Space A").lag_1d == 909
    assert row(f, "2024-03-14", 9, "Space B").lag_1d == 10_909


def test_mean_4w_averages_the_same_weekday_over_four_weeks():
    f = features.add_features(make_hourly())
    # Day 28 -> days 21, 14, 7, 0 at hour 5: (2105 + 1405 + 705 + 5) / 4
    assert row(f, "2024-04-01", 5).mean_4w == pytest.approx((2105 + 1405 + 705 + 5) / 4)


def test_yesterday_mean_is_the_previous_days_average():
    f = features.add_features(make_hourly())
    assert row(f, "2024-03-14", 0).yesterday_mean == pytest.approx(900 + 11.5)


def test_semester_flag_uses_teaching_dates():
    dates = pd.Series(["2024-02-25", "2024-02-26", "2024-05-24", "2024-06-10", "2025-08-01"])
    assert list(features.in_semester(dates)) == [0, 1, 1, 0, 1]


def test_model_learns_a_simple_weekday_pattern():
    """Busy 9–5 on weekdays, empty otherwise: the model should tell them apart."""
    def pattern(d, h):
        weekday = pd.Timestamp("2024-03-04") + pd.Timedelta(days=d)
        return 5.0 if weekday.weekday() < 5 and 9 <= h < 17 else 0.0
    f = features.add_features(make_hourly(days=56, value=pattern))
    model = GradientBoostingForecaster(max_iter=100).fit(f)
    pred = model.predict(f)
    busy = f.mean_headcount > 0
    assert pred[busy].mean() > 4 and pred[~busy].mean() < 1


def test_model_forecasts_are_never_negative_and_accept_missing_inputs():
    f = features.add_features(make_hourly(days=40, value=lambda d, h: h % 3))
    model = GradientBoostingForecaster(max_iter=50).fit(f)
    test = f.tail(24).copy()
    test[["lag_1d", "lag_7d"]] = np.nan                     # e.g. a sensor gap
    pred = model.predict(test)
    assert pred.notna().all() and (pred >= 0).all()
