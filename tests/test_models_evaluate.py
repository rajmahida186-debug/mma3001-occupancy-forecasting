"""Tests for the baseline, the timetable and the shared scoring functions."""

import pandas as pd
import pytest

from occupancy import evaluate
from occupancy.models import FixedTimetable, ProfileBaseline


def table(rows):
    """Small hourly table from (date, weekday, hour, headcount) tuples for one space."""
    df = pd.DataFrame(rows, columns=["date", "weekday", "hour_of_day", "mean_headcount"])
    df["space"] = "Space A"
    df["day_type"] = df["weekday"].isin(["Saturday", "Sunday"]).map({True: "Weekend", False: "Weekday"})
    return df


def test_baseline_predicts_the_historical_average():
    """Two past Tuesdays at 2 pm with 2 and 4 people -> forecast 3 people."""
    train = table([("2024-05-07", "Tuesday", 14, 2.0), ("2024-05-14", "Tuesday", 14, 4.0)])
    model = ProfileBaseline().fit(train)
    assert model.predict(table([("2025-05-06", "Tuesday", 14, 0.0)])).iloc[0] == pytest.approx(3.0)


def test_baseline_falls_back_when_weekday_unseen():
    """No Wednesday data -> use the average for that hour across all days."""
    train = table([("2024-05-07", "Tuesday", 14, 2.0), ("2024-05-09", "Thursday", 14, 6.0)])
    pred = ProfileBaseline().fit(train).predict(table([("2025-05-07", "Wednesday", 14, 0.0)]))
    assert pred.iloc[0] == pytest.approx(4.0)


def test_timetable_is_on_7am_to_7pm_weekdays_only():
    df = table([("d", "Monday", 6, 0), ("d", "Monday", 7, 0), ("d", "Monday", 18, 0),
                ("d", "Monday", 19, 0), ("d", "Saturday", 12, 0)])
    assert list(FixedTimetable().schedule(df)) == [False, True, True, False, False]


def test_split_is_chronological_and_complete():
    df = table([("2024-12-31", "Tuesday", 1, 1), ("2025-01-01", "Wednesday", 1, 1),
                ("2025-12-31", "Wednesday", 1, 1), ("2026-01-01", "Thursday", 1, 1)])
    parts = evaluate.chronological_split(df)
    assert [len(parts[k]) for k in ("train", "validation", "test")] == [1, 2, 1]
    assert parts["train"]["date"].max() < parts["validation"]["date"].min()


def test_error_metrics_by_hand():
    """Misses of 1 and 3 people: MAE = 2, RMSE = sqrt(5)."""
    assert evaluate.mae([1, 5], [2, 2]) == pytest.approx(2.0)
    assert evaluate.rmse([1, 5], [2, 2]) == pytest.approx(5 ** 0.5)


def test_hvac_outcomes_count_every_case():
    """One hour of each type: correct on, wasted, discomfort, correct off."""
    actual = [2.0, 0.2, 3.0, 0.1]
    on = [True, True, False, False]
    out = evaluate.hvac_outcomes(actual, on, threshold=1.0)
    assert (out["correct_on"], out["wasted"], out["discomfort"], out["correct_off"]) == (1, 1, 1, 1)
    assert out["wasted_rate"] == pytest.approx(0.5) and out["accuracy"] == pytest.approx(0.5)


def test_negative_forecasts_are_clipped_to_zero():
    df = table([("d", "Monday", 3, 0.0)])
    res = evaluate.score_forecast(df, pd.Series([-2.0]), "x")
    assert res["mae"] == pytest.approx(0.0)


def test_schedule_score_has_no_headcount_error():
    df = table([("d", "Monday", 9, 2.0)])
    res = evaluate.score_schedule(df, pd.Series([True]), "timetable")
    assert pd.isna(res["mae"]) and res["correct_on"] == 1
