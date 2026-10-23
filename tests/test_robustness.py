"""Tests for the robustness-check helpers and the random-forest comparison model."""

import numpy as np
import pandas as pd
import pytest

from occupancy import features, robustness
from occupancy.config import LOCAL_TZ
from occupancy.models import GradientBoostingForecaster, LinearRegressionForecaster, RandomForestForecaster

from test_features_ml import make_hourly


def with_hour(df):
    """Add a time-zone-aware ``hour`` column built from date + hour_of_day."""
    local = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour_of_day"], unit="h")
    return df.assign(hour=local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT"))


def test_bootstrap_detects_a_clearly_better_forecast():
    df = make_hourly(days=30, value=lambda d, h: 2.0)
    good, bad = pd.Series(2.1, index=df.index), pd.Series(3.0, index=df.index)
    res = robustness.bootstrap_mae_difference(df, good, bad, n_boot=500)
    assert res["difference"] == pytest.approx(0.1 - 1.0)
    assert res["ci_high"] < 0 and res["share_a_better"] == 1.0


def test_bootstrap_interval_includes_zero_for_equal_forecasts():
    rng = np.random.default_rng(1)
    df = make_hourly(days=30, value=lambda d, h: 2.0)
    a = pd.Series(2 + rng.normal(0, 1, len(df)), index=df.index)
    b = pd.Series(2 + rng.normal(0, 1, len(df)), index=df.index)
    res = robustness.bootstrap_mae_difference(df, a, b, n_boot=500)
    assert res["ci_low"] < 0 < res["ci_high"]


def test_threshold_curve_lower_threshold_means_more_hours_on():
    df = make_hourly(days=2, value=lambda d, h: float(h % 3))     # 0, 1, 2 people
    curve = robustness.threshold_curve(df, df["mean_headcount"], [0.5, 1.5])
    on = curve["correct_on"] + curve["wasted"]
    assert on.iloc[0] > on.iloc[1]
    # At 1.5 the 1-person hours are missed -> discomfort; at 0.5 nothing is missed.
    assert curve["discomfort"].tolist() == [0, 16]


def test_walk_forward_trains_only_on_earlier_months():
    data = features.add_features(make_hourly(days=70, start="2024-01-01"))
    seen = []

    class Spy:
        def fit(self, train):
            seen.append(train["date"].max())
            return self

        def predict(self, df):
            return pd.Series(0.0, index=df.index)

    out = robustness.walk_forward(data, ["2024-02", "2024-03"], {"spy": Spy})
    assert seen == ["2024-01-31", "2024-02-29"]
    assert set(out["month"]) == {"2024-02", "2024-03"}


def test_fill_missing_as_empty_fills_only_days_with_readings():
    h = with_hour(make_hourly(days=3, value=lambda d, h: 1.0))
    h = h[~((h.date == "2024-03-05") & (h.hour_of_day == 3))]      # one missing hour
    h = h[h.date != "2024-03-06"]                                   # a whole missing day
    out = robustness.fill_missing_as_empty(h)
    filled = out[out.imputed]
    assert len(filled) == 1 and filled.iloc[0].mean_headcount == 0
    assert (out.date == "2024-03-06").sum() == 0


def test_fill_missing_skips_the_hour_removed_by_daylight_saving():
    """On 6 Oct 2024, 2:00–2:59 am does not exist in Melbourne."""
    h = with_hour(make_hourly(days=1, start="2024-10-06"))
    h = h[h.hour.notna()]
    out = robustness.fill_missing_as_empty(h)
    assert not out.imputed.any() and len(out) == 23


def test_hour_ahead_lag_is_the_previous_hour_across_midnight():
    h = with_hour(make_hourly(days=2))
    out = robustness.add_hour_ahead_lag(h)
    r = out[(out.date == "2024-03-05") & (out.hour_of_day == 0)].iloc[0]
    assert r.lag_1h == 23                                           # day 0, hour 23


def test_public_holidays_include_anzac_day():
    flags = robustness.public_holidays(pd.Series(["2025-04-25", "2025-04-24"]))
    assert flags.tolist() == [True, False]


def test_model_can_run_with_a_reduced_set_of_inputs():
    f = features.add_features(make_hourly(days=40, value=lambda d, h: h % 4))
    m = GradientBoostingForecaster(max_iter=30, features=["hour_of_day", "weekday_num"]).fit(f)
    assert m.predict(f).notna().all()
    assert list(m.input_importance(f, n_repeats=2)["feature"].sort_values()) == ["hour_of_day", "weekday_num"]


def test_random_forest_handles_missing_inputs_and_is_non_negative():
    f = features.add_features(make_hourly(days=40, value=lambda d, h: h % 3))
    m = RandomForestForecaster(n_estimators=20).fit(f)
    pred = m.predict(f)
    assert pred.notna().all() and (pred >= 0).all()


def test_linear_regression_recovers_an_hourly_pattern_and_handles_gaps():
    f = features.add_features(make_hourly(days=40, value=lambda d, h: 3.0 if 9 <= h < 17 else 0.0))
    test = f.tail(24).copy()
    test[["lag_1d", "lag_7d"]] = np.nan
    pred = LinearRegressionForecaster().fit(f).predict(test)
    assert pred.notna().all() and (pred >= 0).all()
    busy = test.hour_of_day.between(9, 16)
    assert pred[busy].mean() > 2 and pred[~busy].mean() < 1


def test_fit_cost_reports_time_memory_and_size():
    from occupancy.models import ProfileBaseline
    f = features.add_features(make_hourly(days=10))
    cost = robustness.fit_cost(ProfileBaseline, f, f.tail(24), repeats=2)
    assert set(cost) == {"fit_seconds", "forecast_one_day_seconds", "training_memory_mb", "stored_model_mb"}
    assert cost["fit_seconds"] >= 0 and cost["stored_model_mb"] > 0
