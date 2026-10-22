"""Robustness checks: how far can the Stage 3 results be trusted?

Each function answers one question used in notebook 04:

* :func:`bootstrap_mae_difference` – is the ML model's lower error real, or
  could it be luck in which days happened to be in the test period?
* :func:`walk_forward` – does the result hold when the model is retrained
  every month, as it would be in use?
* :func:`threshold_curve` – how do wasted and discomfort hours trade off as the
  switch-on threshold changes?
* :func:`fill_missing_as_empty` – do conclusions change if hours with no
  readings are assumed empty instead of unknown?
* :func:`add_hour_ahead_lag` – how much better is a forecast made only one
  hour ahead?
* :func:`public_holidays` – which days are Victorian public holidays?
"""

from __future__ import annotations

from typing import Callable

import holidays
import numpy as np
import pandas as pd

from . import evaluate
from .config import LOCAL_TZ


def bootstrap_mae_difference(df: pd.DataFrame, pred_a: pd.Series, pred_b: pd.Series,
                             n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Confidence interval for MAE(a) − MAE(b) by resampling whole days.

    Days, not single hours, are resampled because hours within one day are
    strongly related (a busy morning usually means a busy afternoon).
    Resampling hours would make the interval look narrower than it really is.

    Args:
        df: Rows scored, with ``date`` and ``mean_headcount``.
        pred_a: Forecast A (e.g. ML) for the same rows.
        pred_b: Forecast B (e.g. baseline).
        n_boot: Number of resamples.
        seed: Random seed (repeatable result).

    Returns:
        Dict with ``difference`` (A − B on the real data, people), ``ci_low`` and
        ``ci_high`` (95 % interval) and ``share_a_better`` (share of resamples in
        which A had the lower MAE).
    """
    actual = df["mean_headcount"].to_numpy()
    err_a = np.abs(actual - np.clip(np.asarray(pred_a, float), 0, None))
    err_b = np.abs(actual - np.clip(np.asarray(pred_b, float), 0, None))
    daily = pd.DataFrame({"date": df["date"].to_numpy(), "a": err_a, "b": err_b, "n": 1}) \
        .groupby("date")[["a", "b", "n"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(n_boot, len(daily)))   # days drawn with replacement
    totals = daily[picks].sum(axis=1)                                 # (n_boot, 3): sum a, sum b, hours
    diffs = (totals[:, 0] - totals[:, 1]) / totals[:, 2]
    return {"difference": float((err_a.sum() - err_b.sum()) / len(actual)),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "share_a_better": float(np.mean(diffs < 0))}


def walk_forward(data: pd.DataFrame, months: list[str],
                 models: dict[str, Callable[[], object]]) -> pd.DataFrame:
    """Retrain at the start of each month on everything before it, forecast that month.

    Args:
        data: Hourly table with features (see :func:`occupancy.features.add_features`).
        months: Month starts to forecast, e.g. ``["2025-01", "2025-02"]``.
        models: Name → function returning a fresh, unfitted model with
            ``fit`` and ``predict`` methods.

    Returns:
        The forecast rows with a ``month`` column and one prediction column per model.
    """
    parts = []
    for month in months:
        start = f"{month}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
        train = data[data["date"] < start]
        test = data[(data["date"] >= start) & (data["date"] < end)].copy()
        if test.empty:
            continue
        test["month"] = month
        for name, make in models.items():
            test[name] = make().fit(train).predict(test)
        parts.append(test)
    return pd.concat(parts)


def threshold_curve(df: pd.DataFrame, forecast: pd.Series, thresholds,
                    occupied_threshold: float = evaluate.OCCUPIED_THRESHOLD) -> pd.DataFrame:
    """HVAC outcomes for several switch-on thresholds.

    The *occupied* definition (what counts as a real occupied hour) stays
    fixed; only the forecast level at which the HVAC is switched on changes.
    A lower switch-on threshold conditions more hours: fewer discomfort hours
    but more wasted ones.

    Args:
        df: Rows scored, with ``mean_headcount``.
        forecast: Forecast headcount for the same rows.
        thresholds: Switch-on levels to try (people).
        occupied_threshold: Measured headcount counted as occupied.

    Returns:
        One row per threshold with the keys of :func:`occupancy.evaluate.hvac_outcomes`.
    """
    f = np.clip(np.asarray(forecast, float), 0, None)
    return pd.DataFrame([{"switch_on_at": t, **evaluate.hvac_outcomes(
        df["mean_headcount"], f >= t, occupied_threshold)} for t in thresholds])


def fill_missing_as_empty(hourly: pd.DataFrame) -> pd.DataFrame:
    """Alternative assumption: an hour with no readings had nobody present.

    Only hours on days when that space's sensor reported at least once are
    filled (so a sensor that is switched off for months is still treated as
    missing, not as an empty room). Filled rows get headcount 0 and
    ``imputed = True``. Clock hours skipped or repeated by daylight saving are
    not added.

    Args:
        hourly: Hourly table from :func:`occupancy.clean.build_hourly_occupancy`.

    Returns:
        Table with the original rows plus the filled ones, sorted by space and hour.
    """
    have = hourly.assign(imputed=False)
    days = have[["space", "date"]].drop_duplicates()
    grid = days.merge(pd.DataFrame({"hour_of_day": range(24)}), how="cross")
    grid = grid.merge(have[["space", "date", "hour_of_day"]].drop_duplicates(),
                      how="left", indicator=True)
    new = grid[grid["_merge"] == "left_only"].drop(columns="_merge")
    local = pd.to_datetime(new["date"]) + pd.to_timedelta(new["hour_of_day"], unit="h")
    new["hour"] = local.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    new = new[new["hour"].notna()].copy()
    new["weekday"] = new["hour"].dt.day_name()
    new["day_type"] = np.where(new["hour"].dt.dayofweek < 5, "Weekday", "Weekend")
    new["mean_headcount"] = 0.0
    new["max_headcount"] = 0
    new["n_readings"] = 0
    new["is_event"] = False
    new["imputed"] = True
    out = pd.concat([have, new], ignore_index=True)
    return out.sort_values(["space", "hour"]).reset_index(drop=True)


def add_hour_ahead_lag(data: pd.DataFrame, target: str = "mean_headcount") -> pd.DataFrame:
    """Add ``lag_1h``: the headcount in the previous clock hour (same space).

    Only valid for an *hour-ahead* forecast (made at the start of each hour),
    not for the day-ahead forecast used in the main model. Uses true elapsed
    time, so daylight-saving changes are handled correctly.

    Args:
        data: Table with ``space``, ``hour`` (time-zone aware) and the target.

    Returns:
        Copy of ``data`` with a ``lag_1h`` column (NaN if that hour has no reading).
    """
    prev = data[["space", "hour", target]].copy()
    prev["hour"] = prev["hour"] + pd.Timedelta(hours=1)   # value now labels the *next* hour
    prev = prev.rename(columns={target: "lag_1h"})
    out = data.merge(prev, on=["space", "hour"], how="left")
    out.index = data.index
    return out


def public_holidays(dates: pd.Series) -> pd.Series:
    """Flag Victorian public holidays (from the ``holidays`` package).

    Args:
        dates: Dates as ``YYYY-MM-DD`` strings.

    Returns:
        Boolean Series, True on a public holiday.
    """
    years = sorted({int(d[:4]) for d in dates.unique()})
    vic = holidays.Australia(subdiv="VIC", years=years)
    return dates.isin({d.strftime("%Y-%m-%d") for d in vic})
