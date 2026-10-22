"""Model inputs (features) for the day-ahead occupancy forecast.

**The forecasting situation.** At midnight before day *D*, the building
system asks: "how many people will be in each space in each hour of day *D*?"
At that moment it knows the calendar (hour, weekday, semester or not) and
everything measured up to the end of day *D − 1*. It does **not** know anything
measured on day *D* itself. Every input below obeys that rule, so the model
never "peeks" at the answer (no data leakage).

Inputs built here, for target hour *h* of day *D* in a given space:

=====================  =====================================================
Column                 Meaning
=====================  =====================================================
``hour_of_day``        *h* (0–23)
``weekday_num``        0 = Monday … 6 = Sunday
``is_weekend``         1 on Saturday/Sunday, else 0
``in_semester``        1 during Monash semester teaching weeks, else 0
``space``              which space (categorical)
``lag_1d``             headcount at hour *h* on day *D − 1* ("yesterday")
``lag_7d``             headcount at hour *h* on day *D − 7* ("last week")
``mean_4w``            mean headcount at hour *h* on days *D − 7, −14, −21,
                       −28* (same weekday, previous four weeks)
``yesterday_mean``     mean headcount over all hours of day *D − 1*
=====================  =====================================================

Hours with no sensor readings stay missing (NaN). The gradient-boosting model
handles missing inputs natively, so no values are invented.
"""

from __future__ import annotations

import pandas as pd

#: Monash Clayton semester *teaching weeks* (first to last teaching day,
#: including the one-week mid-semester break; excludes SWOT VAC and exams).
#: Sources: Monash "Census dates and teaching periods 2024" (S1/S2 2024);
#: Monash principal dates 2025 (S1 2025 and start of S2 2025); Scape "Semester
#: dates and deadlines 2026" (S1 2026). The end of S2 2025 (24 Oct) was not
#: listed on the pages accessed and follows the standard 13-week pattern
#: (12 teaching weeks + 1 break week) used in every other semester.
SEMESTER_TEACHING = [
    ("2024-02-26", "2024-05-24"),  # Semester one 2024
    ("2024-07-22", "2024-10-18"),  # Semester two 2024
    ("2025-03-03", "2025-05-30"),  # Semester one 2025
    ("2025-07-28", "2025-10-24"),  # Semester two 2025 (end inferred, see above)
    ("2026-03-02", "2026-05-29"),  # Semester one 2026
]

#: Input columns used by the machine-learning model, in order.
FEATURES = ["space", "hour_of_day", "weekday_num", "is_weekend", "in_semester",
            "lag_1d", "lag_7d", "mean_4w", "yesterday_mean"]

#: Same-weekday lags (days back) averaged into ``mean_4w``.
WEEKLY_LAGS = (7, 14, 21, 28)


def in_semester(dates: pd.Series) -> pd.Series:
    """Flag dates that fall within Monash semester teaching weeks.

    Args:
        dates: Dates as ``YYYY-MM-DD`` strings.

    Returns:
        Integer Series, 1 inside a teaching period and 0 outside.
    """
    flag = pd.Series(False, index=dates.index)
    for start, end in SEMESTER_TEACHING:
        flag |= dates.between(start, end)  # string comparison works for ISO dates
    return flag.astype(int)


def _day_by_hour_grid(hourly: pd.DataFrame, target: str) -> pd.DataFrame:
    """Reshape to one row per (space, calendar day), one column per hour 0–23.

    Every calendar day between the first and last reading is included, so a
    row shifted down by *k* rows is exactly *k* days earlier. Days or hours with
    no readings are NaN. (On the autumn daylight-saving day one clock hour
    occurs twice; the two values are averaged.)
    """
    grid = hourly.pivot_table(index=["space", "date"], columns="hour_of_day",
                              values=target, aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    all_days = pd.date_range(hourly["date"].min(), hourly["date"].max(), freq="D").strftime("%Y-%m-%d")
    full_index = pd.MultiIndex.from_product([grid.index.levels[0], all_days], names=["space", "date"])
    return grid.reindex(full_index)


def add_features(hourly: pd.DataFrame, target: str = "mean_headcount") -> pd.DataFrame:
    """Add the day-ahead model inputs to an hourly occupancy table.

    Lags are computed on the full table (all periods together), which is
    correct: a forecast for 2 Jan 2025 may use readings from 26 Dec 2024,
    because those were known by midnight on 1 Jan.

    Args:
        hourly: Hourly table with ``space``, ``date``, ``hour_of_day``,
            ``weekday``, ``day_type`` and the target column.
        target: Column to forecast.

    Returns:
        Copy of ``hourly`` with the extra columns listed in :data:`FEATURES`.
    """
    grid = _day_by_hour_grid(hourly, target)
    by_space = grid.groupby(level="space", sort=False)

    def lagged(days: int) -> pd.DataFrame:
        # Shift within each space so one space's data never leaks into another's.
        return by_space.shift(days)

    lags = {"lag_1d": lagged(1), "lag_7d": lagged(7)}
    lags["mean_4w"] = sum(lagged(k).fillna(0) for k in WEEKLY_LAGS) / \
        sum(lagged(k).notna().astype(int) for k in WEEKLY_LAGS)       # mean of available weeks
    yesterday_mean = grid.mean(axis=1, skipna=True).groupby(level="space", sort=False).shift(1)

    # Back to long format: one row per (space, date, hour) with each lag.
    long = pd.concat({name: frame.stack(future_stack=True) for name, frame in lags.items()}, axis=1)
    long.index.names = ["space", "date", "hour_of_day"]
    long = long.reset_index()
    long = long.merge(yesterday_mean.rename("yesterday_mean").reset_index(), on=["space", "date"])

    out = hourly.merge(long, on=["space", "date", "hour_of_day"], how="left")
    out["weekday_num"] = pd.to_datetime(out["date"]).dt.weekday
    out["is_weekend"] = out["day_type"].eq("Weekend").astype(int)
    out["in_semester"] = in_semester(out["date"])
    out.index = hourly.index
    return out
