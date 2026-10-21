"""Forecasting models and HVAC schedules.

Two simple reference solutions are defined here, against which the
machine-learning model (added in a later stage) is compared:

* :class:`ProfileBaseline` – a *forecast*: the historical average headcount for
  each space, weekday and hour of day ("Tuesdays at 2 pm average 2.3 people").
* :class:`FixedTimetable` – a *schedule* with no forecast at all: HVAC on during
  fixed hours (default 8 am–8 pm, Monday–Friday, matching scheduled class
  hours), as many buildings run today.
"""

from __future__ import annotations

import pandas as pd


class ProfileBaseline:
    """Average-profile forecast: mean headcount per space × weekday × hour.

    If a combination was never seen in training, the forecast falls back to the
    space × hour average, then to the space average, so every hour gets a value.

    Attributes:
        profile_: Learned means indexed by (space, weekday, hour_of_day).
        by_hour_: Fallback means indexed by (space, hour_of_day).
        by_space_: Fallback means indexed by space.
    """

    keys = ["space", "weekday", "hour_of_day"]

    def fit(self, train: pd.DataFrame, target: str = "mean_headcount") -> "ProfileBaseline":
        """Learn the average profile from training data.

        Args:
            train: Hourly table with ``space``, ``weekday``, ``hour_of_day`` and
                the target column.
            target: Column to average (people).

        Returns:
            The fitted model (to allow ``ProfileBaseline().fit(train)``).
        """
        self.profile_ = train.groupby(self.keys)[target].mean()
        self.by_hour_ = train.groupby(["space", "hour_of_day"])[target].mean()
        self.by_space_ = train.groupby("space")[target].mean()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Forecast mean headcount for each row of ``df``.

        Args:
            df: Rows to forecast, with ``space``, ``weekday`` and ``hour_of_day``.

        Returns:
            Forecast mean headcount (people), aligned with ``df``'s index.

        Raises:
            AttributeError: If called before :meth:`fit`.
        """
        idx = pd.MultiIndex.from_frame(df[self.keys])
        pred = pd.Series(self.profile_.reindex(idx).to_numpy(), index=df.index)
        # Fall back to coarser averages where the exact combination is unseen.
        hour_idx = pd.MultiIndex.from_frame(df[["space", "hour_of_day"]])
        pred = pred.fillna(pd.Series(self.by_hour_.reindex(hour_idx).to_numpy(), index=df.index))
        pred = pred.fillna(df["space"].map(self.by_space_))
        return pred


class FixedTimetable:
    """Conventional HVAC timetable: on during fixed hours on chosen days.

    This is an *assumed* schedule; the real Monash HVAC schedule was not
    supplied with the data. The default 8 am–8 pm weekday window follows the
    hours in which classes are scheduled (Monash's Student Timetable Procedure
    permits teaching 8.00 am–9.00 pm Mon–Fri; weekday hours from 8 pm are occupied
    less than half the time, and 8–8 performed best of the windows tested
    on 2025 data).

    Args:
        start_hour: First hour of the day the system is on (inclusive).
        end_hour: Hour at which the system turns off (exclusive).
        weekdays_only: If True, the system is off all weekend.
    """

    def __init__(self, start_hour: int = 8, end_hour: int = 20, weekdays_only: bool = True):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.weekdays_only = weekdays_only

    def schedule(self, df: pd.DataFrame) -> pd.Series:
        """Return True for each hour the timetable would condition the space.

        Args:
            df: Rows with ``hour_of_day`` and ``day_type`` columns.

        Returns:
            Boolean Series, True = HVAC on.
        """
        on = df["hour_of_day"].between(self.start_hour, self.end_hour - 1)
        if self.weekdays_only:
            on &= df["day_type"].eq("Weekday")
        return on
