"""Forecasting models and HVAC schedules.

The machine-learning model, :class:`GradientBoostingForecaster`, is compared
against two simple reference solutions:

* :class:`ProfileBaseline` – a *forecast*: the historical average headcount for
  each space, weekday and hour of day ("Tuesdays at 2 pm average 2.3 people").
* :class:`FixedTimetable` – a *schedule* with no forecast at all: HVAC on during
  fixed hours (default 8 am–8 pm, Monday–Friday, matching scheduled class
  hours), as many buildings run today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from .features import FEATURES


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


#: Settings chosen by the tuning grid in notebook 03 (lowest 2025 validation MAE).
TUNED_SETTINGS = dict(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, loss="poisson")


class GradientBoostingForecaster:
    """Day-ahead machine-learning forecast using gradient-boosted decision trees.

    Gradient boosting builds many small decision trees one after another; each
    new tree corrects the errors left by the trees before it. scikit-learn's
    ``HistGradientBoostingRegressor`` is used because it is fast on tens of
    thousands of rows, captures non-linear patterns and interactions (e.g.
    "2 pm *and* semester *and* Tuesday") without manual feature design, and
    accepts missing inputs (NaN) natively – important because sensor gaps
    leave some lag values unknown.

    Inputs are the columns in :data:`occupancy.features.FEATURES`, all known at
    midnight before the forecast day. Forecasts are clipped at 0 people.

    Args:
        learning_rate: How much each new tree is allowed to correct (smaller =
            slower, more cautious learning).
        max_iter: Number of trees.
        max_leaf_nodes: Size (complexity) of each tree.
        min_samples_leaf: Minimum hours that must fall in each tree leaf; larger
            values stop the model from memorising one-off events.
        loss: ``"squared_error"`` or ``"poisson"`` (suited to non-negative counts).
        random_state: Seed, so results are repeatable.
        features: Input columns to use (default :data:`occupancy.features.FEATURES`).
            Changed only for robustness tests, e.g. removing inputs.

    Attributes:
        model_: The fitted scikit-learn estimator.
        spaces_: Space names seen in training (fixes the category coding).
    """

    def __init__(self, learning_rate: float = 0.05, max_iter: int = 300,
                 max_leaf_nodes: int = 31, min_samples_leaf: int = 20,
                 loss: str = "squared_error", random_state: int = 0,
                 features: list[str] | None = None):
        self.params = dict(learning_rate=learning_rate, max_iter=max_iter,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           loss=loss, random_state=random_state)
        self.features = list(features) if features is not None else list(FEATURES)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the input columns and encode ``space`` as a fixed category."""
        X = df[self.features].copy()
        if "space" in X:
            X["space"] = pd.Categorical(X["space"], categories=self.spaces_)
        return X

    def fit(self, train: pd.DataFrame, target: str = "mean_headcount") -> "GradientBoostingForecaster":
        """Train on rows that have features (see :func:`occupancy.features.add_features`).

        Args:
            train: Training rows with every column in ``FEATURES`` and the target.
            target: Column to forecast (people).

        Returns:
            The fitted model.
        """
        self.spaces_ = sorted(train["space"].unique())
        self.model_ = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                                    early_stopping=False, **self.params)
        self.model_.fit(self._matrix(train), train[target])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Forecast mean headcount (people, ≥ 0) for each row of ``df``."""
        pred = np.clip(self.model_.predict(self._matrix(df)), 0, None)
        return pd.Series(pred, index=df.index)

    def input_importance(self, df: pd.DataFrame, target: str = "mean_headcount",
                         n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance: how much worse the forecast gets without each input.

        Each input column is shuffled in turn (breaking its link to the
        target) and the rise in MAE is recorded. A large rise means the model
        relies on that input.

        Args:
            df: Rows to evaluate on (normally validation, not training, data).
            target: Measured column.
            n_repeats: Number of shuffles per input (results are averaged).

        Returns:
            Table with ``feature``, ``mae_increase`` (people) and ``std``.
        """
        res = permutation_importance(self.model_, self._matrix(df), df[target],
                                     scoring="neg_mean_absolute_error",
                                     n_repeats=n_repeats, random_state=0)
        return pd.DataFrame({"feature": self.features, "mae_increase": res.importances_mean,
                             "std": res.importances_std}).sort_values("mae_increase", ascending=False)


class RandomForestForecaster(GradientBoostingForecaster):
    """Alternative ML model: a random forest (used only as a comparison).

    A random forest grows many deep decision trees *independently*, each on a
    random resample of the training hours, and averages them. Unlike boosting,
    the trees do not correct each other. Randomness means results vary
    slightly with the seed. Takes the same inputs as
    :class:`GradientBoostingForecaster`; ``space`` is coded as a number.

    Args:
        n_estimators: Number of trees.
        min_samples_leaf: Minimum hours per leaf.
        random_state: Seed for the random resampling.
        features: Input columns (default :data:`occupancy.features.FEATURES`).
    """

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 10,
                 random_state: int = 0, features: list[str] | None = None):
        self.params = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                           random_state=random_state, n_jobs=-1)
        self.features = list(features) if features is not None else list(FEATURES)

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = super()._matrix(df)
        if "space" in X:
            X["space"] = X["space"].cat.codes  # random forest needs numbers
        return X

    def fit(self, train: pd.DataFrame, target: str = "mean_headcount") -> "RandomForestForecaster":
        """Train the forest (same interface as the gradient-boosting model)."""
        self.spaces_ = sorted(train["space"].unique())
        self.model_ = RandomForestRegressor(**self.params)
        self.model_.fit(self._matrix(train), train[target])
        return self


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
