"""Data splitting and scoring, shared by every model and schedule.

Using one set of functions for all approaches guarantees they are compared on
exactly the same hours with exactly the same rules.

**Data split (chronological, never random):** a model is only ever tested on
hours *later* than everything it learned from, as it would be used in practice.

* train: before 1 Jan 2025
* validation: 2025 (used to compare and tune approaches)
* test: 1 Jan – 22 Apr 2026 (kept aside, used once for the final result)

Space C has no data after Dec 2024, so it only appears in training.

**HVAC outcome of each hour**, given an *occupied threshold* (default 1 person):

======================  ==================  ==================
\\                       Actually occupied   Actually empty
======================  ==================  ==================
HVAC on                 correct             **wasted** energy
HVAC off                **discomfort**      correct (saving)
======================  ==================  ==================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Boundaries of the chronological split (local dates).
VALIDATION_START = "2025-01-01"
TEST_START = "2026-01-01"

#: Default mean headcount (people) at or above which an hour counts as occupied.
#: Chosen because ~0.5 people is the night-time sensor background level.
OCCUPIED_THRESHOLD = 1.0


def chronological_split(hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the hourly table into train / validation / test periods by date.

    Args:
        hourly: Hourly table with a ``date`` column (``YYYY-MM-DD`` strings).

    Returns:
        Dict with keys ``"train"``, ``"validation"`` and ``"test"``.
    """
    date = hourly["date"]
    return {
        "train": hourly[date < VALIDATION_START],
        "validation": hourly[(date >= VALIDATION_START) & (date < TEST_START)],
        "test": hourly[date >= TEST_START],
    }


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error, in people: the average size of a forecast miss."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root mean squared error, in people: like MAE but penalises big misses more."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def hvac_outcomes(actual_headcount: pd.Series, hvac_on: pd.Series,
                  threshold: float = OCCUPIED_THRESHOLD) -> dict[str, float]:
    """Count correct, wasted and discomfort hours for an on/off schedule.

    Args:
        actual_headcount: Measured mean headcount per hour (people).
        hvac_on: Boolean, True where the schedule conditions the space.
        threshold: Headcount at or above which an hour is "actually occupied".

    Returns:
        Dict with hour counts ``correct_on``, ``wasted``, ``discomfort``,
        ``correct_off``, plus ``wasted_rate`` (share of empty hours that were
        conditioned), ``discomfort_rate`` (share of occupied hours left
        unconditioned) and ``accuracy`` (share of hours decided correctly).
    """
    occupied = np.asarray(actual_headcount) >= threshold
    on = np.asarray(hvac_on, dtype=bool)
    out = {
        "correct_on": int(np.sum(on & occupied)),
        "wasted": int(np.sum(on & ~occupied)),
        "discomfort": int(np.sum(~on & occupied)),
        "correct_off": int(np.sum(~on & ~occupied)),
    }
    n_empty, n_occ = out["wasted"] + out["correct_off"], out["correct_on"] + out["discomfort"]
    out["wasted_rate"] = out["wasted"] / n_empty if n_empty else float("nan")
    out["discomfort_rate"] = out["discomfort"] / n_occ if n_occ else float("nan")
    out["accuracy"] = (out["correct_on"] + out["correct_off"]) / len(on) if len(on) else float("nan")
    return out


def score_forecast(df: pd.DataFrame, predicted: pd.Series, name: str,
                   threshold: float = OCCUPIED_THRESHOLD) -> dict:
    """Score a headcount forecast: errors in people plus the HVAC outcomes.

    The forecast is turned into an on/off schedule by switching the HVAC on
    wherever the forecast reaches the occupied threshold.

    Args:
        df: Rows being scored, with ``mean_headcount`` (the measured value).
        predicted: Forecast headcount for the same rows.
        name: Label for the approach (appears in results tables).
        threshold: Occupied threshold (people).

    Returns:
        Dict of results: ``approach``, ``hours``, ``mae``, ``rmse`` and the
        keys returned by :func:`hvac_outcomes`.
    """
    pred = np.clip(np.asarray(predicted, dtype=float), 0, None)  # headcount cannot be negative
    return {"approach": name, "hours": len(df),
            "mae": mae(df["mean_headcount"], pred), "rmse": rmse(df["mean_headcount"], pred),
            **hvac_outcomes(df["mean_headcount"], pred >= threshold, threshold)}


def score_schedule(df: pd.DataFrame, hvac_on: pd.Series, name: str,
                   threshold: float = OCCUPIED_THRESHOLD) -> dict:
    """Score an on/off schedule that makes no headcount forecast (MAE not applicable).

    Args:
        df: Rows being scored, with ``mean_headcount``.
        hvac_on: Boolean schedule for the same rows.
        name: Label for the approach.
        threshold: Occupied threshold (people).

    Returns:
        Same keys as :func:`score_forecast`, with ``mae`` and ``rmse`` as NaN.
    """
    return {"approach": name, "hours": len(df), "mae": float("nan"), "rmse": float("nan"),
            **hvac_outcomes(df["mean_headcount"], hvac_on, threshold)}
