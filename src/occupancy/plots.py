"""Figures for the report (audit Figures 1–3, results Figures 4–5).

Each function takes already-processed data and returns a Matplotlib figure,
so the same code is used by the notebook and by the report.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .config import SPACE_ORDER

# Colour-blind-checked categorical palette (fixed order, never cycled).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, TRACK = "#333333", "#777777", "#e6e6e3", "#ececea"


def _style() -> None:
    """Apply a consistent, uncluttered style to all figures."""
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    })


def occupancy_profiles(hourly: pd.DataFrame) -> plt.Figure:
    """Figure 1: mean headcount by hour of day, weekdays vs weekends.

    Args:
        hourly: Hourly table with ``space``, ``hour_of_day``, ``day_type`` and
            ``mean_headcount`` columns.

    Returns:
        Figure with two panels (weekdays, weekends), one line per space.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, day_type, title in [(axes[0], "Weekday", "Weekdays (Mon–Fri)"),
                                (axes[1], "Weekend", "Weekends (Sat–Sun)")]:
        prof = (hourly[hourly["day_type"] == day_type]
                .pivot_table(index="hour_of_day", columns="space", values="mean_headcount"))
        for colour, space in zip(COLOURS, SPACE_ORDER):
            if space in prof:
                ax.plot(prof.index, prof[space], color=colour, lw=2, label=space)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour of day")
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("Mean headcount (people)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


def coverage_timeline(coverage: pd.DataFrame, highlight: str = "Space C") -> plt.Figure:
    """Figure 2: which days each space has readings.

    Args:
        coverage: Output of :func:`occupancy.clean.daily_coverage` (rows = days,
            columns = spaces, values = hours with readings).
        highlight: Space to draw in a contrasting colour.

    Returns:
        Timeline figure, one horizontal track per space.
    """
    _style()
    days = pd.to_datetime(coverage.index)
    start, end = days.min(), days.max()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    spaces = [s for s in SPACE_ORDER if s in coverage.columns][::-1]
    for y, space in enumerate(spaces):
        ax.barh(y, (end - start).days + 1, left=start, height=0.55, color=TRACK)
        has = coverage[space].to_numpy() > 0
        # Find runs of consecutive days with data and draw each as one bar.
        edges = np.diff(np.concatenate([[0], has.astype(int), [0]]))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.barh(y, b - a, left=days[a], height=0.55,
                    color=COLOURS[1] if space == highlight else COLOURS[0])
        ax.text(end + pd.Timedelta(days=12), y, f"{has.mean():.0%} of days",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(spaces)))
    ax.set_yticklabels(spaces, fontsize=8.5, color=INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(start - pd.Timedelta(days=5), end + pd.Timedelta(days=95))
    ax.legend(handles=[Patch(color=COLOURS[0], label="Days with readings"),
                       Patch(color=TRACK, label="No readings")],
              frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    return fig


def frozen_vs_working(env: pd.DataFrame, working: str, frozen: str,
                      start: str, end: str) -> plt.Figure:
    """Figure 3: CO₂ from a working sensor next to a frozen one.

    Args:
        env: Output of :func:`occupancy.audit.load_env_co2`.
        working: Sensor ID of a working sensor.
        frozen: Sensor ID of a frozen sensor.
        start: Start of the window (local date/time string).
        end: End of the window (local date/time string).

    Returns:
        Line figure of reported CO₂ against arrival time.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    tz = env["arrived_at"].dt.tz
    t0, t1 = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
    for sid, colour, label in [(working, COLOURS[0], f"Sensor …{working[-3:]} – working"),
                               (frozen, COLOURS[1], f"Sensor …{frozen[-3:]} – frozen")]:
        d = env[(env["sensorid"] == sid) & env["arrived_at"].between(t0, t1)].sort_values("arrived_at")
        ax.plot(d["arrived_at"].dt.tz_localize(None), d["co2_ppm"], color=colour, lw=1.6, label=label)
    ax.set_ylabel("Reported CO₂ (ppm)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


def feature_importance(importance: pd.DataFrame) -> plt.Figure:
    """Figure 5: permutation importance of each model input.

    Args:
        importance: Table with columns ``feature``, ``mae_increase`` (mean
            rise in MAE, people, when that input is shuffled) and ``std``.

    Returns:
        Horizontal bar chart, most important input at the top.
    """
    _style()
    d = importance.sort_values("mae_increase")
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.barh(d["feature"], d["mae_increase"], xerr=d["std"], color=COLOURS[2],
            error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xlabel("Increase in validation MAE when the input is shuffled (people)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def example_week(df: pd.DataFrame, space: str, start: str, days: int = 7) -> plt.Figure:
    """Figure 4: one week of measured vs forecast headcount, with HVAC schedules.

    Args:
        df: Hourly rows with ``space``, ``hour``, ``mean_headcount``, a
            ``baseline`` forecast column and a boolean ``timetable`` column.
        space: Space to plot.
        start: First day of the week (``YYYY-MM-DD``).
        days: Number of days to show.

    Returns:
        Figure with headcount lines and shaded timetable "on" hours.
    """
    _style()
    tz = df["hour"].dt.tz
    t0 = pd.Timestamp(start, tz=tz)
    d = df[(df["space"] == space) & df["hour"].between(t0, t0 + pd.Timedelta(days=days))].sort_values("hour")
    t = d["hour"].dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.fill_between(t, 0, d["mean_headcount"].max() * 1.05, where=d["timetable"].to_numpy(),
                    step="post", color="#f3e4d6", label="Timetable HVAC on", linewidth=0)
    ax.plot(t, d["mean_headcount"], color=INK, lw=1.4, label="Measured")
    ax.plot(t, d["baseline"], color=COLOURS[0], lw=1.8, label="Profile baseline forecast")
    if "model" in d:
        ax.plot(t, d["model"], color=COLOURS[2], lw=1.8, label="ML forecast")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.text(t.iloc[0], 1.05, "occupied threshold (1 person)", fontsize=7, color=MUTED, va="bottom", ha="left")
    ax.set_ylabel("Mean headcount (people)")
    ax.set_title(f"{space}, week from {t0:%d %b %Y}", fontsize=9.5, color=INK, loc="left", pad=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    fig.tight_layout()
    return fig
