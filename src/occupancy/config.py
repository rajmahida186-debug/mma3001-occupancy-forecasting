"""Project-wide settings: file locations, time zone and space names.

Keeping these in one place means every module (and the notebooks) uses the
same names and paths, so a change only has to be made once.
"""

from pathlib import Path

#: Root folder of the repository (two levels up from this file).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Folder where the user places the original Monash files (not in Git).
RAW_DIR = REPO_ROOT / "data" / "raw"

#: Folder for small cleaned tables produced by the code (kept in Git).
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

#: Folder for figures used in the report.
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

#: Raw occupancy file name as supplied by the unit.
RAW_OCCUPANCY_FILE = "5occupancySensor_MayToDec2024_9MRows.csv"

#: Raw environmental file name as supplied by the unit.
RAW_ENV_FILE = "5EnvSensor_MayToDec2024_180kRows.csv"

#: Local time zone of the Monash Clayton buildings (handles AEST/AEDT).
LOCAL_TZ = "Australia/Melbourne"

#: Short, readable names for the five occupancy spaces, keyed by the
#: ``floorspaceid`` used in the raw data. G.20 and G.25 were matched to rooms
#: via the SpaceId column of the "OCC" tab in ``Sensor ID and Locations.xlsx``;
#: the other three codes do not appear in that spreadsheet.
SPACE_NAMES = {
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b": "G.20 Breakout",
    "227f1068-9e41-469f-bef0-275f27a0bdba": "G.25 MSM Lab",
    "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7": "Space A",
    "941f0352-bb69-47d0-a9c5-ac29eb4f7319": "Space B",
    "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8": "Space C",
}

#: Display order for spaces in tables and figures.
SPACE_ORDER = list(SPACE_NAMES.values())

#: Headcount at or above which an hour is flagged as an unusual event
#: (e.g. Monash Open Day, 4 Aug 2024, peak 72 people). Flagged hours are kept,
#: not deleted, so they can be discussed and excluded explicitly if needed.
EVENT_HEADCOUNT_THRESHOLD = 30
