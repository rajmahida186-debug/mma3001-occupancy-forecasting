# Data

## Source

Monash Smart Infrastructure (MSI) Occupancy and Environmental Data, supplied
as Dataset 2 for MMA3001 (2026). The data is owned by Monash University and is
used here for coursework only. It is **not** covered by this repository's
MIT licence.

## Folders

| Folder | Contents | In Git? |
|---|---|---|
| `raw/` | The original files from the unit (about 2 GB). Place them here yourself. | ❌ No: too large, and not ours to redistribute |
| `processed/` | Small cleaned tables produced by `python -m occupancy.pipeline` | ✅ Yes |

## Raw files expected in `raw/`

| File | Size | Description |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 1.6 GB | ~9.1 M occupancy events (status and headcount), 5 spaces, Nov 2023 – Apr 2026 |
| `5EnvSensor_MayToDec2024_180kRows.csv` | 350 MB | ~179 k environmental readings (CO₂, temperature, humidity, PM, etc.) from 5 sensors, stored as JSON |
| `Sensor ID and Locations.xlsx` | 15 kB | Maps sensor and space IDs to buildings and rooms |

## Known data-quality issues (summary)

Details and evidence go in the data-audit notebook and the report.

- The file names say "May to Dec 2024", but the data actually spans 2023–2026.
- Every environmental sensor "froze" at some point (its own clock stopped while
  records kept arriving). Three never produced valid readings; …869 was valid
  29 Feb – 14 Jun 2024 and …326 was valid 7 Sep 2025 – 22 Apr 2026.
- Only 2 of the 5 occupancy spaces can be matched to a room. The location
  sheet's occupancy tab is itself labelled "Incorrect Data".
- No valid environmental sensor is in a room with an occupancy sensor.
- Timestamps cross daylight-saving changes, which need careful handling.

## Processed files (`processed/`)

| File | Contents |
|---|---|
| `occupancy_hourly.csv` | One row per space per hour with readings: mean/max headcount, number of readings, `is_event` flag (peak ≥ 30 people). Timestamps are local time with UTC offset. |
| `env_sensor_health_monthly.csv` | Per environmental sensor and month: records, distinct sensor timestamps, distinct CO₂ values, working/frozen |
| `env_sensor_valid_periods.csv` | Per environmental sensor: period with genuinely changing readings (blank = never) |
| `env_fig3_extract.csv` | CO₂ readings for the two sensors in Figure 3 (30 Apr – 12 May 2024) |
