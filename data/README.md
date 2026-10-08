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
| `processed/` | Small cleaned tables produced by the code (e.g. hourly occupancy) | ✅ Yes |

## Raw files expected in `raw/`

| File | Size | Description |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 1.6 GB | ~9.1 M occupancy events (status and headcount), 5 spaces, Nov 2023 – Apr 2026 |
| `5EnvSensor_MayToDec2024_180kRows.csv` | 350 MB | ~179 k environmental readings (CO₂, temperature, humidity, PM, etc.) from 5 sensors, stored as JSON |
| `Sensor ID and Locations.xlsx` | 15 kB | Maps sensor and space IDs to buildings and rooms |

## Known data-quality issues (summary)

Details and evidence go in the data-audit notebook and the report.

- The file names say "May to Dec 2024", but the data actually spans 2023–2026.
- 3 of the 5 environmental sensors are frozen: they repeat the same reading
  thousands of times.
- Only 2 of the 5 occupancy spaces can be matched to a room. The location
  sheet's occupancy tab is itself labelled "Incorrect Data".
- No room has both occupancy and environmental sensors with overlapping data.
- Timestamps cross daylight-saving changes, which need careful handling.
