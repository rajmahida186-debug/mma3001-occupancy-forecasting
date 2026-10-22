"""
occupancy: forecasting room occupancy in Monash Smart Infrastructure buildings.

MMA3001 Numerical Methods and Machine Learning project (Dataset 2).

Modules:

- ``config``   - file locations, time zone and space names
- ``clean``    - turn 9.1 M raw occupancy events into an hourly table
- ``audit``    - unpack the environmental data and detect frozen sensors
- ``features`` - day-ahead model inputs (calendar, semester, recent occupancy)
- ``models``   - profile baseline, gradient-boosting forecaster, fixed timetable
- ``evaluate`` - chronological split, error metrics and HVAC outcomes
- ``robustness`` - bootstrap, walk-forward, threshold curve and other checks
- ``plots``    - report figures (audit Figures 1-3, results Figures 4-8)
- ``pipeline`` - rebuild all processed data from the raw files
"""

__version__ = "0.4.0"
