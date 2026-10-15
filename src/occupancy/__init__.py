"""
occupancy: forecasting room occupancy in Monash Smart Infrastructure buildings.

MMA3001 Numerical Methods and Machine Learning project (Dataset 2).

Modules:

- ``config``   - file locations, time zone and space names
- ``clean``    - turn 9.1 M raw occupancy events into an hourly table
- ``audit``    - unpack the environmental data and detect frozen sensors
- ``plots``    - Figures 1-3 of the report (data audit)
- ``pipeline`` - rebuild all processed data from the raw files

Planned (later stages): ``features``, ``models``, ``evaluate``.
"""

__version__ = "0.1.0"
