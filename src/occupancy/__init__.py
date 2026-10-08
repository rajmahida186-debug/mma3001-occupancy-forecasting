"""
occupancy: forecasting room occupancy in Monash Smart Infrastructure buildings.

MMA3001 Numerical Methods and Machine Learning project (Dataset 2).

Planned modules (added as the project develops):

- ``load``     - read the raw Monash sensor exports
- ``clean``    - audit and clean the data (frozen sensors, gaps, daylight saving)
- ``features`` - build model inputs (hour, weekday, semester, recent occupancy)
- ``models``   - baseline profile model and machine-learning model
- ``evaluate`` - validation metrics and model comparison
"""

__version__ = "0.1.0"
