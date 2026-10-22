# Notebooks

Run these in order. Each notebook is the "story" of one stage of the project,
and calls the reusable functions in `src/occupancy/`.

| Notebook | Stage | Status |
|---|---|---|
| `01_data_audit.ipynb` | Explore the raw data and document quality problems; recreates Figures 1–3 | Done |
| `02_baseline_and_timetable.ipynb` | Fixed timetable and average-profile baseline, scored on 2025 | Done |
| `03_ml_model.ipynb` | Day-ahead gradient-boosting forecast: tuning on 2025, comparison with both references, final 2026 test, input importance, forecast export | Done |
| `04_robustness.ipynb` | Robustness checks: per month, walk-forward, seeds, removed inputs, threshold sensitivity, timing | Planned |
