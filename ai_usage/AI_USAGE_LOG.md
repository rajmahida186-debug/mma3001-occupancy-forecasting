# AI usage log

The brief requires all material AI use to be disclosed. This log records it as
the project goes, and feeds the report's AI reflection section.

| Date | Tool | What it was used for | Level of contribution | What I checked / changed myself |
|---|---|---|---|---|
| 2026-10-08 | Claude (Anthropic) | Explored the dataset, identified data-quality issues, suggested the project question | Medium: analysis and suggestions | Chose the project direction |
| 2026-10-08 | Claude (Anthropic) | Set up this repository's folder structure, README files and licence | High: generated the files | Reviewed the structure against the brief |
| 2026-10-08 | Claude (Anthropic) | Produced data-audit figures and a Word report template (section structure from the brief/rubric, evidence and sources per section) | Medium: structure, figures and evidence notes; report prose to be written by me | Checked the template against the brief; checked cited sources |
| 2026-10-08 | Claude (Anthropic) | Drafted report Section 1 (Engineering problem) from my project direction and the rubric | High: wrote the first draft | [To do: review, edit into my own words, check claims against the cited sources] |
| 2026-10-09 | Claude (Anthropic) | Built an Excel workbook of hourly occupancy (5 spaces) with AVERAGEIFS profile tables, so I can make the Figure 1 charts myself | Medium: data aggregation and workbook | Made the charts myself in Excel |
| 2026-10-15 | Claude (Anthropic) | Wrote the cleaning pipeline (src/occupancy: clean, audit, plots, pipeline), 21 tests, and data-audit notebook 01 | High: wrote the code | [To do: read the plain-English summary, check I can explain each step] |
| 2026-10-15 | Claude (Anthropic) | Wrote Stage 2: ProfileBaseline, FixedTimetable, shared scoring (MAE/RMSE, wasted/discomfort hours), chronological split, 8 tests, notebook 02. Proposed the 1-person occupied threshold from night-time data | High: wrote the code; threshold choice based on data shown in notebook 02 | [To do: decide whether I agree with the 1-person threshold and the 7am–7pm timetable assumption] |
| 2026-10-22 | Claude (Anthropic) | Compared 7–7, 8–8 and 8–9 timetable windows on 2025 data and found Monash's Student Timetable Procedure (teaching 8 am–9 pm Mon–Fri) | Medium | **My decision:** I questioned Claude's assumed 7am–7pm timetable because classes run 8am–8pm; the data confirmed 8–8 performs best, so the default was changed |
