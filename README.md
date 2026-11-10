# Avivengers - 2026 IEEE Big Data Traffic Flow Bench

Kaggle solution (best public LB 0.85800).

- `notebooks/` - numbered experiment scripts and notebooks
- `kaggle_pull/` - Kaggle notebooks used for the final submission
- `backup/` - earlier versions
- `results/` - local evaluation tables
- `Avivengers_handoff.docx` - handoff notes

Final pipeline: Task 1 CatBoost (task-1-v2), Task 2 onset model (task-2-onset), Task 3 density A, Task 4 month-fit, merged in `sub-final` v10.

## Reproduce the final submission (sub-final v10)

Run on Kaggle with the competition data attached (internet on, GPU T4 for Task 1). Python 3.12, lightgbm, catboost, pandas, numpy, pyarrow.

1. `kaggle_pull/task-1-v2/task-1-v2.ipynb` builds `state.csv` (Task 1).
2. `kaggle_pull/task-2-onset/task-2-onset.ipynb` builds `queue.csv` (Task 2). It needs `notebooks/t2lib_clean.py` as an input file.
3. Task 4 notebook (`notebooks/task4_v2.ipynb`) builds `odme.csv`.
4. `kaggle_pull/sub-final/sub-final.ipynb` applies the Task 3 density step (`notebooks/31_t3_density_cell.py`, variant A), merges the three files and runs the format checks. Output: `submission.csv`.

The report is in `report/Avivengers_TrafficFlowBench_Report.pdf`.
