# Avivengers - 2026 IEEE Big Data Traffic Flow Bench

Kaggle solution (best public LB 0.85800).

- `notebooks/` - numbered experiment scripts and notebooks
- `kaggle_pull/` - Kaggle notebooks used for the final submission
- `backup/` - earlier versions
- `results/` - local evaluation tables
- `Avivengers_handoff.docx` - handoff notes

Final pipeline: Task 1 CatBoost (task-1-v2), Task 2 onset model (task-2-onset), Task 3 density A, Task 4 month-fit, merged in `sub-final` v10.

## Reproduce the final submission (sub-final v10)

Entry script: `python run_all.py` (`--dry-run` only prints the steps).

Environment: Python 3.12 with the `kaggle` CLI (2.0.0 or newer) and a Kaggle API token in `~/.kaggle`. The heavy work runs on Kaggle (competition data attached, internet on, GPU T4 for Task 1). Inside the notebooks: lightgbm, catboost, pandas, numpy, pyarrow.

What it does, in order (each folder in `kaggle_pull/` holds the notebook and its `kernel-metadata.json`):
1. `task-1-v2`: Task 1 state file (`state.csv`).
2. `task-2-onset`: Task 2 queue file (`queue.csv`). It reads `t2lib.py` from the output of the earlier kernels `task-2-final` and `task-1`, which must already be completed on the account (the script checks this).
3. `task-4`: Task 4 file (`odme.csv`).
4. `sub-final-v10`: applies the Task 3 density step (variant A, same code as `notebooks/31_t3_density_cell.py`), merges the three files with the organizers' `merge_submissions.py`, checks the format and writes `submission.csv`, which the script downloads to `final_output/`.

Limits: the kernel ids point to the team account `yetolegen`, so another account has to change `USER` in `run_all.py` and the `id` and `kernel_sources` in each `kernel-metadata.json`. Task 1 takes several hours on the GPU. The run was not repeated end to end after the competition closed.

The report is in `report/Avivengers_TrafficFlowBench_Report.pdf`.
