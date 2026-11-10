import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PULL = ROOT / "kaggle_pull"
USER = "yetolegen"
STEPS = ["task-1-v2", "task-2-onset", "task-4", "sub-final-v10"]
PREREQ = ["task-2-final", "task-1"]
POLL_SECONDS = 120
MAX_HOURS = 12


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def status(slug):
    code, out = kaggle("kernels", "status", f"{USER}/{slug}")
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out:
            return word
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser(description="Rebuild the final submission (sub-final v10) by running the Kaggle notebooks in order.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "final_output"))
    a = ap.parse_args()

    for slug in PREREQ:
        s = "skipped" if a.dry_run else status(slug)
        print(f"prerequisite kernel {USER}/{slug}: {s}")
        if not a.dry_run and s != "COMPLETE":
            sys.exit(f"{slug} must be completed first: task-2-onset reads its output")

    for slug in STEPS:
        folder = PULL / slug
        print(f"push {folder}")
        if a.dry_run:
            continue
        code, out = kaggle("kernels", "push", "-p", str(folder))
        print(out.strip())
        if code != 0:
            sys.exit(f"push failed for {slug}")
        t0 = time.time()
        while True:
            time.sleep(POLL_SECONDS)
            s = status(slug)
            print(f"{slug}: {s} after {round((time.time() - t0) / 60)} min", flush=True)
            if s == "COMPLETE":
                break
            if s in ("ERROR", "CANCEL") or time.time() - t0 > MAX_HOURS * 3600:
                sys.exit(f"{slug} did not finish: {s}")

    if a.dry_run:
        print(f"download {USER}/sub-final-v10 output to {a.out}")
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    code, out = kaggle("kernels", "output", f"{USER}/sub-final-v10", "-p", a.out)
    print(out.strip())
    if not (Path(a.out) / "submission.csv").exists():
        sys.exit("submission.csv was not downloaded")
    print("done:", Path(a.out) / "submission.csv")


if __name__ == "__main__":
    main()
