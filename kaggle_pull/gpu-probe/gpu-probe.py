# Which CatBoost/XGBoost GPU setup works on this Kaggle GPU? Each try runs in its own process (CUDA state can't be reset in-process).
import subprocess, sys
sh = lambda c: print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-1500:], flush=True)
sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi | head -4; python -V; pip list 2>/dev/null | grep -iE 'catboost|xgboost|lightgbm'")

TEST = r'''
import numpy as np, sys
X = np.random.rand(200000, 30).astype("float32"); y = X[:, 0] * 3 + np.random.rand(200000)
which = sys.argv[1]
if which == "catboost":
    import catboost; from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, depth=8, task_type="GPU", verbose=0).fit(X, y)
    print("CATBOOST GPU OK", catboost.__version__, m.get_best_score())
else:
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=200, max_depth=8, device="cuda", tree_method="hist").fit(X, y)
    print("XGBOOST GPU OK", xgb.__version__, m.get_booster().attributes())
'''
open("/tmp/t.py", "w").write(TEST)
def attempt(label, which):
    r = subprocess.run([sys.executable, "/tmp/t.py", which], capture_output=True, text=True)
    print(f"--- {label}:", (r.stdout + r.stderr).strip().splitlines()[-1:] or "no output", flush=True)

attempt("preinstalled catboost", "catboost")
attempt("preinstalled xgboost", "xgboost")
for v in ["1.2.8", "1.2.7"]:
    sh(f"pip install -q catboost=={v} 2>&1 | tail -1")
    attempt(f"catboost {v}", "catboost")
