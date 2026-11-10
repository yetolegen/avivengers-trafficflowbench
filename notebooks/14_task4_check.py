import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
REL = Path('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jacky850/trafficflowbench-public.git', '/tmp/tfb2'], check=True, capture_output=True)
sys.path.insert(0, '/tmp/tfb2/src')
from task4 import build_task4_odme_artifacts as B4
panels = [p['corridor_id'] for p in json.load(open('/tmp/tfb2/config/corridors.json'))['panels']]
SPLIT = 'validation'
rows = []
for panel in panels:
    ts = time.time()
    path_ids, link_ids, A, paths = B4.load_operator(REL / 'corridors' / panel / 'network')
    base = B4.prior_values(REL, panel, SPLIT, path_ids)
    tr = B4.released_counts(REL, panel, 'train')
    if tr is None:
        tr = B4.counts_for_split(REL / 'corridors' / panel, 'train')
    sp = B4.released_counts(REL, panel, SPLIT)
    tc = tr.set_index('link_id').reindex(link_ids).fillna(0.0)['count'].to_numpy(float)
    sc = sp.set_index('link_id').reindex(link_ids).fillna(0.0)['count'].to_numpy(float)
    seen = set(sp.link_id)
    meas = np.array([l in seen for l in link_ids])
    Am = A[meas]; c = sc[meas]
    def slink(f):
        return max(0.0, 1 - np.abs(Am @ f - c).sum() / max(c.sum(), 1e-9))
    f_tr = B4.solve(Am, tc[meas], base)
    f_sp = B4.solve(Am, c, base)
    rows.append(dict(panel=panel, paths=len(path_ids), links_counted=int(meas.sum()), S_link_prior=slink(base), S_link_trainfit=slink(f_tr), S_link_monthfit=slink(f_sp),
                     counts_rel_diff=np.abs(tc[meas] - c).sum() / c.sum(), counts_corr=np.corrcoef(tc[meas], c)[0, 1],
                     dev_trainfit=np.abs(f_tr - base).sum() / max(base.sum(), 1e-9), dev_monthfit=np.abs(f_sp - base).sum() / max(base.sum(), 1e-9),
                     dist_train_vs_month=np.abs(f_tr - f_sp).sum() / max(f_sp.sum(), 1e-9), sec=round(time.time() - ts)))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print(df.round(4).to_string())
print(df.mean(numeric_only=True).round(4).to_string())
