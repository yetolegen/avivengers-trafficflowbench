import glob, subprocess, sys, time
from pathlib import Path
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
REL = Path('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
REPO = Path('/tmp/tfb')
subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/jacky850/trafficflowbench-public.git', str(REPO)], check=True)
src = [p for p in glob.glob('/kaggle/input/**/state.csv', recursive=True)]
print('state files found', src, flush=True)
st = pd.read_csv(src[0], dtype={'station_id': str, 'link_id': str})
print('rows', len(st), 'months', st.timestamp.str[:7].value_counts().to_dict(), flush=True)
val = st[st.timestamp.str[:7] == '2031-03']
val.to_csv('/kaggle/working/state_validation.csv', index=False)
del st
print(subprocess.run(['python', 'src/task3/score_task3.py', '-h'], cwd=REPO, capture_output=True, text=True).stdout, flush=True)
t = time.time()
r = subprocess.run(['python', 'src/task3/score_task3.py', '--state-submission', '/kaggle/working/state_validation.csv', '--split', 'validation',
                    '--release-root', str(REL), '--output', '/kaggle/working/t3_validation.csv'], cwd=REPO, capture_output=True, text=True)
print('exit', r.returncode, 'sec', round(time.time() - t), flush=True)
print(r.stdout[-6000:]); print(r.stderr[-4000:], flush=True)
out = pd.read_csv('/kaggle/working/t3_validation.csv')
print(out.to_string(index=False))
num = out.select_dtypes('number').columns
print('MEAN'); print(out[num].mean().round(4).to_string(), flush=True)
