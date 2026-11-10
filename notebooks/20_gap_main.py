import glob, sys, time, numpy as np, pandas as pd
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
TP = glob.glob('/kaggle/input/**/t2lib.py', recursive=True)[0]
sys.path.insert(0, str(Path(TP).parent))
import t2lib as T
REL = Path(T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public'))
FAM = T.load_family_map(REL)
PANELS = ['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
SPLITS = ['train', 'validation', 'private']
COLS = ['timestamp', 'station_id', 'link_id', 'speed_kmh', 'is_score_eligible']

def ts_str(s):
    return pd.to_datetime(s, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def layer_files(panel, split, layer):
    d = REL / 'corridors' / panel / split / layer
    return sorted(d.rglob('*.parquet')) if d.exists() else []

def read_layer(files, days):
    out = []
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        d['timestamp'] = ts_str(d['timestamp'])
        d = d[d.timestamp.str[:10].isin(days)]
        if len(d):
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else None

print('==================== A: WHERE DOES THE OFFICIAL WINDOW HISTORY COME FROM ====================', flush=True)
A = []
for panel in PANELS:
    ts = time.time()
    for split in SPLITS:
        wh = pd.read_parquet(T.task2_path(REL, panel, split, 'window_history.parquet'))
        if panel == PANELS[0]:
            print(split, 'history columns', list(wh.columns), flush=True)
        wh['timestamp'] = ts_str(wh['timestamp'])
        keys = [k for k in ('timestamp', 'station_id', 'link_id') if k in wh.columns]
        days = set(wh.timestamp.str[:10])
        r = dict(panel=panel, split=split, windows=wh.window_id.nunique(), rows=len(wh), hist_speed_nan=round(float(wh.speed_kmh.isna().mean()), 4))
        for layer in ('mainline_states', 'mainline_states_masked'):
            files = layer_files(panel, split, layer)
            if not files:
                continue
            L = read_layer(files, days)
            if L is None:
                continue
            L = L.drop_duplicates(keys)
            m = wh[keys + ['speed_kmh']].merge(L[keys + ['speed_kmh']], on=keys, how='left', suffixes=('', '_L'))
            has = m.speed_kmh.notna()
            tag = 'unmasked' if layer == 'mainline_states' else 'masked'
            r[f'match_{tag}'] = round(float(np.isclose(m.speed_kmh[has], m.speed_kmh_L[has], atol=0.01).mean()), 4)
            r[f'{tag}_nan_where_hist_has'] = round(float(m.speed_kmh_L[has].isna().mean()), 4)
        A.append(r)
    print(panel, 'sec', round(time.time() - ts), flush=True)
A = pd.DataFrame(A)
print(A.to_string(index=False))
print(A.groupby('split').mean(numeric_only=True).round(4).to_string(), flush=True)

print('==================== B: DO JAMS SIT IN DIFFERENT PLACES IN THE HIDDEN MONTHS ====================', flush=True)
B = []
for panel in PANELS:
    ts = time.time()
    ctx = T.PanelContext(REL, panel, FAM.get(panel), T.collect_target_links(REL, panel, FAM), 0)
    tl = list(np.asarray(ctx.link_ids)[ctx.target_mask])
    thr = pd.Series(ctx.thr_dict)
    prof, hprof = {}, {}
    for split in SPLITS:
        for f in layer_files(panel, split, 'mainline_states_masked'):
            d = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
            d = d[d.link_id.astype(str).isin(tl) & d.speed_kmh.notna() & (d.is_score_eligible.astype(float) > 0)]
            if not len(d):
                continue
            t = d.timestamp.astype(str) if d.timestamp.dtype == object else ts_str(d.timestamp)
            mo = f'{split}:{t.iloc[0][:7]}'
            d = d.assign(q=(d.speed_kmh <= d.link_id.astype(str).map(thr).fillna(T.OFFICIAL_UNKNOWN_LINK_THRESHOLD)).astype(float), hr=t.str[11:13].astype(int).to_numpy())
            g = d.groupby(d.link_id.astype(str)).q.agg(['sum', 'count'])
            prof[mo] = prof[mo].add(g, fill_value=0) if mo in prof else g
            gh = d.groupby('hr').q.agg(['sum', 'count'])
            hprof[mo] = hprof[mo].add(gh, fill_value=0) if mo in hprof else gh
    share = pd.DataFrame({k: v['sum'] / v['count'] for k, v in prof.items()}).reindex(tl).fillna(0)
    hshare = pd.DataFrame({k: v['sum'] / v['count'] for k, v in hprof.items()}).reindex(range(24)).fillna(0)
    trm = [c for c in share.columns if c.startswith('train:')]
    for c in share.columns:
        ref = [x for x in trm if x != c]
        rl = share[ref].mean(1); rh = hshare[ref].mean(1)
        top_ref = set(rl.sort_values(ascending=False).index[:5]); top_c = set(share[c].sort_values(ascending=False).index[:5])
        B.append(dict(panel=panel, month=c, kind=c.split(':')[0], jam_share=round(float(share[c].mean()), 4),
                      link_corr=round(float(np.corrcoef(share[c], rl)[0, 1]), 3), hour_corr=round(float(np.corrcoef(hshare[c], rh)[0, 1]), 3),
                      top5_overlap=len(top_ref & top_c), peak_hour=int(hshare[c].idxmax())))
    print(panel, 'months', len(share.columns), 'sec', round(time.time() - ts), flush=True)
B = pd.DataFrame(B)
print(B.to_string(index=False))
print('MEAN BY KIND (train months are compared with the other train months)')
print(B.groupby('kind')[['jam_share', 'link_corr', 'hour_corr', 'top5_overlap']].mean().round(3).to_string(), flush=True)
