import glob, sys, time, numpy as np, pandas as pd
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
TP = glob.glob('/kaggle/input/**/t2lib.py', recursive=True)[0]
sys.path.insert(0, str(Path(TP).parent))
import t2lib as T
REL = Path(T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public'))
FAM = T.load_family_map(REL)
PANELS = ['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
COLS = ['timestamp', 'station_id', 'link_id', 'speed_kmh', 'is_score_eligible']
DEFS = ['any_noisy', 'mean_noisy', 'mean_smooth3', 'mean_smooth5']

def day_of(f):
    return Path(f).stem[-10:].replace('_', '-')

def load(files):
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True); d['station_id'] = d.station_id.astype(str); d['link_id'] = d.link_id.astype(str)
    d['ok'] = d.is_score_eligible.astype(float).fillna(0).gt(0) & d.speed_kmh.notna()
    return d

def link_labels(d, thr, tl):
    d = d[d.link_id.isin(tl) & d.ok].copy()
    d['thr'] = d.link_id.map(thr)
    key = [d.link_id, d.timestamp]
    out = pd.DataFrame({'any_noisy': (d.speed_kmh <= d.thr).groupby(key).any(), 'mean_speed': d.speed_kmh.groupby(key).mean()}).reset_index()
    out = out.sort_values(['link_id', 'timestamp'])
    out['thr'] = out.link_id.map(thr)
    sm3 = out.groupby('link_id').mean_speed.transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    sm5 = out.groupby('link_id').mean_speed.transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
    out['mean_noisy'] = out.mean_speed <= out.thr
    out['mean_smooth3'] = sm3 <= out.thr
    out['mean_smooth5'] = sm5 <= out.thr
    return out

NOISE, LAB, PERS = [], [], []
for panel in PANELS:
    ts = time.time()
    ctx = T.PanelContext(REL, panel, FAM.get(panel), T.collect_target_links(REL, panel, FAM), 0)
    thr = dict(ctx.thr_dict); tl = set(np.asarray(ctx.link_ids)[ctx.target_mask])
    files = sorted((REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet'))
    d = load(files[::9]).sort_values(['station_id', 'timestamp'])
    day = d.timestamp.dt.strftime('%Y-%m-%d')
    s = d.speed_kmh.where(d.ok)
    grp = [d.station_id, day]
    d1 = s - s.groupby(grp).shift(1); d2 = s - s.groupby(grp).shift(2); d1p = d1.groupby(grp).shift(1)
    v1, v2 = np.nanvar(d1), np.nanvar(d2)
    ac = pd.concat([d1, d1p], axis=1).dropna().corr().iloc[0, 1]
    nst = d[d.link_id.isin(tl)].groupby('link_id').station_id.nunique()
    NOISE.append(dict(panel=panel, var_step1=round(v1, 2), var_step2=round(v2, 2), noise_sigma_kmh=round(np.sqrt(max((4 * v1 - v2) / 6, 0)), 2),
                      step_autocorr=round(ac, 3), target_links=len(nst), stations_per_link=round(nst.mean(), 2), share_multi_station=round((nst > 1).mean(), 3)))
    lab = link_labels(d, thr, tl)
    r = dict(panel=panel)
    for k in DEFS:
        r['jam_' + k] = round(float(lab[k].mean()), 4)
        r['flip_' + k] = round(float(lab.groupby('link_id')[k].apply(lambda v: (v.astype(int).diff().abs() > 0).mean()).mean()), 4)
    LAB.append(r)
    idx = T.read_window_index(REL, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(REL, panel, 'train'), idx, panel)
    tpl = tpl.assign(timestamp=pd.to_datetime(tpl.timestamp, utc=True), link_id=tpl.link_id.astype(str), window_id=tpl.window_id.astype(str))
    cond = idx.assign(window_id=idx.window_id.astype(str)).set_index('window_id').condition
    wdays = set(tpl.timestamp.dt.strftime('%Y-%m-%d'))
    lw = link_labels(load([f for f in files if day_of(f) in wdays]), thr, tl)
    wh = pd.read_parquet(T.task2_path(REL, panel, 'train', 'window_history.parquet'))
    wh = wh.assign(timestamp=pd.to_datetime(wh.timestamp, utc=True), link_id=wh.link_id.astype(str), window_id=wh.window_id.astype(str))
    last = wh[wh.timestamp == wh.groupby('window_id').timestamp.transform('max')]
    last = last[last.is_score_eligible.astype(float).fillna(0).gt(0) & last.speed_kmh.notna()]
    pq = (last.speed_kmh <= last.link_id.map(thr)).groupby([last.window_id, last.link_id]).any().rename('pers').reset_index()
    m = tpl.merge(lw, on=['link_id', 'timestamp'], how='left').merge(pq, on=['window_id', 'link_id'], how='left')
    m['pers'] = m.pers.fillna(False).astype(bool)
    for k in DEFS:
        mk = m[m[k].notna()]
        t_ = mk[k].astype(bool); p_ = mk.pers
        g = pd.DataFrame({'w': mk.window_id, 'i': t_ & p_, 'u': t_ | p_, 'n': t_}).groupby('w').sum()
        res = pd.Series(np.where(g.u > 0, g.i / g.u.clip(lower=1), 1.0), index=g.index)
        c = cond.reindex(res.index)
        PERS.append(dict(panel=panel, label_rule=k, onset=float(res[c == 'queue_onset'].mean()), ongoing=float(res[c == 'queue_ongoing'].mean()),
                         cells_onset=float(g.n[c == 'queue_onset'].mean()), cells_ongoing=float(g.n[c == 'queue_ongoing'].mean())))
    print(panel, 'sec', round(time.time() - ts), flush=True)
print('NOISE'); print(pd.DataFrame(NOISE).to_string(index=False))
print('LABEL RULES (jam share and how often the label flips between 5-min steps)'); print(pd.DataFrame(LAB).to_string(index=False))
P = pd.DataFrame(PERS); P['persistence_score'] = (P.onset + P.ongoing) / 2
print('PERSISTENCE BASELINE ON SHIPPED TRAIN WINDOWS (official validation value 0.2518)'); print(P.round(4).to_string(index=False))
print(P.groupby('label_rule')[['onset', 'ongoing', 'persistence_score', 'cells_onset', 'cells_ongoing']].mean().round(4).to_string(), flush=True)
