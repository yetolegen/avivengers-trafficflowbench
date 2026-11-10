import time
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
EXP = ['D12_I5_N', 'D7_I210_W', 'D7_I405_S']
PARAMS = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'num_threads': 4, 'verbose': -1, 'seed': 2026}
ROUNDS = 400
FE = ['speed_kmh', 'flow_vph', 'span_min', 'f_u1', 'f_d1', 'f_u2', 'f_d2', 's_u1', 's_d1', 's_u2', 's_d2', 'nb_flow', 'nb_speed',
      'flow_minus_nb', 'speed_minus_nb', 'hour', 'dow', 'lanes', 'free', 'regime']

def station_order(panel):
    f = sorted((REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet'))[0]
    d = pd.read_parquet(f, columns=['station_id', 'milepost']).drop_duplicates('station_id')
    d['station_id'] = d['station_id'].astype(str)
    return d.sort_values('milepost')['station_id'].tolist()

def free_map(panel):
    topo = pd.read_csv(REL / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv', dtype={'link_id': str})
    return topo.drop_duplicates('link_id').set_index('link_id')['free_speed_kmh']

def table_for(panel, dates, aggs):
    days = masked_days(panel, 'train')
    preds, _ = predict_split(panel, 'train', OLD, aggs, only_dates=set(dates))
    truth_files = {f.name: f for f in (REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')}
    order = station_order(panel); pos = {s: i for i, s in enumerate(order)}; nS = len(order)
    lanes_map = s1.station_lanes(REL / 'corridors' / panel); free = free_map(panel)
    preds = preds.assign(day=preds['ts'].dt.normalize().dt.tz_localize(None))
    out = []
    for date in dates:
        p = preds[preds['day'] == date].copy()
        if p.empty:
            continue
        regime, f = days[date]
        df = read_day(f); df = df[df['pct_observed'] >= 75]
        F = df.pivot_table(index='timestamp', columns='station_id', values='flow_vph', aggfunc='mean').reindex(columns=order)
        S = df.pivot_table(index='timestamp', columns='station_id', values='speed_kmh', aggfunc='mean').reindex(columns=order)
        Fv, Sv = F.to_numpy(), S.to_numpy()
        ti = F.index.get_indexer(p['ts']); si = p['station_id'].map(pos).fillna(-100).astype(int).to_numpy()
        def take(M, k):
            j = si + k
            okb = (j >= 0) & (j < nS) & (ti >= 0)
            res = np.full(len(p), np.nan)
            res[okb] = M[ti[okb], j[okb]]
            return res
        for k, name in ((-1, 'u1'), (1, 'd1'), (-2, 'u2'), (2, 'd2')):
            p['f_' + name] = take(Fv, k); p['s_' + name] = take(Sv, k)
        near_f = np.nanmean(np.vstack([p['f_u1'], p['f_d1']]), axis=0)
        far_f = np.nanmean(np.vstack([p['f_u2'], p['f_d2']]), axis=0)
        near_s = np.nanmean(np.vstack([p['s_u1'], p['s_d1']]), axis=0)
        far_s = np.nanmean(np.vstack([p['s_u2'], p['s_d2']]), axis=0)
        p['nb_flow'] = np.where(np.isnan(near_f), far_f, near_f); p['nb_speed'] = np.where(np.isnan(near_s), far_s, near_s)
        p['flow_minus_nb'] = p['flow_vph'] - p['nb_flow']; p['speed_minus_nb'] = p['speed_kmh'] - p['nb_speed']
        t = read_day(truth_files[f.name])[KEYS + VALUES].rename(columns={'speed_kmh': 's_true', 'flow_vph': 'f_true'})
        m = p.drop(columns=['timestamp']).rename(columns={'ts': 'timestamp'}).merge(t, on=KEYS, how='left')
        m['hour'] = m['timestamp'].dt.hour; m['dow'] = m['timestamp'].dt.dayofweek
        m['lanes'] = s1.lane_vector(m['station_id'], m['link_id'], lanes_map)
        m['free'] = m['link_id'].map(free).to_numpy(); m['regime'] = REGIMES.index(regime)
        out.append(m)
    return pd.concat(out, ignore_index=True)

def s_state(m, speed, flow):
    rows = []
    for r in range(3):
        k = (m['regime'] == r) & m['s_true'].notna() & m['f_true'].notna()
        rs = np.sqrt(np.mean((speed[k] - m['s_true'][k]) ** 2))
        rf = np.sqrt(np.mean(((flow[k] - m['f_true'][k]) / m['lanes'][k]) ** 2))
        rows.append(0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rf / 600))
    return float(np.mean(rows))

def score_sub(panel, sub, names):
    original = s1.masked_files
    s1.masked_files = lambda rel, p, split, regime: [f for f in original(rel, p, split, regime) if f.name in names]
    try:
        return pd.DataFrame(s1.score_panel(panel, REL, 'train', sub))
    finally:
        s1.masked_files = original

tables, evaldates = {}, {}
for panel in EXP:
    ts0 = time.time()
    aggs = daily_profiles(panel)
    days = masked_days(panel, 'train')
    dates = sorted(sample_days(days)); cut = int(len(dates) * 0.7)
    tables[panel] = table_for(panel, dates, aggs); evaldates[panel] = dates[cut:]
    tables[panel]['train_part'] = tables[panel]['timestamp'].dt.tz_localize(None).dt.normalize().isin(dates[:cut])
    tables[panel]['panel_code'] = EXP.index(panel)
    print(panel, 'days', len(dates), 'rows', len(tables[panel]), 'sec', round(time.time() - ts0), flush=True)
    del aggs
T_ALL = pd.concat(tables.values(), ignore_index=True)
tr = T_ALL[T_ALL['train_part'] & T_ALL['s_true'].notna() & T_ALL['f_true'].notna()]
ev = T_ALL[~T_ALL['train_part']]
FEATS = FE + ['panel_code']
t1 = time.time()
mf = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr['f_true'] - tr['flow_vph']), ROUNDS)
ms = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr['s_true'] - tr['speed_kmh']), ROUNDS)
print('trained on', len(tr), 'rows in', round(time.time() - t1), 'sec', flush=True)
ev = ev.assign(flow_fix=np.clip(ev['flow_vph'] + mf.predict(ev[FEATS]), 0, None), speed_fix=np.clip(ev['speed_kmh'] + ms.predict(ev[FEATS]), 1, None))
rows = []
for panel in EXP + ['ALL']:
    e = ev if panel == 'ALL' else ev[ev['panel'] == panel]
    rows.append({'road': panel, 'cells': len(e), 'S_state_now': s_state(e, e['speed_kmh'], e['flow_vph']),
                 'S_state_flow_only': s_state(e, e['speed_kmh'], e['flow_fix']), 'S_state_speed_only': s_state(e, e['speed_fix'], e['flow_vph']),
                 'S_state_both': s_state(e, e['speed_fix'], e['flow_fix'])})
print(pd.DataFrame(rows).round(4).to_string(), flush=True)
rm = lambda a, b, l=1: float(np.sqrt(np.mean(((a - b) / l) ** 2)))
k = ev['s_true'].notna() & ev['f_true'].notna(); e = ev[k]
print('RMSE speed now', round(rm(e['speed_kmh'], e['s_true']), 3), 'fixed', round(rm(e['speed_fix'], e['s_true']), 3),
      '| RMSE flow/lane now', round(rm(e['flow_vph'], e['f_true'], e['lanes']), 2), 'fixed', round(rm(e['flow_fix'], e['f_true'], e['lanes']), 2))
print('flow model top features:', dict(sorted(zip(FEATS, mf.feature_importance('gain')), key=lambda x: -x[1])[:8]))
print('speed model top features:', dict(sorted(zip(FEATS, ms.feature_importance('gain')), key=lambda x: -x[1])[:8]))
pn = EXP[0]
sub0 = ev[ev['panel'] == pn][['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']]
sub1 = ev[ev['panel'] == pn][['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].assign(speed_kmh=ev[ev['panel'] == pn]['speed_fix'], flow_vph=ev[ev['panel'] == pn]['flow_fix'])
names = {f.name for d, (_, f) in masked_days(pn, 'train').items() if d in set(evaldates[pn])}
for label, sub in (('official, now', sub0), ('official, fixed', sub1)):
    det = score_sub(pn, sub, names)
    print(label, pn, 'S_state', round(float(det[det['level'] == 'panel']['S_state'].iloc[0]), 4), 'missing', int(det['n_missing_predictions'].sum()), flush=True)
