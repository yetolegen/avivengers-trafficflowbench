print('==================== IDEA 2: DENSITY FIRST ====================', flush=True)
def safe_k(f, s):
    return f / np.clip(s, 1, None)
trk = tr.assign(k_now=safe_k(tr['flow_vph'], tr['speed_kmh']), k_true=safe_k(tr['f_true'], tr['s_true']))
t2 = time.time()
mk = lgb.train(PARAMS, lgb.Dataset(trk[FEATS], trk['k_true'] - trk['k_now']), ROUNDS)
print('density model trained in', round(time.time() - t2), 'sec', flush=True)
ev2 = ev.assign(k_now=safe_k(ev['flow_vph'], ev['speed_kmh']), k_true=safe_k(ev['f_true'], ev['s_true']))
ev2['k_fix'] = np.clip(ev2['k_now'] + mk.predict(ev2[FEATS]), 0, None)
ev2['flow_B'] = ev2['k_fix'] * ev2['speed_fix']
ev2['flow_H'] = 0.5 * (ev2['flow_fix'] + ev2['flow_B'])
rows = []
for panel in EXP + ['ALL']:
    e = ev2 if panel == 'ALL' else ev2[ev2['panel'] == panel]
    rows.append({'road': panel, 'now': s_state(e, e['speed_kmh'], e['flow_vph']), 'A_current_fix': s_state(e, e['speed_fix'], e['flow_fix']),
                 'B_density_first': s_state(e, e['speed_fix'], e['flow_B']), 'H_average_of_A_and_B': s_state(e, e['speed_fix'], e['flow_H'])})
print(pd.DataFrame(rows).round(4).to_string(), flush=True)
k = ev2['s_true'].notna() & ev2['f_true'].notna() & (ev2['k_true'] > 1)
e = ev2[k].copy()
e['k_A'] = safe_k(e['flow_fix'], e['speed_fix']); e['k_B'] = e['k_fix']; e['k_H'] = safe_k(e['flow_H'], e['speed_fix'])
print('DENSITY ERROR (median relative, %):', {n: round(float(np.median(np.abs(e[c] - e['k_true']) / e['k_true'])) * 100, 2) for n, c in (('now', 'k_now'), ('A', 'k_A'), ('B', 'k_B'), ('H', 'k_H'))})
e = e.sort_values(['panel', 'station_id', 'link_id', 'timestamp'])
grp = e.groupby(['panel', 'station_id', 'link_id'], sort=False)
gap = (e['timestamp'] - grp['timestamp'].shift(1)).dt.total_seconds() / 60
pair = gap == 5
print('consecutive 5-min pairs of blanked cells:', int(pair.sum()))
den = float(np.mean(np.abs(e['k_true'][pair])))
res = {}
for n, c in (('now', 'k_now'), ('A', 'k_A'), ('B', 'k_B'), ('H', 'k_H')):
    d_est = e[c] - grp[c].shift(1); d_true = e['k_true'] - grp['k_true'].shift(1)
    res[n] = round(float(np.mean(np.abs(d_est[pair] - d_true[pair]))) / den * 100, 3)
print('STEP-TO-STEP DENSITY CHANGE ERROR (% of mean density; smaller is better):', res, flush=True)
print('==================== IDEA 3: RAMP FILES ====================', flush=True)
pn = EXP[0]
for name in ('ramp_attachment_map.csv', 'ramp_station_map.csv', 'links.csv'):
    f = REL / 'corridors' / pn / 'network' / name
    if f.exists():
        d = pd.read_csv(f)
        print('NETWORK FILE', name, d.shape, list(d.columns)); print(d.head(4).to_string())
rf = sorted((REL / 'corridors' / pn / 'train' / 'ramp_states').rglob('*.parquet'))
print('ramp files', len(rf))
if rf:
    d = pd.read_parquet(rf[0])
    print('RAMP STATES', d.shape, list(d.columns)); print(d.head(6).to_string())
    print(d.describe().round(2).to_string())
