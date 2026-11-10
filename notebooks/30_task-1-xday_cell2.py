import time
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')
PARAMS = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'num_threads': 4, 'verbose': -1, 'seed': 2026}
ROUNDS = 400
CFG = dict(OLD, cross_day=True)  # gaps at day edges also use the previous/next day
FE = ['speed_kmh', 'flow_vph', 'span_min', 'f_u1', 'f_d1', 'f_u2', 'f_d2', 's_u1', 's_d1', 's_u2', 's_d2', 'nb_flow', 'nb_speed',
      'flow_minus_nb', 'speed_minus_nb', 'hour', 'dow', 'lanes', 'free', 'regime']
GAP_FE = [f'g_{a}_{k}' for k in 'sf' for a in ('dp', 'dn', 'pv', 'nv')] + \
         [f'{a}_{k}_{e}' for k in 'sf' for a in ('nbd', 'est') for e in 'pn']
FEATS_OLD = FE + ['panel_code']
FEATS = FEATS_OLD + GAP_FE

def station_order(panel):
    f = sorted((REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet'))[0]
    d = pd.read_parquet(f, columns=['station_id', 'milepost']).drop_duplicates('station_id')
    d['station_id'] = d['station_id'].astype(str)
    return d.sort_values('milepost')['station_id'].tolist()

def free_map(panel):
    topo = pd.read_csv(REL / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv', dtype={'link_id': str})
    return topo.drop_duplicates('link_id').set_index('link_id')['free_speed_kmh']

def add_features(panel, split, preds, truth=False):
    days = masked_days(panel, split)
    truth_files = {f.name: f for f in (REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')} if truth else {}
    order = station_order(panel); pos = {s: i for i, s in enumerate(order)}; nS = len(order)
    lanes_map = s1.station_lanes(REL / 'corridors' / panel); free = free_map(panel)
    preds = preds.assign(day=preds['ts'].dt.normalize().dt.tz_localize(None))
    out = []
    for date, (regime, f) in days.items():
        p = preds[preds['day'] == date].copy()
        if p.empty:
            continue
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
        # neighbours at the gap edges: how much they changed since the last good reading / until the next one
        def nb_at(M, tix):
            res = []
            for ks in ((-1, 1), (-2, 2)):
                vals = []
                for k in ks:
                    j = si + k; okb = (j >= 0) & (j < nS) & (tix >= 0)
                    r = np.full(len(p), np.nan); r[okb] = M[tix[okb], j[okb]]; vals.append(r)
                res.append(np.nanmean(np.vstack(vals), axis=0))
            return np.where(np.isnan(res[0]), res[1], res[0])
        for k, M, nb in (('f', Fv, p['nb_flow'].to_numpy()), ('s', Sv, p['nb_speed'].to_numpy())):
            for e, col in (('p', f'g_pt_{k}'), ('n', f'g_nt_{k}')):
                tix = F.index.get_indexer(pd.DatetimeIndex(p[col].to_numpy()).tz_localize('UTC'))
                d = nb - nb_at(M, tix)
                p[f'nbd_{k}_{e}'] = d
                p[f'est_{k}_{e}'] = p[f'g_{"pv" if e == "p" else "nv"}_{k}'].to_numpy() + d
        m = p.rename(columns={'timestamp': 'ts_str', 'ts': 'timestamp'})
        if truth:
            t = read_day(truth_files[f.name])[KEYS + VALUES].rename(columns={'speed_kmh': 's_true', 'flow_vph': 'f_true'})
            m = m.merge(t, on=KEYS, how='left')
        m['hour'] = m['timestamp'].dt.hour; m['dow'] = m['timestamp'].dt.dayofweek
        m['lanes'] = s1.lane_vector(m['station_id'], m['link_id'], lanes_map)
        m['free'] = m['link_id'].map(free).to_numpy(); m['regime'] = REGIMES.index(regime)
        out.append(m)
    return pd.concat(out, ignore_index=True)

AGGS, TRAIN = {}, []
for panel in panels:
    ts0 = time.time()
    AGGS[panel] = daily_profiles(panel)
    days = masked_days(panel, 'train')
    dates = sorted(sample_days(days))
    preds, _ = predict_split(panel, 'train', CFG, AGGS[panel], only_dates=set(dates))
    t = add_features(panel, 'train', preds, truth=True)
    t['panel_code'] = panels.index(panel)
    hold_days = set(pd.Timestamp(d) for d in dates[2::4])  # every 4th sampled day, never used to fit the check models
    t['_hold'] = t['day'].isin(hold_days).astype('float32')
    TRAIN.append(t[FEATS + ['s_true', 'f_true', '_hold']].astype({c: 'float32' for c in FEATS + ['s_true', 'f_true']}))
    print('train rows', panel, len(t), 'sec', round(time.time() - ts0), flush=True)
    del preds, t
TR = pd.concat(TRAIN, ignore_index=True); del TRAIN
TR = TR[TR['s_true'].notna() & TR['f_true'].notna()]

# quick check on held-out days: keep the gap features only if they beat the old feature set
t1 = time.time()
hold = TR['_hold'].to_numpy() > 0
fit_i = np.flatnonzero(~hold); fit_i = np.random.default_rng(0).choice(fit_i, min(len(fit_i), 3_000_000), replace=False)
H = TR[hold]
def rmse(pred_s, pred_f):
    return (float(np.sqrt(np.mean((pred_s - H['s_true']) ** 2))),
            float(np.sqrt(np.mean(((pred_f - H['f_true']) / H['lanes'].clip(lower=1)) ** 2))))
CHECK = {'straight line': rmse(H['speed_kmh'], H['flow_vph'])}
for name, cols in (('old features', FEATS_OLD), ('+ gap features', FEATS)):
    Xf = TR.iloc[fit_i]
    mf = lgb.train(PARAMS, lgb.Dataset(Xf[cols], Xf['f_true'] - Xf['flow_vph']), ROUNDS)
    ms = lgb.train(PARAMS, lgb.Dataset(Xf[cols], Xf['s_true'] - Xf['speed_kmh']), ROUNDS)
    CHECK[name] = rmse(H['speed_kmh'] + ms.predict(H[cols]), (H['flow_vph'] + mf.predict(H[cols])).clip(lower=0))
    del Xf, mf, ms
print(pd.DataFrame(CHECK, index=['rmse speed km/h', 'rmse flow per lane']).round(3).to_string(), flush=True)
o, n = CHECK['old features'], CHECK['+ gap features']
GAIN = (n[0] / o[0] + n[1] / o[1]) / 2
if GAIN > 0.998:
    FEATS = FEATS_OLD
print('relative error new/old', round(GAIN, 4), '-> using', 'gap features' if FEATS is not FEATS_OLD else 'old features',
      '| check sec', round(time.time() - t1), flush=True)
del H
t1 = time.time()
MODEL_FLOW = lgb.train(PARAMS, lgb.Dataset(TR[FEATS], TR['f_true'] - TR['flow_vph']), ROUNDS)
MODEL_SPEED = lgb.train(PARAMS, lgb.Dataset(TR[FEATS], TR['s_true'] - TR['speed_kmh']), ROUNDS)
print('trained on', len(TR), 'rows in', round(time.time() - t1), 'sec', flush=True)
del TR

def build_split(split):
    out_path = OUT / f'state_{split}.csv'
    total, first = 0, True
    for panel in panels:
        aggs = AGGS[panel]
        first_day = min(masked_days(panel, split))
        result, sheet = predict_split(panel, split, CFG, aggs, hist_fixed=hist_before(aggs, first_day))
        feats = add_features(panel, split, result)
        assert len(feats) == len(result), (panel, split, len(feats), len(result))
        feats['panel_code'] = panels.index(panel)
        flow = np.clip(feats['flow_vph'] + MODEL_FLOW.predict(feats[FEATS]), 0, None)
        speed = np.clip(feats['speed_kmh'] + MODEL_SPEED.predict(feats[FEATS]), 1, None)
        k5 = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
        final = feats[['panel', 'ts_str', 'station_id', 'link_id', 'mask_regime']].rename(columns={'ts_str': 'timestamp'}).assign(speed_kmh=speed.to_numpy(), flow_vph=flow.to_numpy())
        assert len(final) == len(sheet), (panel, split, len(final), len(sheet))
        assert not final.duplicated(k5).any(), (panel, split)
        assert final.merge(sheet[k5], on=k5, how='outer', indicator=True)['_merge'].eq('both').all(), (panel, split)
        assert np.isfinite(final[VALUES].to_numpy()).all(), (panel, split)
        final[k5 + VALUES].to_csv(out_path, mode='w' if first else 'a', header=first, index=False)
        first = False
        total += len(final)
        print(split, panel, len(final), 'mean change speed', round(float((speed.to_numpy() - feats['speed_kmh'].to_numpy()).mean()), 3),
              'flow', round(float((flow.to_numpy() - feats['flow_vph'].to_numpy()).mean()), 2), flush=True)
    assert total == EXPECTED_ROWS[split], (split, total)
    return out_path

parts = [build_split(s) for s in ['validation', 'private']]
with open(OUT / 'state.csv', 'w') as out:
    for i, p in enumerate(parts):
        with open(p) as f:
            header = f.readline()
            if i == 0:
                out.write(header)
            out.writelines(f)
        p.unlink()
n = sum(1 for _ in open(OUT / 'state.csv')) - 1
print('state.csv rows:', f'{n:,}')
assert n == sum(EXPECTED_ROWS.values()), n
