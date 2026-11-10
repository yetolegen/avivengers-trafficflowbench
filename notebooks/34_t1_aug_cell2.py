import time
import warnings
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')
PARAMS = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'num_threads': 4, 'verbose': -1, 'seed': 2026}
ROUNDS = 400
CB_ITERS_CHECK, CB_ITERS_FINAL = 1500, 3000
CHECK_ROWS = 3_000_000
AUG_ON = True
FE = ['speed_kmh', 'flow_vph', 'span_min', 'f_u1', 'f_d1', 'f_u2', 'f_d2', 's_u1', 's_d1', 's_u2', 's_d2', 'nb_flow', 'nb_speed',
      'flow_minus_nb', 'speed_minus_nb', 'hour', 'dow', 'lanes', 'free', 'regime']
GAP_FE = [f'g_{a}_{k}' for k in 'sf' for a in ('dp', 'dn', 'pv', 'nv')] + \
         [f'{a}_{k}_{e}' for k in 'sf' for a in ('nbd', 'est') for e in 'pn']
CTX_OFF = (-12, -6, -3, -2, 2, 3, 6, 12)  # own station at -60/-30/-15/-10 and +10/+15/+30/+60 min
CTX_FE = [f'own_{k}_{"m" if o < 0 else "p"}{abs(o) * 5}' for o in CTX_OFF for k in 'sf']
FEATS_OLD = FE + ['panel_code']
FEATS_GAP = FEATS_OLD + GAP_FE          # = the 0.84862 task-1-gap model
FEATS_ALL = FEATS_GAP + CTX_FE

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
        grid = pd.date_range(pd.Timestamp(date).tz_localize('UTC'), periods=288, freq='5min')  # full day, so row offsets = minutes
        F = df.pivot_table(index='timestamp', columns='station_id', values='flow_vph', aggfunc='mean').reindex(index=grid, columns=order)
        S = df.pivot_table(index='timestamp', columns='station_id', values='speed_kmh', aggfunc='mean').reindex(index=grid, columns=order)
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
        okS = (si >= 0) & (si < nS) & (ti >= 0)
        for o in CTX_OFF:  # the station's own released readings around the target time
            r = ti + o; okr = okS & (r >= 0) & (r < len(grid))
            for k, M in (('s', Sv), ('f', Fv)):
                v = np.full(len(p), np.nan); v[okr] = M[r[okr], si[okr]]
                p[f'own_{k}_{"m" if o < 0 else "p"}{abs(o) * 5}'] = v
        m = p.rename(columns={'timestamp': 'ts_str', 'ts': 'timestamp'})
        if truth:
            t = read_day(truth_files[f.name])[KEYS + VALUES].rename(columns={'speed_kmh': 's_true', 'flow_vph': 'f_true'})
            m = m.merge(t, on=KEYS, how='left')
        m['hour'] = m['timestamp'].dt.hour; m['dow'] = m['timestamp'].dt.dayofweek
        m['lanes'] = s1.lane_vector(m['station_id'], m['link_id'], lanes_map)
        m['free'] = m['link_id'].map(free).to_numpy(); m['regime'] = REGIMES.index(regime)
        out.append(m)
    return pd.concat(out, ignore_index=True)

# ------------------------------------------------------------------ mask augmentation
# Each sampled training day borrows the organizers' real hiding pattern from a nearby day of the same regime.
# Readings that are released on this day but hidden on the donor day become extra targets: they are blanked
# (read_day honours AUG) before the straight line and the features are built, and their truth comes from the
# unmasked file. Only fit days are augmented; held-out days keep their official targets only.
def donor_of(d, days):
    reg = days[d][0]
    for off in (1, -1, 2, -2, 3, -3, 7, -7):
        c = d + pd.Timedelta(days=off)
        if c in days and days[c][0] == reg:
            return c
    return None

def aug_rows(panel, days, dates, aggs):
    donors = {d: donor_of(d, days) for d in dates}
    need = {x.strftime('%Y-%m-%d') for x in donors.values() if x is not None}
    if not need:
        return None
    sheet = load_sheet(panel, 'train', need)
    out = []
    for d, dn in donors.items():
        if dn is None:
            continue
        regime, f = days[d]
        r = sheet[sheet['date'] == dn.strftime('%Y-%m-%d')]
        ts = pd.DatetimeIndex(r['ts']) + (d - dn)
        raw = _read(f)
        ok = (raw['pct_observed'] >= 75) & raw['speed_kmh'].notna() & raw['flow_vph'].notna()
        obs = pd.MultiIndex.from_arrays([raw.loc[ok, 'station_id'].to_numpy(), pd.DatetimeIndex(raw.loc[ok, 'timestamp'])])
        cand = pd.MultiIndex.from_arrays([r['station_id'].to_numpy(), ts])
        keep = cand.isin(obs)
        if not keep.any():
            continue
        tgt = pd.DataFrame({'station_id': r['station_id'].to_numpy()[keep], 'link_id': r['link_id'].to_numpy()[keep], 'timestamp': ts[keep]})
        AUG[str(f)] = cand[keep]
        try:
            speed, flow, span, gap = predict_day(days, d, tgt, hist_before(aggs, d), OLD)
            piece = pd.DataFrame({'panel': panel, 'timestamp': tgt['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ'), 'ts': tgt['timestamp'],
                                  'station_id': tgt['station_id'], 'link_id': tgt['link_id'], 'mask_regime': regime})
            piece['speed_kmh'], piece['flow_vph'], piece['span_min'] = speed, flow, span
            for k_, v_ in gap.items():
                piece[k_] = v_
            out.append(add_features(panel, 'train', piece, truth=True))
        finally:
            AUG.pop(str(f), None)
    return pd.concat(out, ignore_index=True) if out else None

COLS = FEATS_ALL + ['s_true', 'f_true']
AGGS, TRAIN = {}, []
for panel in panels:
    ts0 = time.time()
    AGGS[panel] = daily_profiles(panel)
    days = masked_days(panel, 'train')
    dates = sorted(sample_days(days))
    hold_days = set(pd.Timestamp(d) for d in dates[2::4])  # every 4th sampled day, never used to fit the check models
    preds, _ = predict_split(panel, 'train', OLD, AGGS[panel], only_dates=set(dates))
    t = add_features(panel, 'train', preds, truth=True)
    t['panel_code'] = panels.index(panel)
    t['_hold'] = t['day'].isin(hold_days).astype('float32'); t['_aug'] = np.float32(0)
    TRAIN.append(t[COLS + ['_hold', '_aug']].astype({c: 'float32' for c in COLS}))
    n_aug = 0
    if AUG_ON:
        ta = aug_rows(panel, days, [d for d in dates if pd.Timestamp(d) not in hold_days], AGGS[panel])
        if ta is not None:
            ta['panel_code'] = panels.index(panel); ta['_hold'] = np.float32(0); ta['_aug'] = np.float32(1)
            TRAIN.append(ta[COLS + ['_hold', '_aug']].astype({c: 'float32' for c in COLS})); n_aug = len(ta)
            del ta
    print('train rows', panel, len(t), '+ augmented', n_aug, 'sec', round(time.time() - ts0), flush=True)
    del preds, t
TR = pd.concat(TRAIN, ignore_index=True); del TRAIN
TR = TR[TR['s_true'].notna() & TR['f_true'].notna()].reset_index(drop=True)
print('training rows', len(TR), '| augmented share', round(float(TR['_aug'].mean()), 3), flush=True)

# ------------------------------------------------------------------ models
def cb_fit(X, y, iters):
    kw = dict(iterations=iters, depth=8, learning_rate=0.08, loss_function='RMSE', random_seed=2026, verbose=0)
    try:
        m = CatBoostRegressor(task_type='GPU', border_count=128, **kw); m.fit(X, y)
    except Exception as exc:  # no GPU in this session: CPU fallback, slower
        print('CatBoost GPU unavailable, using CPU:', repr(exc)[:200], flush=True)
        m = CatBoostRegressor(thread_count=4, **kw); m.fit(X, y)
    return m

def fit_models(kind, cols, D, cb_iters):
    X = D[cols].to_numpy(np.float32)
    rf, rs = (D['f_true'] - D['flow_vph']).to_numpy(np.float32), (D['s_true'] - D['speed_kmh']).to_numpy(np.float32)
    M = {'cols': cols}
    if kind in ('lgb', 'blend'):
        M['lgb'] = (lgb.train(PARAMS, lgb.Dataset(X, rf), ROUNDS), lgb.train(PARAMS, lgb.Dataset(X, rs), ROUNDS))
    if kind in ('cb', 'blend'):
        M['cb'] = (cb_fit(X, rf, cb_iters), cb_fit(X, rs, cb_iters))
    return M

def residuals(M, D):
    X = D[M['cols']].to_numpy(np.float32); outs = []
    for key in ('lgb', 'cb'):
        if key in M:
            outs.append((M[key][0].predict(X), M[key][1].predict(X)))
    return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0)

# ------------------------------------------------------------------ check on held-out days (official targets only)
t1 = time.time()
hold = TR['_hold'].to_numpy() > 0; aug = TR['_aug'].to_numpy() > 0
rng = np.random.default_rng(0)
pick = lambda m: rng.choice(np.flatnonzero(m), min(int(m.sum()), CHECK_ROWS), replace=False)
off_i, all_i = pick(~hold & ~aug), pick(~hold)
H = TR[hold]
def rmse(rf, rs):
    return (float(np.sqrt(np.mean((H['speed_kmh'] + rs - H['s_true']) ** 2))),
            float(np.sqrt(np.mean((((H['flow_vph'] + rf).clip(lower=0) - H['f_true']) / H['lanes'].clip(lower=1)) ** 2))))
CHECK = {'straight line': rmse(0, 0)}
CONF = {'gap (0.84862)': ('lgb', FEATS_GAP, off_i), '+ctx': ('lgb', FEATS_ALL, off_i),
        '+ctx +aug': ('lgb', FEATS_ALL, all_i), 'catboost +ctx +aug': ('cb', FEATS_ALL, all_i)}
RES = {}
for name, (kind, cols, idx) in CONF.items():
    M = fit_models(kind, cols, TR.iloc[idx], CB_ITERS_CHECK)
    RES[name] = residuals(M, H); CHECK[name] = rmse(*RES[name]); del M
    print('check', name, CHECK[name], round(time.time() - t1), 's', flush=True)
RES['blend lgb+cb +ctx +aug'] = tuple((a + b) / 2 for a, b in zip(RES['+ctx +aug'], RES['catboost +ctx +aug']))
CHECK['blend lgb+cb +ctx +aug'] = rmse(*RES['blend lgb+cb +ctx +aug'])
print(pd.DataFrame(CHECK, index=['rmse speed km/h', 'rmse flow per lane']).round(3).to_string(), flush=True)
g = CHECK['gap (0.84862)']
rel = {k: (v[0] / g[0] + v[1] / g[1]) / 2 for k, v in CHECK.items() if k != 'straight line'}
CHOICE = min(rel, key=rel.get)
FINAL = {'gap (0.84862)': ('lgb', FEATS_GAP, False), '+ctx': ('lgb', FEATS_ALL, False), '+ctx +aug': ('lgb', FEATS_ALL, True),
         'catboost +ctx +aug': ('cb', FEATS_ALL, True), 'blend lgb+cb +ctx +aug': ('blend', FEATS_ALL, True)}[CHOICE]
print('relative error vs gap model:', {k: round(v, 4) for k, v in rel.items()}, '-> using', CHOICE, flush=True)
del H, RES

t1 = time.time()
kind, FEATS, use_aug = FINAL
MODEL = fit_models(kind, FEATS, TR if use_aug else TR[~aug], CB_ITERS_FINAL)
print('final', kind, 'trained on', len(TR) if use_aug else int((~aug).sum()), 'rows in', round(time.time() - t1), 'sec', flush=True)
del TR

def build_split(split):
    out_path = OUT / f'state_{split}.csv'
    total, first = 0, True
    for panel in panels:
        aggs = AGGS[panel]
        first_day = min(masked_days(panel, split))
        result, sheet = predict_split(panel, split, OLD, aggs, hist_fixed=hist_before(aggs, first_day))
        feats = add_features(panel, split, result)
        assert len(feats) == len(result), (panel, split, len(feats), len(result))
        feats['panel_code'] = panels.index(panel)
        rf, rs = residuals(MODEL, feats)
        flow = np.clip(feats['flow_vph'] + rf, 0, None)
        speed = np.clip(feats['speed_kmh'] + rs, 1, None)
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
