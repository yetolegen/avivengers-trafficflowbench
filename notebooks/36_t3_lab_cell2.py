
# ------------------------------------------------------------------ Task 3 lab: export held-out Task 1 predictions + truth frames
import shutil
LAB = OUT / 'lab'; LAB.mkdir(exist_ok=True)
TRAIN, HOLD = [], []
for panel in panels:
    ts0 = time.time()
    aggs = daily_profiles(panel)
    days = masked_days(panel, 'train')
    dates = sorted(sample_days(days))
    hold_days = sorted(set(pd.Timestamp(d) for d in dates[2::4]))
    preds, _ = predict_split(panel, 'train', OLD, aggs, only_dates=set(dates))
    t = add_features(panel, 'train', preds, truth=True)
    t['panel_code'] = panels.index(panel)
    t = t[t['s_true'].notna() & t['f_true'].notna()]
    h = t['day'].isin(hold_days)
    TRAIN.append(t.loc[~h, FEATS_ALL + ['s_true', 'f_true']].astype('float32'))
    HOLD.append(t.loc[h, FEATS_ALL + ['s_true', 'f_true', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']])
    # station-level released truth + target flags on the held-out days (what the Task 3 scorer reconstructs)
    names = {days[d][1].name: days[d][0] for d in hold_days}
    sheet = load_sheet(panel, 'train', {d.strftime('%Y-%m-%d') for d in hold_days})
    tg = set(zip(sheet['ts'], sheet['station_id'], sheet['link_id'], sheet['mask_regime']))
    parts = []
    for f in sorted((REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')):
        if f.name not in names:
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        d = d[d['is_score_eligible'].astype(bool)].drop(columns='is_score_eligible')
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True); d['station_id'] = d['station_id'].astype(str); d['link_id'] = d['link_id'].astype(str)
        r = names[f.name]
        d['mask_regime'] = r
        d['is_target'] = np.fromiter((k in tg for k in zip(d['timestamp'], d['station_id'], d['link_id'], d['mask_regime'])), bool, len(d))
        parts.append(d)
    st = pd.concat(parts, ignore_index=True).rename(columns={'speed_kmh': 'true_speed', 'flow_vph': 'true_flow'})
    st['panel'] = panel
    st.to_parquet(LAB / f'state_{panel}.parquet', index=False)
    ramps = s3.load_ramp_flows(REL / 'corridors' / panel, 'train')
    if len(ramps):
        ramps['timestamp'] = pd.to_datetime(ramps['timestamp'], utc=True)
        ramps = ramps[ramps['timestamp'].dt.normalize().dt.tz_localize(None).isin(hold_days)]
    ramps.to_parquet(LAB / f'ramps_{panel}.parquet', index=False)
    nd = LAB / 'corridors' / panel / 'network'; nd.mkdir(parents=True, exist_ok=True)
    for fn in ('links.csv', 'fd_parameters.csv', 'lwr_mainline_topology.csv', 'ramp_attachment_map.csv'):
        src = REL / 'corridors' / panel / 'network' / fn
        if src.exists():
            shutil.copy(src, nd / fn)
    print(panel, 'hold days', len(hold_days), 'state rows', len(st), 'targets', int(st['is_target'].sum()), 'sec', round(time.time() - ts0), flush=True)
    del preds, t

TR = pd.concat(TRAIN, ignore_index=True); del TRAIN
HO = pd.concat(HOLD, ignore_index=True); del HOLD
print('train rows', len(TR), 'hold rows', len(HO), flush=True)
for tgt, base, k in (('f_true', 'flow_vph', 'f'), ('s_true', 'speed_kmh', 's')):
    m = lgb.train(PARAMS, lgb.Dataset(TR[FEATS_ALL], TR[tgt] - TR[base]), ROUNDS)
    HO['pred_' + k] = HO[base].to_numpy() + m.predict(HO[FEATS_ALL].to_numpy(np.float32))
HO['pred_s'] = HO['pred_s'].clip(lower=1); HO['pred_f'] = HO['pred_f'].clip(lower=0)
keep = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'pred_s', 'pred_f', 's_true', 'f_true', 'speed_kmh', 'flow_vph',
        'lanes', 'free', 'g_dp_s', 'g_dn_s', 'g_dp_f', 'g_dn_f']
HO[keep].to_parquet(LAB / 'hold_preds.parquet', index=False)
e = HO.assign(es=HO.pred_s - HO.s_true, ef=(HO.pred_f - HO.f_true) / HO.lanes.clip(lower=1))
print('hold rmse speed', float(np.sqrt((e.es ** 2).mean())), 'flow/lane', float(np.sqrt((e.ef ** 2).mean())), flush=True)
