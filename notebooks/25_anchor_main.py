import time, traceback, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
EXP = ['D12_I5_N', 'D7_I210_W', 'D7_I405_S', 'D7_I10_E']

def t1(panel, sub, names):
    original = s1.masked_files
    s1.masked_files = lambda rel, p, split, regime: [f for f in original(rel, p, split, regime) if f.name in names]
    try:
        det = pd.DataFrame(s1.score_panel(panel, REL, 'train', sub))
        return float(det[det['level'] == 'panel']['S_state'].iloc[0])
    finally:
        s1.masked_files = original

def sfd(panel, sub, names):
    original = s3.files
    s3.files = lambda panel_dir, split: [f for f in original(panel_dir, split) if f.name in names]
    try:
        out = []
        for r in REGIMES:
            d, _ = s3.derive_physics(panel, REL, 'train', sub, r)
            out.append(float(s3.score_panel(panel, d, REL, modes, r).get('S_FD', np.nan)))
        return float(np.nanmean(out))
    finally:
        s3.files = original

def link_grid(df, times, lk, L):
    g = df.groupby(['timestamp', 'link_id'])[['speed_kmh', 'flow_vph']].mean().reset_index()
    ti = times.get_indexer(g.timestamp); li = g.link_id.map({l: i for i, l in enumerate(lk)})
    m = (ti >= 0) & li.notna().to_numpy()
    N = np.full((len(times), len(lk)), np.nan)
    ci = li[m].astype(int).to_numpy()
    N[ti[m], ci] = (g.flow_vph / g.speed_kmh.clip(lower=1.0)).to_numpy()[m] * L[ci]
    return N

def runs(col):
    idx = np.flatnonzero(col)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))

def variants(Np, No, TG):
    out = {k: Np.copy() for k in ('linear', 'shift', 'shift+smooth', 'mix')}
    T, Lc = Np.shape
    for c in range(Lc):
        for s, e in runs(TG[:, c]):
            aL = No[s - 1, c] if s > 0 and np.isfinite(No[s - 1, c]) else np.nan
            aR = No[e + 1, c] if e + 1 < T and np.isfinite(No[e + 1, c]) else np.nan
            seg = Np[s:e + 1, c]; n = e - s + 1
            tau = np.arange(1, n + 1) / (n + 1)
            if np.isfinite(aL) and np.isfinite(aR):
                lin = aL + (aR - aL) * tau; oL, oR = aL - seg[0], aR - seg[-1]
            elif np.isfinite(aL):
                lin = np.full(n, aL); oL = oR = aL - seg[0]
            elif np.isfinite(aR):
                lin = np.full(n, aR); oL = oR = aR - seg[-1]
            else:
                continue
            sh = np.maximum(seg + oL * (1 - tau) + oR * tau, 0.0)
            sm = sh.copy()
            if n >= 3:
                pad = np.r_[aL if np.isfinite(aL) else sh[0], sh, aR if np.isfinite(aR) else sh[-1]]
                sm = (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0
            out['linear'][s:e + 1, c] = np.maximum(lin, 0.0)
            out['shift'][s:e + 1, c] = sh
            out['shift+smooth'][s:e + 1, c] = sm
            out['mix'][s:e + 1, c] = 0.5 * np.maximum(lin, 0.0) + 0.5 * sh
    return out

def lwr_proxy(Nsub, Nt):
    d_s = np.diff(Nsub, axis=0); d_t = np.diff(Nt, axis=0)
    ok = np.isfinite(d_s) & np.isfinite(d_t)
    return float(np.abs(d_s - d_t)[ok].sum()), float(np.abs(d_t)[ok].sum())

def apply(sub, ratio_tab, mode):
    m = sub.merge(ratio_tab, on=['timestamp', 'link_id'], how='left')
    r = m.ratio.fillna(1.0).clip(0.2, 5.0).to_numpy()
    if mode == 'keep speed':
        m['flow_vph'] = m.flow_vph * r
    else:
        m['speed_kmh'] = (m.speed_kmh / r).clip(1.0, 140.0)
    return m.drop(columns=['ratio'])

ROWS = []
for panel in EXP:
    ts = time.time()
    try:
        aggs = daily_profiles(panel); days = masked_days(panel, 'train')
        dates = sorted(sample_days(days)); pick = []
        for r in REGIMES:
            pick += [d for d in dates if days[d][0] == r][:3]
        preds, _ = predict_split(panel, 'train', OLD, aggs, only_dates=set(pick))
        names = {days[d][1].name for d in pick}
        sub = preds.drop(columns=['timestamp']).rename(columns={'ts': 'timestamp'})[['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']]
        sub['link_id'] = sub.link_id.astype(str)
        links = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).drop_duplicates('link_id')
        Lmap = links.set_index('link_id')['length_km'].astype(float).to_dict()
        truth_files = {f.name: f for f in (REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')}
        acc = {k: [0.0, 0.0] for k in ('ours', 'linear', 'shift', 'shift+smooth', 'mix')}
        ratio_parts = {k: [] for k in ('linear', 'shift', 'shift+smooth', 'mix')}
        for d in pick:
            day = d.strftime('%Y-%m-%d'); f = days[d][1]
            times = pd.date_range(day + 'T00:00:00Z', periods=288, freq='5min')
            obs = read_day(f); tru = read_day(truth_files[f.name])
            obs = obs.assign(link_id=obs.link_id.astype(str)); tru = tru.assign(link_id=tru.link_id.astype(str))
            sd = sub[sub.timestamp.dt.strftime('%Y-%m-%d') == day]
            lk = sorted(set(tru.link_id) | set(sd.link_id))
            Lv = np.array([Lmap.get(l, 1.0) for l in lk])
            obs = obs[obs[['speed_kmh', 'flow_vph']].notna().all(1)]
            tkeys = set(zip(sd.timestamp, sd.link_id))
            obs = obs[[(a, b) not in tkeys for a, b in zip(obs.timestamp, obs.link_id)]]
            No = link_grid(obs, times, lk, Lv); Np = link_grid(sd, times, lk, Lv); Nt = link_grid(tru, times, lk, Lv)
            TG = np.isfinite(Np)
            a, b = lwr_proxy(np.where(TG, Np, No), Nt); acc['ours'][0] += a; acc['ours'][1] += b
            tt, cc = np.nonzero(TG)
            for k, Nv in variants(Np, No, TG).items():
                a, b = lwr_proxy(np.where(TG, Nv, No), Nt); acc[k][0] += a; acc[k][1] += b
                ratio_parts[k].append(pd.DataFrame({'timestamp': times[tt], 'link_id': [lk[c] for c in cc], 'ratio': Nv[tt, cc] / np.maximum(Np[tt, cc], 1e-6)}))
        ROWS.append(dict(panel=panel, variant='ours', how='-', S_state=t1(panel, sub, names), S_FD=sfd(panel, sub, names), LWR_proxy=1 - min(1, acc['ours'][0] / acc['ours'][1])))
        print(ROWS[-1], flush=True)
        for k in ratio_parts:
            rt = pd.concat(ratio_parts[k], ignore_index=True)
            for mode in ('keep speed', 'keep flow'):
                ps = apply(sub, rt, mode)
                ROWS.append(dict(panel=panel, variant=k, how=mode, S_state=t1(panel, ps, names), S_FD=sfd(panel, ps, names),
                                 LWR_proxy=1 - min(1, acc[k][0] / acc[k][1])))
                print(ROWS[-1], flush=True)
    except Exception:
        traceback.print_exc()
    print(panel, 'sec', round(time.time() - ts), flush=True)
R = pd.DataFrame(ROWS)
R['S_physics_est'] = R.S_FD / 3 + 2 * R.LWR_proxy / 3
base = R[R.variant == 'ours'].set_index('panel')
R['d_total_est'] = 0.35 * (R.S_state - R.panel.map(base.S_state)) + 0.15 * (R.S_physics_est - R.panel.map(base.S_physics_est))
print(R.round(4).to_string(index=False))
print('MEAN OVER ROADS'); print(R.groupby(['variant', 'how']).mean(numeric_only=True).round(4).sort_values('d_total_est', ascending=False).to_string(), flush=True)
