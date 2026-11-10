import inspect, time, traceback, warnings
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
for fn in ('files', 'state_for_regime', 'derive_physics', 'topology_boundary_flows', 'ramp_flows', 'score_panel', 'lwr_error'):
    if hasattr(s3, fn):
        print('#### SOURCE', fn); print(inspect.getsource(getattr(s3, fn))[:6000], flush=True)
print('s3 functions:', [n for n in dir(s3) if not n.startswith('_')], flush=True)
print('modes config sample:', str(modes)[:800], flush=True)
EXP = ['D12_I5_N', 'D7_I210_W', 'D7_I405_S']
LAMS = [0.3, 1.0, 3.0]
DT = 5.0 / 60.0

def with_files(names, fn):
    original = s3.files
    s3.files = lambda panel_dir, split: [f for f in original(panel_dir, split) if f.name in names]
    try:
        return fn()
    finally:
        s3.files = original

def t3(panel, sub, names):
    def run():
        rows, frames = [], {}
        for r in REGIMES:
            d, missing = s3.derive_physics(panel, REL, 'train', sub, r)
            res = s3.score_panel(panel, d, REL, modes, r)
            rows.append(dict(regime=r, **{k: float(v) for k, v in res.items() if isinstance(v, (int, float, np.floating, np.integer))}))
            frames[r] = d
        return pd.DataFrame(rows), frames
    return with_files(names, run)

def t1(panel, sub, names):
    original = s1.masked_files
    s1.masked_files = lambda rel, p, split, regime: [f for f in original(rel, p, split, regime) if f.name in names]
    try:
        det = pd.DataFrame(s1.score_panel(panel, REL, 'train', sub))
        return float(det[det['level'] == 'panel']['S_state'].iloc[0])
    finally:
        s1.masked_files = original

def network(panel):
    nd = REL / 'corridors' / panel / 'network'
    links = pd.read_csv(nd / 'links.csv', dtype={'link_id': str}).drop_duplicates('link_id')
    topo = pd.read_csv(nd / 'lwr_mainline_topology.csv', dtype=str).drop_duplicates('link_id')
    L = links.set_index('link_id')['length_km'].astype(float).to_dict()
    inc = {r.link_id: [x.strip() for x in str(r.incoming_link_ids).split(';') if x.strip() and x.strip() != 'nan'] for r in topo.itertuples()}
    out = {r.link_id: [x.strip() for x in str(r.outgoing_link_ids).split(';') if x.strip() and x.strip() != 'nan'] for r in topo.itertuples()}
    rm = pd.read_csv(nd / 'ramp_attachment_map.csv', dtype=str) if (nd / 'ramp_attachment_map.csv').exists() else None
    return L, inc, out, rm

def ramps(panel, dates, rm):
    if rm is None:
        return None
    parts = []
    for f in sorted((REL / 'corridors' / panel / 'train' / 'ramp_states').rglob('*.parquet')):
        d = pd.read_parquet(f, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'pct_observed'])
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
        d = d[d.timestamp.dt.strftime('%Y-%m-%d').isin(dates)]
        if len(d):
            parts.append(d)
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d['ramp_link_id'] = d.ramp_link_id.astype(str)
    d = d.merge(rm[['ramp_link_id', 'nearest_mainline_link_id']], on='ramp_link_id', how='left')
    d['f'] = np.where(d.pct_observed >= 75, d.flow_vph, np.nan)
    d['on'] = np.where(d.ramp_type == 'OR', d.f, 0.0); d['off'] = np.where(d.ramp_type == 'FR', d.f, 0.0)
    return d.groupby(['timestamp', 'nearest_mainline_link_id'])[['on', 'off']].sum(min_count=1).reset_index().rename(columns={'nearest_mainline_link_id': 'link_id'})

def project(sub, frames, net, rmp, lam):
    L, inc, out, _ = net
    tgt = sub.groupby(['timestamp', 'link_id'])[['speed_kmh', 'flow_vph']].mean().reset_index()
    tgt['is_t'] = True
    new_rows = []
    for r, d in frames.items():
        g = d[['timestamp', 'link_id', 'speed_kmh', 'flow_vph']].copy()
        g['link_id'] = g.link_id.astype(str)
        g = g.merge(tgt, on=['timestamp', 'link_id'], how='left', suffixes=('', '_t'))
        g['is_t'] = g.is_t.fillna(False).astype(bool)
        g['speed_kmh'] = np.where(g.is_t, g.speed_kmh_t, g.speed_kmh); g['flow_vph'] = np.where(g.is_t, g.flow_vph_t, g.flow_vph)
        for day, gd in g.groupby(g.timestamp.dt.strftime('%Y-%m-%d')):
            times = pd.date_range(day + 'T00:00:00Z', periods=288, freq='5min')
            lk = sorted(set(gd.link_id))
            li = {l: i for i, l in enumerate(lk)}
            ti = times.get_indexer(gd.timestamp)
            ok = ti >= 0
            nT, nL = 288, len(lk)
            Q = np.full((nT, nL), np.nan); V = np.full((nT, nL), np.nan); TG = np.zeros((nT, nL), bool)
            c = gd.link_id.map(li).to_numpy()
            Q[ti[ok], c[ok]] = gd.flow_vph.to_numpy()[ok]; V[ti[ok], c[ok]] = gd.speed_kmh.to_numpy()[ok]; TG[ti[ok], c[ok]] = gd.is_t.to_numpy()[ok]
            Lv = np.array([L.get(l, 1.0) for l in lk])
            N = Q / np.maximum(V, 1.0) * Lv[None, :]
            ON = np.zeros((nT, nL)); OFF = np.zeros((nT, nL)); RV = np.ones((nT, nL), bool)
            if rmp is not None:
                rd = rmp[rmp.timestamp.dt.strftime('%Y-%m-%d') == day]
                rt = times.get_indexer(rd.timestamp); rc = rd.link_id.map(li)
                m = (rt >= 0) & rc.notna().to_numpy()
                rc = rc[m].astype(int).to_numpy(); rt = rt[m]
                on = rd.on.to_numpy()[m]; off = rd.off.to_numpy()[m]
                ON[rt, rc] = np.nan_to_num(on); OFF[rt, rc] = np.nan_to_num(off)
                RV[rt, rc] = ~(np.isnan(on) | np.isnan(off))
            tt, cc = np.nonzero(TG)
            nu = len(tt)
            if nu == 0:
                continue
            uid = -np.ones((nT, nL), np.int64); uid[tt, cc] = np.arange(nu)
            rows, cols, vals, rhs = [], [], [], []
            e = 0
            for i, l in enumerate(lk):
                ins = [li[u] for u in inc.get(l, []) if u in li] or [i]
                outs = [li[u] for u in out.get(l, []) if u in li] or [i]
                for t in range(nT - 1):
                    terms = [(t + 1, i, 'N', 1.0), (t, i, 'N', -1.0)] + [(t, u, 'Q', -DT) for u in ins] + [(t, u, 'Q', DT) for u in outs]
                    if not any(TG[a, b] for a, b, _, _ in terms) or not RV[t, i]:
                        continue
                    const = DT * (ON[t, i] - OFF[t, i]); good = True; rr = []
                    for a, b, kind, co in terms:
                        if TG[a, b]:
                            rr.append((uid[a, b] + (0 if kind == 'N' else nu), co))
                        else:
                            val = N[a, b] if kind == 'N' else Q[a, b]
                            if not np.isfinite(val):
                                good = False; break
                            const -= co * val
                    if not good:
                        continue
                    for j, co in rr:
                        rows.append(e); cols.append(j); vals.append(co)
                    rhs.append(const); e += 1
            if e == 0:
                continue
            A = sp.csr_matrix((vals, (rows, cols)), shape=(e, 2 * nu))
            x0 = np.concatenate([np.nan_to_num(N[tt, cc]), np.nan_to_num(Q[tt, cc])])
            sN = 0.1 * x0[:nu] + 0.5; sQ = DT * (0.05 * x0[nu:] + 50.0)
            W = sp.diags(np.concatenate([lam / sN, lam * DT / sQ]))
            M = sp.vstack([A, W]).tocsr()
            b = np.concatenate([np.array(rhs) - A @ x0, np.zeros(2 * nu)])
            y = lsqr(M, b, atol=1e-7, btol=1e-7, iter_lim=3000)[0]
            x = x0 + y
            Nn = np.maximum(x[:nu], 0.05); Qn = np.maximum(x[nu:], 0.0)
            Vn = np.clip(Qn * Lv[cc] / Nn, 3.0, 140.0)
            new_rows.append(pd.DataFrame({'timestamp': times[tt], 'link_id': [lk[k] for k in cc], 'v_new': Vn, 'q_new': Qn}))
    if not new_rows:
        return sub
    nw = pd.concat(new_rows, ignore_index=True)
    out_sub = sub.merge(nw, on=['timestamp', 'link_id'], how='left')
    out_sub['speed_kmh'] = out_sub.v_new.fillna(out_sub.speed_kmh); out_sub['flow_vph'] = out_sub.q_new.fillna(out_sub.flow_vph)
    return out_sub.drop(columns=['v_new', 'q_new'])

ALLR = []
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
        truth_files = {f.name: f for f in (REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')}
        tr = pd.concat([read_day(truth_files[days[d][1].name])[KEYS + VALUES] for d in pick], ignore_index=True)
        tsub = sub.drop(columns=VALUES).merge(tr, on=KEYS, how='left')
        tsub['speed_kmh'] = tsub.speed_kmh.fillna(sub.speed_kmh); tsub['flow_vph'] = tsub.flow_vph.fillna(sub.flow_vph)
        res_p, frames = t3(panel, sub, names)
        print(panel, 'frame columns', list(frames['R1'].columns), flush=True)
        print(frames['R1'].head(3).to_string(), flush=True)
        res_t, _ = t3(panel, tsub, names)
        ALLR.append(dict(panel=panel, variant='our predictions', S_state=t1(panel, sub, names), **res_p.mean(numeric_only=True).to_dict()))
        ALLR.append(dict(panel=panel, variant='TRUTH as submission', S_state=t1(panel, tsub, names), **res_t.mean(numeric_only=True).to_dict()))
        print(pd.DataFrame(ALLR[-2:]).to_string(), flush=True)
        net = network(panel)
        rmp = ramps(panel, {d.strftime('%Y-%m-%d') for d in pick}, net[3])
        for lam in LAMS:
            t0 = time.time()
            psub = project(sub, frames, net, rmp, lam)
            res_x, _ = t3(panel, psub, names)
            ALLR.append(dict(panel=panel, variant=f'projection lam={lam}', S_state=t1(panel, psub, names), **res_x.mean(numeric_only=True).to_dict()))
            print(ALLR[-1], 'sec', round(time.time() - t0), flush=True)
    except Exception:
        traceback.print_exc()
    print(panel, 'sec', round(time.time() - ts), flush=True)
R = pd.DataFrame(ALLR)
print(R.round(4).to_string(index=False))
print(R.groupby('variant').mean(numeric_only=True).round(4).to_string(), flush=True)
