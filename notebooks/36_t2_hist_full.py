import numpy as np
BIG = 10 ** 6

def chain_of(ctx):
    L = ctx.L; dn = ctx.seq_dn[:, 0]; up = ctx.seq_up[:, 0]
    pos = np.full(L, -1, np.int64); order = []
    for h0 in np.flatnonzero((up < 0) & (dn >= 0)):
        cur = int(h0)
        while cur >= 0 and pos[cur] < 0:
            order.append(cur); pos[cur] = len(order) - 1; cur = int(dn[cur])
    return np.array(order, np.int64)

def head_totals(spd_u, thr, ch, day_mask_rows):
    S = spd_u[:, ch]; obs = ~np.isnan(S)
    jam = obs & (np.nan_to_num(S, nan=np.inf) <= thr[ch][None, :])
    head = jam[:, :-1] & obs[:, 1:] & ~jam[:, 1:]
    tot = np.zeros(len(ch), np.float64); tot[:-1] = head[day_mask_rows].sum(0)
    return tot

def new_features(wl, tensor, ctx, ch, bn_set, head_rate, last_valid):
    W, L = wl['last_ratio'].shape; n = len(ch); ar = np.arange(n)[None, :]
    spd = tensor['spd']; thr = ctx.thr; valid = ~np.isnan(spd); rows = np.arange(W)[:, None]
    out = {}
    def back(a):
        o = np.full((W, L), np.nan, np.float32); o[:, ch] = a; return o
    def bcast(v):
        return np.broadcast_to(np.asarray(v, np.float32)[:, None], (W, L)).copy()
    def jam_at(s):
        val = last_valid(spd[..., :s + 1], valid[..., :s + 1])[2]
        return np.nan_to_num(val, nan=np.inf) <= thr[None, :]
    jams = {'now': np.nan_to_num(wl['last_ratio'], nan=np.inf) <= 1.0, 'm15': jam_at(9), 'm30': jam_at(6)}
    dd = {}
    for key, jam in jams.items():
        jc = jam[:, ch]
        nxt = np.minimum.accumulate(np.where(jc, ar, BIG)[:, ::-1], axis=1)[:, ::-1]
        prv = np.maximum.accumulate(np.where(jc, ar, -BIG), axis=1)
        dn = np.where(nxt < BIG, nxt - ar, np.nan).astype(np.float32)
        up = np.where(prv > -BIG, ar - prv, np.nan).astype(np.float32)
        dd[key] = (dn, up, nxt, prv)
        out[f'jd_dn_{key}'] = back(dn); out[f'jd_up_{key}'] = back(up)
        starts = (jc[:, 1:] & ~jc[:, :-1]).sum(1) + jc[:, 0]
        out[f'n_jam_{key}'] = bcast(jc.sum(1)); out[f'n_clu_{key}'] = bcast(starts)
    for k in ('15', '30'):
        out[f'jd_dn_d{k}'] = back(dd['now'][0] - dd[f'm{k}'][0]); out[f'jd_up_d{k}'] = back(dd['now'][1] - dd[f'm{k}'][1])
    out['n_jam_d30'] = out['n_jam_now'] - out['n_jam_m30']
    nxt, prv = dd['now'][2], dd['now'][3]
    for name in ('last_ratio', 'trend_15', 'run_state_min', 'flow_over_cap_last', 'occ_trend_15'):
        v = wl[name][rows, ch[np.clip(nxt, 0, n - 1)]]
        out[f'nj_dn_{name}'] = back(np.where(nxt >= BIG, np.nan, v))
    for name in ('last_ratio', 'trend_15', 'run_state_min'):
        v = wl[name][rows, ch[np.clip(prv, 0, n - 1)]]
        out[f'nj_up_{name}'] = back(np.where(prv <= -BIG, np.nan, v))
    jslot = np.where(valid, spd, np.inf) <= thr[None, :, None]
    anyj = jslot.any(-1); sl = np.arange(spd.shape[-1])[None, None, :]; last = spd.shape[-1] - 1
    lj = last - np.where(jslot, sl, -1).max(-1); fj = last - np.where(jslot, sl, 99).min(-1)
    out['min_since_jam'] = np.where(anyj, lj * 5, np.nan).astype(np.float32)
    out['jam_span_min'] = np.where(anyj, (fj - lj) * 5, np.nan).astype(np.float32)
    out['jam_slots_n'] = jslot.sum(-1).astype(np.float32)
    out['recovering'] = ((np.nan_to_num(wl['trend_15'], nan=0) > 0) & (np.nan_to_num(wl['last_ratio'], nan=9) <= 1.25)).astype(np.float32)
    isb = np.zeros(n, bool); isb[list(bn_set)] = True
    nb = np.minimum.accumulate(np.where(isb, np.arange(n), BIG)[::-1])[::-1]; okb = nb < BIG
    nbl = ch[np.clip(nb, 0, n - 1)]
    out['bn_dist'] = back(np.broadcast_to(np.where(okb, nb - np.arange(n), np.nan), (W, n)))
    out['bn_rate_self'] = back(np.broadcast_to(head_rate.astype(np.float32), (W, n)))
    out['bn_rate_next'] = back(np.broadcast_to(np.where(okb, head_rate[np.clip(nb, 0, n - 1)], np.nan), (W, n)))
    for name in ('last_ratio', 'trend_15', 'flow_over_cap_last', 'occ_trend_15', 'flow_trend_15', 'run_state_min', 'qoff_last'):
        out[f'bn_{name}'] = back(np.where(okb[None, :], wl[name][:, nbl], np.nan))
    return {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}

QH_EARLY = 24  # slots T-36..T-13 come from the masked mainline timeline; T-12..T come from the window tensor


def hist_block(G, Ti, spd13):
    """(W, L, 37) speeds for slots T-36..T: masked timeline G for the early part, window tensor (12 hist + T) for the last hour."""
    idx = np.asarray(Ti, np.int64)[:, None] + np.arange(-36, -12)[None, :]
    e = G[np.clip(idx, 0, None)].astype(np.float32)
    e[idx < 0] = np.nan
    return np.concatenate([e.transpose(0, 2, 1), np.asarray(spd13, np.float32)], axis=2)


def hist_feats(spd, thr, ch, dn1, up1):
    """Causal 3-hour queue history per (window, link). Unobserved slots carry the last observed state forward."""
    W, L, S = spd.shape; t = np.arange(S)
    obs = ~np.isnan(spd)
    jam = obs & (np.nan_to_num(spd, nan=np.inf) <= np.nan_to_num(thr, nan=-1.0)[None, :, None])
    li = np.maximum.accumulate(np.where(obs, t, -1), axis=2)
    known = li >= 0
    st = known & np.take_along_axis(jam, np.maximum(li, 0), axis=2)
    out = {}
    last_not = np.where(known & ~st, t, -1).max(2)
    out['qh_age'] = np.where(known[..., -1], np.where(st[..., -1], (S - 1 - last_not) * 5.0, 0.0), np.nan)
    lastj = np.where(st, t, -1).max(2)
    out['qh_since'] = np.where(lastj >= 0, (S - 1 - lastj) * 5.0, np.nan)
    for name, sl in (('qh_share', slice(None)), ('qh_share_early', slice(0, QH_EARLY))):
        no = obs[..., sl].sum(2)
        out[name] = np.where(no > 0, jam[..., sl].sum(2) / np.maximum(no, 1), np.nan)

    def nb(a, idx):
        o = np.full_like(a, np.nan); ok = idx >= 0; o[:, ok] = a[:, idx[ok]]; return o
    for k in ('qh_age', 'qh_share'):
        out[k + '_dn'] = nb(out[k], dn1); out[k + '_up'] = nb(out[k], up1)

    n = len(ch); ar = np.arange(n)[None, :]

    def jd(s):
        jc = st[:, ch, s]
        nxt = np.minimum.accumulate(np.where(jc, ar, BIG)[:, ::-1], axis=1)[:, ::-1]
        prv = np.maximum.accumulate(np.where(jc, ar, -BIG), axis=1)
        d = np.full((W, L), np.nan); d[:, ch] = np.where(nxt < BIG, nxt - ar, np.nan)
        u = np.full((W, L), np.nan); u[:, ch] = np.where(prv > -BIG, ar - prv, np.nan)
        return d, u, jc.sum(1).astype(np.float64)
    J = {lag: jd(S - 1 - 12 * k) for k, lag in enumerate(('now', 'm60', 'm120', 'm180'))}
    for lag, (d, u, nj) in J.items():
        out[f'qh_jd_dn_{lag}'] = d; out[f'qh_jd_up_{lag}'] = u
        out[f'qh_nj_{lag}'] = np.broadcast_to(nj[:, None], (W, L))
    for lag, m in (('m60', 12), ('m180', 36)):
        out[f'qh_front_v_{lag}'] = (J[lag][0] - J['now'][0]) / m   # links per 5 min the jam front moved upstream
        out[f'qh_nj_d_{lag}'] = out['qh_nj_now'] - out[f'qh_nj_{lag}']
    return {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}

import glob, json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
TP = glob.glob('/kaggle/input/**/t2lib.py', recursive=True)[0]
sys.path.insert(0, str(Path(TP).parent))
import t2lib as T
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
FAM = T.load_family_map(REL)
OUT = Path('/kaggle/working')
PANELS = ['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
PC = {p: i for i, p in enumerate(PANELS)}
USED = {'speed', 'flow', 'occ', 'pct', 'elig'}
ELIG = 'column'
LC = {'speed': 'speed_kmh', 'flow': 'flow_vph', 'occ': 'occupancy', 'pct': 'pct_observed', 'elig': 'is_score_eligible',
      'timestamp': 'timestamp', 'link_id': 'link_id', 'station_id': 'station_id'}
HC = {'speed': 'speed_kmh', 'flow': 'flow_vph', 'occ': 'occupancy', 'pct': 'pct_observed', 'elig': 'is_score_eligible',
      'timestamp': 'timestamp', 'link_id': 'link_id', 'window_id': 'window_id'}
N_DAYS = 273
N_T = N_DAYS * 288
T0 = T.text_to_minutes('2030-06-01T00:00:00Z')
SPLIT_SPAN = {'validation': (T.text_to_minutes('2031-03-01T00:00:00Z'), 31), 'private': (T.text_to_minutes('2031-04-01T00:00:00Z'), 30)}
G_DAYS = N_DAYS + 31 + 30
HOLD_DAY = 245
REC_DAYS = (28, 7)
WCFG = {'min_history_coverage': 0.7, 'max_persistence_iou': 0.9, 'min_spacing_minutes': 360, 'max_windows_per_day_condition': 2}
PARAMS = {'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255, 'num_threads': 4, 'seed': 2026,
          'deterministic': True, 'force_col_wise': True, 'verbose': -1}
ROUNDS = {'R': 340, 'A': 320}
THRESHOLDS = {'onset_like': 0.3, 'other': 0.25}
ONSET_CAP = 700

def rec_tables(spd_parts, thr):
    spd = np.concatenate(spd_parts, axis=0)
    assert spd.shape[0] == G_DAYS * 288, spd.shape
    obs = ~np.isnan(spd)
    jam = obs & (np.nan_to_num(spd, nan=1e9) <= thr[None, :])
    L = spd.shape[1]; k = 2
    out = []
    for A in (jam, obs):
        A = A.reshape(G_DAYS, 288, L).astype(np.float32)
        c = np.cumsum(np.pad(A, ((0, 0), (k + 1, k), (0, 0))), axis=1)
        A = c[:, 2 * k + 1:, :] - c[:, :-(2 * k + 1), :]
        out.append(np.concatenate([np.zeros((1, 288, L), np.float32), np.cumsum(A, axis=0)], axis=0))
    return out

def rec_feats(gs, l, h, CJ, CO):
    d = gs // 288
    f = {}
    for nd in REC_DAYS:
        d0 = np.maximum(d - nd, 0)
        for name, s in (('now', gs % 288), ('h', (gs + h + 1) % 288)):
            j = CJ[d, s, l] - CJ[d0, s, l]; o = CO[d, s, l] - CO[d0, s, l]
            f[f'rec{nd}_{name}'] = np.where(o > 0, j / np.maximum(o, 1), np.nan).astype(np.float32)
        f[f'rec{nd}_n'] = o.astype(np.float32)
    return f

def add_struct(X, nf, w_idx, l_idx, h_idx):
    for k, v in nf.items():
        X[k] = v[w_idx, l_idx]
    hh = (h_idx + 1) * 5 / 30.0
    X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
    X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh

def select_k(prob, gid, cand, kmin=0):
    pred = np.zeros(len(prob), np.int8)
    idx = np.flatnonzero(cand)
    if not len(idx):
        return pred
    o = idx[np.lexsort((-prob[idx], gid[idx]))]
    gg = gid[o]; p = prob[o]
    s = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1]]); e = np.r_[s[1:], len(o)]
    for a, b in zip(s, e):
        c = np.cumsum(p[a:b]); k = np.arange(1, b - a + 1)
        kb = int(np.argmax(c / (k + c[-1] - c))) + 1 if c[-1] > 0 else 0
        pred[o[a:a + max(kb, kmin)]] = 1
    return pred

def eiou_decode(prob, gid, h, is_on):
    on = select_k(prob, gid, is_on & (h == 5), kmin=1)
    og = select_k(prob, gid, ~is_on)
    return np.where(is_on, on, og).astype(np.int8)

STATE, XT, YT, META = {}, [], [], []
HCOV = {}
gid0 = 0
for panel in PANELS:
    ts = time.time(); rel = Path(REL)
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked'), LC, USED, ELIG, ctx, T0, N_T, 'masked')
    um, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states'), LC, USED, ELIG, ctx, T0, N_T, 'unmasked')
    parts = [hm['spd']]
    for sp_, (t0s, nd) in SPLIT_SPAN.items():
        s_, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / sp_ / 'mainline_states_masked'), LC, USED, ELIG, ctx, t0s, nd * 288, 'masked')
        parts.append(s_['spd']); del s_
    CJ, CO = rec_tables(parts, ctx.thr); G = np.concatenate(parts, axis=0).astype(np.float16); del parts
    tm = ctx.target_mask; nT = int(tm.sum())
    q = (um['qoff'][:, tm] > 0).any(1); cs = np.concatenate([[0], np.cumsum(q)])
    Tx = np.arange(12, N_T - 6)
    ccs = np.concatenate([[0], np.cumsum((um['nst'][:, tm] > 0).sum(1))])
    cov = (ccs[Tx] - ccs[Tx - 12]) / (12.0 * nT)
    onset = Tx[((cs[Tx + 6] - cs[Tx - 12]) == 0) & q[Tx + 6] & (cov >= 0.7)]
    rng = np.random.default_rng(2026 + PC[panel])
    if len(onset) > ONSET_CAP:
        onset = np.sort(rng.choice(onset, ONSET_CAP, replace=False))
    og_idx, og_cond, _ = T.select_proxy_windows(um, ctx, 0, N_T, WCFG, rng)
    ongoing = og_idx[np.array(og_cond) == 'proxy_ongoing']
    Ti = np.concatenate([onset, ongoing]).astype(np.int64); Tm = T0 + Ti * 5
    W = pd.DataFrame({'gid': np.arange(gid0, gid0 + len(Ti)), 'Ti': Ti, 'condition': ['proxy_onset'] * len(onset) + ['proxy_ongoing'] * len(ongoing)})
    gid0 += len(Ti)
    W['panel'] = panel; W['family_id'] = FAM.get(panel); W['window_key'] = [f'{panel}|{c}|{t}' for c, t in zip(W.condition, W.Ti)]
    META.append(W)
    ch = chain_of(ctx); tot = head_totals(um['spd'], ctx.thr, ch, np.ones(N_T, bool))
    hr = tot / N_DAYS; bn_set = set(np.argsort(-tot)[:4].tolist())
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, T0); clim = T.Climatology.from_days(P, C, d0, np.ones(N_DAYS, bool))
    tensor = T.tensor_from_series(um, hm, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
    w, l, h = T.target_rows_for_generated(len(Ti), ctx)
    X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=True)
    add_struct(X, nf, w, l, h)
    for k, v in rec_feats(Ti[w], l, h, CJ, CO).items():
        X[k] = v
    hb = hist_block(G, Ti, tensor['spd'])
    assert not any(c.startswith('qh_') for c in X.columns)
    for k, v in hist_feats(hb, ctx.thr, ch, ctx.seq_dn[:, 0], ctx.seq_up[:, 0]).items():
        X[k] = v[w, l]
    HCOV[panel] = float((~np.isnan(hb[..., :QH_EARLY])).mean()); del hb
    X = X.astype(np.float32)
    X['_gid'] = W.gid.to_numpy()[w]; X['_h'] = h; X['_day'] = Ti[w] // 288
    X['_on'] = (W.condition.to_numpy()[w] == 'proxy_onset')
    lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    XT.append(X); YT.append(lab[w, l, h])
    STATE[panel] = {'ctx': ctx, 'ch': ch, 'bn': bn_set, 'hr': hr, 'clim': clim, 'CJ': CJ, 'CO': CO, 'G': G}
    print(panel, 'onset', len(onset), 'ongoing', len(ongoing), 'rows', len(X), 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl, nf
XF = pd.concat(XT, ignore_index=True); yF = np.concatenate(YT); del XT, YT
META = pd.concat(META, ignore_index=True)
AUX = ['_gid', '_h', '_day', '_on']
BASE = list(T.FEATURES); REC = [c for c in XF.columns if c.startswith('rec')]
HIST = [c for c in XF.columns if c.startswith('qh_')]
NEWF = [c for c in XF.columns if c not in BASE and c not in REC and c not in AUX and c not in HIST]
print('history features', len(HIST), '| train early-history coverage', {p: round(v, 3) for p, v in HCOV.items()}, flush=True)
SETS = {'without month features': (BASE, BASE + NEWF), 'with month features': (BASE + REC, BASE + NEWF + REC)}
known = ~np.isnan(yF)
print('rows', len(XF), 'rec features', REC, flush=True)

def fit(cols, mask, rounds):
    return lgb.train(PARAMS, lgb.Dataset(XF.loc[mask, cols], label=yF[mask], categorical_feature=['panel_code']), rounds)

ev = XF['_day'].to_numpy() >= HOLD_DAY
gE = XF.loc[ev, '_gid'].to_numpy(); hE = XF.loc[ev, '_h'].to_numpy(); onE = XF.loc[ev, '_on'].to_numpy().astype(bool); yE = yF[ev]
wt = META[META.gid.isin(np.unique(gE))].reset_index(drop=True)
loc = np.full(gid0, -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
# v32: dedicated onset model. Onset windows only matter at T+30 (h index 5), so a model trained only on those
# cells learns "which bottleneck breaks first" instead of sharing capacity with ongoing queues.
from catboost import CatBoostClassifier
ON_PARAMS = dict(PARAMS, learning_rate=0.05, num_leaves=31, min_data_in_leaf=100)
ON_ROUNDS, ON_CB_ITERS, ON_BLEND = 400, 800, 0.5
onset_rows = XF['_on'].to_numpy().astype(bool) & (XF['_h'].to_numpy() == 5)

def fit_onset(cols, mask):
    m = mask & onset_rows
    lg = lgb.train(ON_PARAMS, lgb.Dataset(XF.loc[m, cols], label=yF[m], categorical_feature=['panel_code']), ON_ROUNDS)
    cb = CatBoostClassifier(iterations=ON_CB_ITERS, depth=6, learning_rate=0.05, thread_count=4, verbose=0, random_seed=2026)
    cb.fit(XF.loc[m, cols].to_numpy(np.float32), yF[m].astype(int))
    print('onset models trained on', int(m.sum()), 'cells, positive share', round(float(yF[m].mean()), 4), flush=True)
    return lg, cb

def onset_prob(models, X, cols):
    lg, cb = models; A = X[cols].to_numpy(np.float32)
    return 0.5 * lg.predict(A) + 0.5 * cb.predict_proba(A)[:, 1]

# v35: dedicated ongoing model (every horizon of ongoing windows): LightGBM + CatBoost on GPU.
OG_ROUNDS, OG_CB_ITERS, OG_BLEND = 400, 1500, 0.5
og_rows = ~XF['_on'].to_numpy().astype(bool)

def cb_clf_fit(A, y, iters, depth):
    kw = dict(iterations=iters, depth=depth, learning_rate=0.08, verbose=0, random_seed=2026)
    try:
        m = CatBoostClassifier(task_type='GPU', **kw); m.fit(A, y); return m
    except Exception as exc:
        print('CatBoost GPU unavailable, using CPU:', repr(exc)[:200], flush=True)
        m = CatBoostClassifier(thread_count=4, **kw); m.fit(A, y); return m

def fit_ongoing(cols, mask, iters=OG_CB_ITERS, depth=8, lg=None):
    m = mask & og_rows; A = XF.loc[m, cols]
    if lg is None:
        lg = lgb.train(PARAMS, lgb.Dataset(A, label=yF[m], categorical_feature=['panel_code']), OG_ROUNDS)
    cb = cb_clf_fit(A.to_numpy(np.float32), yF[m].astype(int), iters, depth)
    print('ongoing models trained on', int(m.sum()), 'cells, positive share', round(float(yF[m].mean()), 4), flush=True)
    return lg, cb

CR, CA = SETS['with month features']
t1 = time.time()
mr = fit(CR, known & ~ev, ROUNDS['R']); ma = fit(CA, known & ~ev, ROUNDS['A'])
XE = XF.loc[ev]
prob = (mr.predict(XE[CR].to_numpy(np.float32)) + ma.predict(XE[CA].to_numpy(np.float32))) / 2
mo = fit_onset(CA, known & ~ev)
cellE = onE & (hE == 5)
prob_on = prob.copy(); prob_on[cellE] = (1 - ON_BLEND) * prob[cellE] + ON_BLEND * onset_prob(mo, XE.loc[cellE], CA)
prob_only = prob.copy(); prob_only[cellE] = onset_prob(mo, XE.loc[cellE], CA)
rows = []
for dname, p_ in (('shared model (0.84511 recipe)', prob), ('+ onset model 50/50', prob_on), ('onset model only', prob_only)):
    det = T.score_proxy(wt[['window_key', 'panel', 'family_id', 'condition']], loc[gE], yE, eiou_decode(p_, gE, hE, onE))['window']
    b = det.groupby('condition').proxy_IoU.mean()
    rows.append(dict(onset_scoring=dname, onset=round(b.get('proxy_onset', np.nan), 4), ongoing=round(b.get('proxy_ongoing', np.nan), 4)))
HO = pd.DataFrame(rows)
print('HOLDOUT FEB 2031 (train Jun-Jan, score Feb windows), expected-IoU decode'); print(HO.to_string(index=False), flush=True)
best_i = int(HO.onset.idxmax())
ONSET_MODE = ('shared', 'blend', 'only')[best_i] if HO.onset[best_i] > HO.onset[0] + 0.002 else 'shared'
print('ONSET_MODE =', ONSET_MODE, '| holdout sec', round(time.time() - t1), flush=True)
base_p = {'shared': prob, 'blend': prob_on, 'only': prob_only}[ONSET_MODE]
mg = fit_ongoing(CA, known & ~ev)
cellG = ~onE; p_g = onset_prob(mg, XE.loc[cellG], CA)
og_blend = base_p.copy(); og_blend[cellG] = (1 - OG_BLEND) * base_p[cellG] + OG_BLEND * p_g
og_only = base_p.copy(); og_only[cellG] = p_g
rows = []
for dname, p_ in (('shared (onset as chosen)', base_p), ('+ ongoing model 50/50', og_blend), ('ongoing model only', og_only)):
    det = T.score_proxy(wt[['window_key', 'panel', 'family_id', 'condition']], loc[gE], yE, eiou_decode(p_, gE, hE, onE))['window']
    b = det.groupby('condition').proxy_IoU.mean()
    rows.append(dict(ongoing_scoring=dname, onset=round(b.get('proxy_onset', np.nan), 4), ongoing=round(b.get('proxy_ongoing', np.nan), 4)))
HG = pd.DataFrame(rows); print('HOLDOUT FEB, ongoing model'); print(HG.to_string(index=False), flush=True)
bg = int(HG.ongoing.idxmax())
OG_MODE = ('shared', 'blend', 'only')[bg] if HG.ongoing[bg] > HG.ongoing[0] + 0.002 else 'shared'
print('OG_MODE =', OG_MODE, '| holdout sec', round(time.time() - t1), flush=True)
# v36: queue history (last 3 h, causal) and bigger CatBoost for ongoing windows; history features for the onset model.
CH = CA + HIST
def score(p_):
    det = T.score_proxy(wt[['window_key', 'panel', 'family_id', 'condition']], loc[gE], yE, eiou_decode(p_, gE, hE, onE))['window']
    b = det.groupby('condition').proxy_IoU.mean()
    return round(b.get('proxy_onset', np.nan), 4), round(b.get('proxy_ongoing', np.nan), 4)
def with_og(models, cols):
    p = og_only.copy(); p[cellG] = onset_prob(models, XE.loc[cellG], cols); return p
V = {'v0 current (CatBoost 1500, depth 8)': ((CA, OG_CB_ITERS, 8), og_only)}
m2 = fit_ongoing(CH, known & ~ev); V['v2 + history (CatBoost 1500, depth 8)'] = ((CH, OG_CB_ITERS, 8), with_og(m2, CH)); del m2   # CPU run: GPU slots taken
rows = [dict(variant=n_, **dict(zip(('onset', 'ongoing'), score(p_)))) for n_, (_, p_) in V.items()]
HV = pd.DataFrame(rows); print('HOLDOUT FEB, ongoing variants'); print(HV.to_string(index=False), flush=True)
bv = int(HV.ongoing.idxmax())
OG_PICK = HV.variant[bv] if HV.ongoing[bv] > HV.ongoing[0] + 0.002 and OG_MODE == 'only' else HV.variant[0]
(OG_COLS, OG_ITERS, OG_DEPTH), p_best = V[OG_PICK]
ON_COLS = CA
if ONSET_MODE == 'only':
    m4 = fit_onset(CH, known & ~ev)
    p4 = p_best.copy(); p4[cellE] = onset_prob(m4, XE.loc[cellE], CH); del m4
    o0, o4 = score(p_best)[0], score(p4)[0]
    print('onset with history features:', o4, 'vs', o0, flush=True)
    if o4 > o0 + 0.002:
        ON_COLS = CH
print('OG_PICK =', OG_PICK, '| onset uses history:', ON_COLS is CH, '| holdout sec', round(time.time() - t1), flush=True)
del mr, ma, mo, mg, XE, V
t1 = time.time()
model_r = fit(CR, known, ROUNDS['R']); model_a = fit(CA, known, ROUNDS['A'])
model_on = fit_onset(ON_COLS, known) if ONSET_MODE != 'shared' else None
model_og = fit_ongoing(OG_COLS, known, OG_ITERS, OG_DEPTH) if OG_MODE != 'shared' else None
# fallbacks without history features, used for a panel whose test timeline has too little early history
model_on_fb = fit_onset(CA, known) if model_on is not None and ON_COLS is CH else None
model_og_fb = fit_ongoing(CA, known) if model_og is not None and OG_COLS is CH else None
print('final models trained in', round(time.time() - t1), 'sec', flush=True)
del XF
MODELS = OUT / 'task2_models'; MODELS.mkdir(exist_ok=True)
model_r.save_model(str(MODELS / 'lgbm_r.txt')); model_a.save_model(str(MODELS / 'lgbm_a.txt'))
(MODELS / 'config.json').write_text(json.dumps({'params': PARAMS, 'rounds': ROUNDS, 'features_r': CR, 'features_a': CA, 'rec_days': REC_DAYS}, indent=2))

def predict_split(split):
    outs, outs_th, templates = [], [], []
    for panel in PANELS:
        st = STATE[panel]; ctx = st['ctx']
        idx = T.read_window_index(REL, panel, split, FAM)
        tpl = T.restrict_template(T.read_template(REL, panel, split), idx, f'{panel}/{split}')
        hist, raw = T.load_official_history(REL, panel, split, HC, USED)
        hmax = hist[hist._valid_ts].groupby('window_id')._min.max().to_dict()
        origins, _ = T.resolve_origins(idx, tpl, hmax, f'{panel}/{split}')
        files = T.parquet_files(Path(REL) / 'corridors' / panel / split / 'mainline_states_masked')
        dm = T.scan_dates(files, LC); origin_list = list(origins.values())
        t_rows = T.read_t_rows(T.files_covering(dm, origin_list), LC, USED, origin_list)
        o = T.prepare_official_panel(ctx, idx, tpl, origins, hist, raw, t_rows, ELIG, True, clim=st['clim'])
        X = o['X']
        nf = new_features(o['wl'], o['tensor'], ctx, st['ch'], st['bn'], st['hr'], T._last_valid)
        add_struct(X, nf, o['w_idx'], o['l_idx'], o['h_idx'])
        gs = ((np.asarray(o['T_min'], np.int64) - T0) // 5)[o['w_idx']]
        assert (gs // 288 >= N_DAYS).all() and (gs // 288 < G_DAYS).all()
        for k, v in rec_feats(gs, o['l_idx'], o['h_idx'], st['CJ'], st['CO']).items():
            X[k] = v
        hb = hist_block(st['G'], (np.asarray(o['T_min'], np.int64) - T0) // 5, o['tensor']['spd'])
        cov = float((~np.isnan(hb[..., :QH_EARLY])).mean()); hist_ok = cov >= 0.5 * HCOV[panel]
        for k, v in hist_feats(hb, ctx.thr, st['ch'], ctx.seq_dn[:, 0], ctx.seq_up[:, 0]).items():
            X[k] = v[o['w_idx'], o['l_idx']]
        print(split, panel, 'early-history coverage', round(cov, 3), 'train', round(HCOV[panel], 3),
              'ok' if hist_ok else 'LOW -> models without history', flush=True)
        prob = (model_r.predict(X[CR].to_numpy(np.float32)) + model_a.predict(X[CA].to_numpy(np.float32))) / 2
        if not np.isfinite(prob).all():
            raise ValueError(f'{panel}/{split}: non-finite probabilities')
        cmap = idx.assign(window_id=idx.window_id.astype(str)).set_index('window_id').condition
        wids = np.asarray(o['window_ids'], dtype=object)[o['w_idx']]
        is_on = cmap.reindex(wids.astype(str)).to_numpy() == 'queue_onset'
        if model_on is not None:
            cell = is_on & (np.asarray(o['h_idx']) == 5)
            if cell.any():
                mon, con = (model_on, ON_COLS) if hist_ok or model_on_fb is None else (model_on_fb, CA)
                p_on = onset_prob(mon, X.loc[cell], con)
                prob[cell] = p_on if ONSET_MODE == 'only' else (1 - ON_BLEND) * prob[cell] + ON_BLEND * p_on
        if model_og is not None and (~is_on).any():
            mg_, cols_ = (model_og, OG_COLS) if hist_ok or model_og_fb is None else (model_og_fb, CA)
            cg = ~is_on; p_g = onset_prob(mg_, X.loc[cg], cols_)
            prob[cg] = p_g if OG_MODE == 'only' else (1 - OG_BLEND) * prob[cg] + OG_BLEND * p_g
        pred = eiou_decode(prob, o['w_idx'], o['h_idx'], is_on)
        pred_th = T.thresholded(prob, X['hist_cond'].to_numpy(), THRESHOLDS)
        base = tpl[['window_id', 'timestamp', 'link_id']].reset_index(drop=True)
        outs.append(base.assign(queue_pred=pred)); outs_th.append(base.assign(queue_pred=pred_th)); templates.append(base)
        print(split, panel, 'rows', len(base), 'queued (eiou)', int(pred.sum()), 'queued (cutoffs)', int(pred_th.sum()),
              'onset windows', int(cmap.eq('queue_onset').sum()), flush=True)
    tpl_all = pd.concat(templates, ignore_index=True)
    res = []
    for o_ in (outs, outs_th):
        out = pd.concat(o_, ignore_index=True); out['queue_pred'] = out['queue_pred'].astype(np.int64)
        T.validate_queue_output(out, tpl_all, split); res.append(out)
    return res

RES = {s: predict_split(s) for s in ('validation', 'private')}
for i, name in enumerate(('queue.csv', 'queue_cutoffs.csv')):
    comb = pd.concat([RES['validation'][i], RES['private'][i]], ignore_index=True)
    comb.to_csv(OUT / name, index=False)
    back = pd.read_csv(OUT / name, dtype=str, keep_default_na=False)
    assert list(back.columns) == T.OUTPUT_COLUMNS and len(back) == 174000 and back.queue_pred.isin(['0', '1']).all()
    print(name, 'rows', len(back), 'positive share', round(float((back.queue_pred == '1').mean()), 4), flush=True)
