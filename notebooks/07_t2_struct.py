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
import glob, json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
B = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)[0]
sys.path.insert(0, B)
import t2lib as T
cfg = json.load(open(f'{B}/config.json')); sd = json.load(open(f'{B}/split_definitions.json'))
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
FAM = T.load_family_map(REL); PC = cfg['panel_codes']; USED = set(cfg['used_channels']); ELIG = cfg['elig_mode']
TH = cfg['evaluation_models']['v2']['thresholds']; GRID = cfg['threshold_grid']
first = sd['dates_used'][0]; n_days = sd['dates_used'][2]; n_t = n_days * 288
t0 = T.text_to_minutes(first + 'T00:00:00Z')
fit_days = np.zeros(n_days, bool)
for d in [p for p in sd['partitions'] if p['name'] == 'fit'][0]['dates']:
    fit_days[(pd.Timestamp(d) - pd.Timestamp(first)).days] = True
day_rows = np.repeat(fit_days, 288)
EVAL = ['tune', 'test', 'holdout', 'offtest']
cat = pd.read_csv(f'{B}/metrics/generated_windows_catalog.csv')
cat = cat[cat.partition.isin(['fit'] + EVAL)].reset_index(drop=True)
cat['Tmin'] = [T.text_to_minutes(x) for x in cat['T']]
cat = cat[(cat.Tmin - t0) // 5 >= 12].reset_index(drop=True); cat['gid'] = np.arange(len(cat))
print('windows', cat.groupby('partition').size().to_dict(), flush=True)
XF, YF, XE, YE, GE = [], [], [], [], []
for panel in cfg['panels']:
    ts = time.time(); rel = Path(REL)
    fm = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked')
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    cm = T.resolve_colmap(T.parquet_columns(fm[0]), 'm'); cu = T.resolve_colmap(T.parquet_columns(fu[0]), 'u')
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(fm, cm, USED, ELIG, ctx, t0, n_t, 'masked')
    um, _ = T.load_series(fu, cu, USED, ELIG, ctx, t0, n_t, 'unmasked')
    ch = chain_of(ctx); tot = head_totals(um['spd'], ctx.thr, ch, day_rows)
    hr = tot / max(int(fit_days.sum()), 1); bn_set = set(np.argsort(-tot)[:4].tolist())
    sel = cat[cat.panel == panel]; Tm = sel.Tmin.to_numpy(np.int64); Ti = (Tm - t0) // 5
    tensor = T.tensor_from_series(hm, hm, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, t0); clim = T.Climatology.from_days(P, C, d0, fit_days)
    w, l, h = T.target_rows_for_generated(len(sel), ctx)
    X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=True)
    for k, v in nf.items():
        X[k] = v[w, l]
    hh = (h + 1) * 5 / 30.0
    X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
    X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
    lab, _ = T.labels_from_series(um, Ti, ctx.thr); y = lab[w, l, h]
    isfit = sel.partition.to_numpy()[w] == 'fit'; g = sel.gid.to_numpy()[w]
    XF.append(X[isfit]); YF.append(y[isfit]); XE.append(X[~isfit]); YE.append(y[~isfit]); GE.append(g[~isfit])
    print(panel, 'chain', len(ch), 'bn', sorted(bn_set), 'rows fit/eval', int(isfit.sum()), int((~isfit).sum()), 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl, nf, X
XF = pd.concat(XF, ignore_index=True); XE = pd.concat(XE, ignore_index=True)
yF = np.concatenate(YF); yE = np.concatenate(YE); gE = np.concatenate(GE)
hcE = XE['hist_cond'].to_numpy(); hcF = XF['hist_cond'].to_numpy()
BASE = list(T.FEATURES); NEWF = [c for c in XF.columns if c not in BASE]; ALL = BASE + NEWF
print('features base', len(BASE), 'new', len(NEWF), 'train rows', len(XF), flush=True)
parts = {}
for p in EVAL:
    wt = cat[cat.partition == p]; loc = np.full(len(cat), -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
    m = np.isin(gE, wt.gid.to_numpy())
    parts[p] = (wt[['window_key', 'panel', 'family_id', 'condition']].reset_index(drop=True), m, loc[gE[m]])
def score(p, pred, detail=False):
    wt, m, rw = parts[p]
    r = T.score_proxy(wt, rw, yE[m], pred[m])
    return r if detail else r['overall']
def tune_th(prob):
    cur = {'onset_like': 0.4, 'other': 0.4}
    for _ in range(2):
        for key in ('onset_like', 'other'):
            sc = []
            for g_ in GRID:
                c = dict(cur); c[key] = g_
                sc.append(score('tune', T.thresholded(prob, hcE, c)))
            cur[key] = GRID[int(np.argmax(sc))]
    return cur
params = dict(cfg['params']); params.update(learning_rate=0.1, num_threads=4)
knownF = ~np.isnan(yF); knownE = ~np.isnan(yE); tuneM = parts['tune'][1]
def fit_model(name, cols, rowF=None, rowE=None):
    ts = time.time(); k = knownF if rowF is None else knownF & rowF
    cat_f = [c for c in cols if c == 'panel_code']
    dtr = lgb.Dataset(XF.loc[k, cols], label=yF[k], categorical_feature=cat_f)
    v = tuneM & knownE if rowE is None else tuneM & knownE & rowE
    dva = lgb.Dataset(XE.loc[v, cols], label=yE[v], reference=dtr)
    bst = lgb.train(params, dtr, 1200, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    print(name, 'rounds', bst.best_iteration, 'train rows', int(k.sum()), 'sec', round(time.time() - ts), flush=True)
    return bst, bst.predict(XE[cols].to_numpy(np.float32), num_iteration=bst.best_iteration, num_threads=4)
p_v2 = lgb.Booster(model_file=f'{B}/lgbm_eval_v2.txt').predict(XE[BASE].to_numpy(np.float32), num_threads=4)
bA, p_A = fit_model('A_new_features', ALL)
bR, p_R = fit_model('R_base_retrain', BASE)
bC, p_C = fit_model('C_no_road_id', [c for c in ALL if c not in ('panel_code', 'rel_pos')])
bB, p_B = fit_model('B_onset_only', ALL, hcF == 0, hcE == 0)
cands = {'V2 his model, his cutoffs': (p_v2, TH), 'V2 his model, retuned': (p_v2, None), 'R base feats retrained': (p_R, None),
         'A new feats': (p_A, None), 'C new feats no road id': (p_C, None), 'A + B onset model': (np.where(hcE == 0, p_B, p_A), None),
         'avg(V2, A)': ((p_v2 + p_A) / 2, None), 'avg(R, A, C)': ((p_R + p_A + p_C) / 3, None)}
rows = []
for name, (pr, th) in cands.items():
    th = th or tune_th(pr); pred = T.thresholded(pr, hcE, th)
    r = {p: round(score(p, pred), 4) for p in EVAL}
    det = pd.concat([score(p, pred, True)['window'] for p in ('test', 'holdout', 'offtest')])
    bc = det.groupby('condition').proxy_IoU.mean()
    r.update(avg3=round(np.mean([r['test'], r['holdout'], r['offtest']]), 4), onset=round(bc.get('proxy_onset', np.nan), 4),
             ongoing=round(bc.get('proxy_ongoing', np.nan), 4), th=str(th))
    r['name'] = name; rows.append(r)
print(pd.DataFrame(rows).set_index('name').to_string(), flush=True)
imp = pd.Series(bA.feature_importance('gain'), index=ALL).sort_values(ascending=False)
print('TOP 30 gain (model A):'); print((imp / imp.sum()).head(30).round(4).to_string())
print('share of gain from new features', round(float(imp[NEWF].sum() / imp.sum()), 3), flush=True)
