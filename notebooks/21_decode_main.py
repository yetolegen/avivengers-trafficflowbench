import glob, json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
TP = glob.glob('/kaggle/input/**/t2lib.py', recursive=True)[0]
sys.path.insert(0, str(Path(TP).parent))
import t2lib as T
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
FAM = T.load_family_map(REL)
PANELS = ['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
PC = {p: i for i, p in enumerate(PANELS)}
USED = {'speed', 'flow', 'occ', 'pct', 'elig'}
ELIG = 'column'
LC = {'speed': 'speed_kmh', 'flow': 'flow_vph', 'occ': 'occupancy', 'pct': 'pct_observed', 'elig': 'is_score_eligible',
      'timestamp': 'timestamp', 'link_id': 'link_id', 'station_id': 'station_id'}
N_DAYS = 273
N_T = N_DAYS * 288
T0 = T.text_to_minutes('2030-06-01T00:00:00Z')
WCFG = {'min_history_coverage': 0.7, 'max_persistence_iou': 0.9, 'min_spacing_minutes': 360, 'max_windows_per_day_condition': 2}
PARAMS = {'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255, 'num_threads': 4, 'seed': 2026,
          'deterministic': True, 'force_col_wise': True, 'verbose': -1}
ROUNDS = {'R': 340, 'A': 320}
TH = {'onset_like': 0.3, 'other': 0.25}
CAP_M = 70
FOLDS = [5, 8]
ONSET = ('proxy_onset', 'queue_onset')
dates = pd.date_range('2030-06-01', periods=N_DAYS)
mon = (dates.year * 12 + dates.month).to_numpy(); mon = mon - mon.min()
slot_mon = np.repeat(mon, 288)
XT = {f: [] for f in FOLDS}; YT = {f: [] for f in FOLDS}
XE = {f: [] for f in FOLDS}; YE = {f: [] for f in FOLDS}; GE = {f: [] for f in FOLDS}; HE = {f: [] for f in FOLDS}; PE = {f: [] for f in FOLDS}
METAS = []
gid_next = 0
for panel in PANELS:
    ts = time.time(); rel = Path(REL)
    fm = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked')
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(fm, LC, USED, ELIG, ctx, T0, N_T, 'masked')
    um, _ = T.load_series(fu, LC, USED, ELIG, ctx, T0, N_T, 'unmasked')
    tm = ctx.target_mask; nT = int(tm.sum())
    q = (um['qoff'][:, tm] > 0).any(1); cs = np.concatenate([[0], np.cumsum(q)])
    Tx = np.arange(12, N_T - 6)
    quiet = (cs[Tx + 6] - cs[Tx - 12]) == 0
    first_jam = q[Tx + 6]
    ccs = np.concatenate([[0], np.cumsum((um['nst'][:, tm] > 0).sum(1))])
    cov = (ccs[Tx] - ccs[Tx - 12]) / (12.0 * nT)
    cand = Tx[quiet & first_jam & (cov >= 0.7)]
    cand = cand[slot_mon[cand - 12] == slot_mon[cand + 6]]
    rng = np.random.default_rng(2026 + PC[panel])
    rows_w = []
    for m in np.unique(mon):
        ids = cand[slot_mon[cand] == m]
        if len(ids) > CAP_M:
            ids = np.sort(rng.choice(ids, CAP_M, replace=False))
        rows_w += [(int(i), 'proxy_onset', 'proxy') for i in ids]
        lo = int(np.flatnonzero(slot_mon == m)[0]); hi = int(np.flatnonzero(slot_mon == m)[-1]) + 1
        og_idx, og_cond, _ = T.select_proxy_windows(um, ctx, lo, hi, WCFG, rng)
        rows_w += [(int(i), 'proxy_ongoing', 'proxy') for i, c in zip(og_idx, og_cond) if c == 'proxy_ongoing']
    idx = T.read_window_index(rel, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(rel, panel, 'train'), idx, panel)
    origins, _ = T.resolve_origins(idx, tpl, None, panel)
    cmap = idx.set_index('window_id').condition.to_dict()
    for wid, tmn in origins.items():
        i = (int(tmn) - T0) // 5
        if 12 <= i < N_T - 6 and slot_mon[i - 12] == slot_mon[i + 6]:
            rows_w.append((i, cmap[wid], 'official'))
    W = pd.DataFrame(rows_w, columns=['Ti', 'condition', 'kind']).drop_duplicates(['Ti', 'condition', 'kind']).reset_index(drop=True)
    W['month'] = slot_mon[W.Ti.to_numpy()]
    W['gid'] = np.arange(gid_next, gid_next + len(W)); gid_next += len(W)
    W['panel'] = panel; W['family_id'] = FAM.get(panel)
    W['window_key'] = [f'{panel}|{k}|{c}|{t}' for k, c, t in zip(W.kind, W.condition, W.Ti)]
    METAS.append(W)
    Ti = W.Ti.to_numpy(np.int64); Tm = T0 + Ti * 5
    ch = chain_of(ctx); posl = np.full(ctx.L, -1, np.int64); posl[ch] = np.arange(len(ch))
    tensor = T.tensor_from_series(um, hm, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    w, l, h = T.target_rows_for_generated(len(Ti), ctx)
    lab, _ = T.labels_from_series(um, Ti, ctx.thr); y = lab[w, l, h]
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, T0)
    wmon = W.month.to_numpy()[w]; wkind = W.kind.to_numpy()[w]; g = W.gid.to_numpy()[w]
    for f in FOLDS:
        fit = mon != f
        tot = head_totals(um['spd'], ctx.thr, ch, np.repeat(fit, 288))
        hr = tot / int(fit.sum()); bn_set = set(np.argsort(-tot)[:4].tolist())
        clim = T.Climatology.from_days(P, C, d0, fit)
        nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
        X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=True)
        for k, v in nf.items():
            X[k] = v[w, l]
        hh = (h + 1) * 5 / 30.0
        X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
        X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
        X = X.astype(np.float32)
        tr = (wmon != f) & (wkind == 'proxy'); ev = wmon == f
        XT[f].append(X[tr]); YT[f].append(y[tr]); XE[f].append(X[ev]); YE[f].append(y[ev]); GE[f].append(g[ev])
        HE[f].append(h[ev]); PE[f].append(posl[l[ev]])
        del X, nf
    print(panel, W.groupby(['kind', 'condition']).size().to_dict(), 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl
META = pd.concat(METAS, ignore_index=True)

def evaluate(f, kind, pred, yE_, gE_):
    wt = META[(META.month == f) & (META.kind == kind)].reset_index(drop=True)
    loc = np.full(gid_next, -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
    m = np.isin(gE_, wt.gid.to_numpy())
    return T.score_proxy(wt[['window_key', 'panel', 'family_id', 'condition']], loc[gE_[m]], yE_[m], pred[m])

def select_k(prob, ge, cand, kmin=0, kfix=None):
    pred = np.zeros(len(prob), np.int8)
    idx = np.flatnonzero(cand)
    o = idx[np.lexsort((-prob[idx], ge[idx]))]
    gg = ge[o]
    s = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1]]); e = np.r_[s[1:], len(o)]
    p = prob[o]
    for a, b in zip(s, e):
        if kfix is not None:
            kb = int(kfix.get(int(gg[a]), 0))
        else:
            c = np.cumsum(p[a:b]); k = np.arange(1, b - a + 1)
            kb = int(np.argmax(c / (k + c[-1] - c))) + 1 if c[-1] > 0 else 0
        kb = max(kb, kmin)
        pred[o[a:a + kb]] = 1
    return pred

def fill_gaps(pred, ge, hE, pE):
    key = (ge.astype(np.int64) * 6 + hE) * 100000 + pE
    on = set(key[(pred > 0) & (pE >= 0)].tolist())
    ci = np.flatnonzero((pred == 0) & (pE >= 1))
    add = np.array([(x - 1) in on and (x + 1) in on for x in key[ci].tolist()], bool)
    out = pred.copy(); out[ci[add]] = 1
    return out

rows, agree = [], []
for f in FOLDS:
    ts = time.time()
    Xf = pd.concat(XT[f], ignore_index=True); yf = np.concatenate(YT[f]); del XT[f], YT[f]
    Xe = pd.concat(XE[f], ignore_index=True); ye = np.concatenate(YE[f]); ge = np.concatenate(GE[f]); hE = np.concatenate(HE[f]); pE = np.concatenate(PE[f])
    del XE[f], YE[f], GE[f], HE[f], PE[f]
    BASE = list(T.FEATURES); ALL = list(Xf.columns)
    known = ~np.isnan(yf)
    mr = lgb.train(PARAMS, lgb.Dataset(Xf.loc[known, BASE], label=yf[known], categorical_feature=['panel_code']), ROUNDS['R'])
    ma = lgb.train(PARAMS, lgb.Dataset(Xf.loc[known, ALL], label=yf[known], categorical_feature=['panel_code']), ROUNDS['A'])
    del Xf
    prob = (mr.predict(Xe[BASE].to_numpy(np.float32)) + ma.predict(Xe[ALL].to_numpy(np.float32))) / 2
    hc = Xe['hist_cond'].to_numpy()
    cond = META.set_index('gid').condition.reindex(ge).to_numpy()
    is_on = np.isin(cond, ONSET)
    wf = pd.DataFrame({'g': ge, 'on': is_on, 'hc_on': hc == T.HIST_COND_ONSET}).groupby('g').first()
    agree.append(dict(fold=f, windows=len(wf), true_onset=int(wf.on.sum()), guess_onset=int(wf.hc_on.sum()), mismatch=int((wf.on != wf.hc_on).sum())))
    yt = np.where(np.isnan(ye), 0, ye) > 0.5
    true_k_on = pd.Series(yt & is_on & (hE == 5)).groupby(ge).sum().to_dict()
    true_k_og = pd.Series(yt & ~is_on).groupby(ge).sum().to_dict()
    V = {}
    V['A production'] = T.thresholded(prob, hc, TH)
    b = (prob >= np.where(is_on, TH['onset_like'], TH['other'])).astype(np.int8); V['B true type cutoffs'] = b
    c = b.copy(); c[is_on & (hE < 5)] = 0; V['C B + onset nothing before T+30'] = c
    top1 = select_k(prob, ge, is_on & (hE == 5), kfix={int(k): 1 for k in np.unique(ge)})
    V['D C + onset never empty'] = np.maximum(c, top1).astype(np.int8)
    on_k = select_k(prob, ge, is_on & (hE == 5), kmin=1)
    e = np.where(is_on, on_k, b).astype(np.int8); V['E onset expected-IoU count'] = e
    og_k = select_k(prob, ge, ~is_on)
    fb = np.where(is_on, on_k, og_k).astype(np.int8); V['F expected-IoU count both'] = fb
    V['G F + fill 1-link gaps'] = fill_gaps(fb, ge, hE, pE)
    V['H E + fill 1-link gaps'] = fill_gaps(e, ge, hE, pE)
    V['ORACLE true count'] = np.where(is_on, select_k(prob, ge, is_on & (hE == 5), kfix=true_k_on), select_k(prob, ge, ~is_on, kfix=true_k_og)).astype(np.int8)
    for name, pred in V.items():
        for kind in ('proxy', 'official'):
            det = evaluate(f, kind, pred, ye, ge)['window']
            for cnd, v in det.groupby('condition').proxy_IoU.mean().items():
                dd = det[det.condition == cnd]
                rows.append(dict(fold=f, variant=name, windows=kind, condition=cnd.replace('proxy_', '').replace('queue_', ''), n=len(dd), IoU=float(v),
                                 mean_true=float(dd.n_true.mean()), mean_pred=float(dd.n_pred.mean())))
    print('fold', f, 'sec', round(time.time() - ts), flush=True)
    del Xe, mr, ma
print('WINDOW TYPE GUESS vs TRUTH'); print(pd.DataFrame(agree).to_string(index=False))
R = pd.DataFrame(rows)
S = R.groupby(['variant', 'windows', 'condition']).agg(IoU=('IoU', 'mean'), mean_true=('mean_true', 'mean'), mean_pred=('mean_pred', 'mean'), n=('n', 'sum')).round(4).reset_index()
print('PER FOLD'); print(R.round(4).to_string(index=False))
print('MEAN OVER FOLDS'); print(S.to_string(index=False))
P2 = S.pivot_table(index='variant', columns=['windows', 'condition'], values='IoU')
for k in ('proxy', 'official'):
    P2[(k, 'score')] = (P2[(k, 'onset')] + P2[(k, 'ongoing')]) / 2
print('SUMMARY (score = average of onset and ongoing)'); print(P2.round(4).to_string(), flush=True)
