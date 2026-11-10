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
    CJ, CO = rec_tables(parts, ctx.thr); del parts
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
    X = X.astype(np.float32)
    X['_gid'] = W.gid.to_numpy()[w]; X['_h'] = h; X['_day'] = Ti[w] // 288
    X['_on'] = (W.condition.to_numpy()[w] == 'proxy_onset')
    lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    XT.append(X); YT.append(lab[w, l, h])
    STATE[panel] = {'ctx': ctx, 'ch': ch, 'bn': bn_set, 'hr': hr, 'clim': clim, 'CJ': CJ, 'CO': CO}
    print(panel, 'onset', len(onset), 'ongoing', len(ongoing), 'rows', len(X), 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl, nf
XF = pd.concat(XT, ignore_index=True); yF = np.concatenate(YT); del XT, YT
META = pd.concat(META, ignore_index=True)
AUX = ['_gid', '_h', '_day', '_on']
BASE = list(T.FEATURES); REC = [c for c in XF.columns if c.startswith('rec')]
NEWF = [c for c in XF.columns if c not in BASE and c not in REC and c not in AUX]
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
del mr, ma, mo, XE
t1 = time.time()
model_r = fit(CR, known, ROUNDS['R']); model_a = fit(CA, known, ROUNDS['A'])
model_on = fit_onset(CA, known) if ONSET_MODE != 'shared' else None
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
        prob = (model_r.predict(X[CR].to_numpy(np.float32)) + model_a.predict(X[CA].to_numpy(np.float32))) / 2
        if not np.isfinite(prob).all():
            raise ValueError(f'{panel}/{split}: non-finite probabilities')
        cmap = idx.assign(window_id=idx.window_id.astype(str)).set_index('window_id').condition
        wids = np.asarray(o['window_ids'], dtype=object)[o['w_idx']]
        is_on = cmap.reindex(wids.astype(str)).to_numpy() == 'queue_onset'
        if model_on is not None:
            cell = is_on & (np.asarray(o['h_idx']) == 5)
            if cell.any():
                p_on = onset_prob(model_on, X.loc[cell], CA)
                prob[cell] = p_on if ONSET_MODE == 'only' else (1 - ON_BLEND) * prob[cell] + ON_BLEND * p_on
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
