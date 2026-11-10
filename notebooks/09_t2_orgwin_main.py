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
part_of_day = np.full(n_days, '', dtype=object)
for p in sd['partitions']:
    for d in p['dates']:
        part_of_day[(pd.Timestamp(d) - pd.Timestamp(first)).days] = p['name']
fit_days = part_of_day == 'fit'; day_rows = np.repeat(fit_days, 288)
EVAL = ['tune', 'test', 'holdout', 'offtest', 'official']
CAP = {'fit': 500, 'tune': 60, 'test': 60, 'holdout': 60, 'offtest': 40}
cat = pd.read_csv(f'{B}/metrics/generated_windows_catalog.csv'); cat['Tmin'] = [T.text_to_minutes(x) for x in cat['T']]
XF, YF, XE, YE, GE, META = [], [], [], [], [], []
gid_next = 0
for panel in cfg['panels']:
    ts = time.time(); rel = Path(REL)
    fm = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked')
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    cm = T.resolve_colmap(T.parquet_columns(fm[0]), 'm'); cu = T.resolve_colmap(T.parquet_columns(fu[0]), 'u')
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(fm, cm, USED, ELIG, ctx, t0, n_t, 'masked')
    um, _ = T.load_series(fu, cu, USED, ELIG, ctx, t0, n_t, 'unmasked')
    tm = ctx.target_mask; nT = int(tm.sum())
    q = (um['qoff'][:, tm] > 0).any(1); cs = np.concatenate([[0], np.cumsum(q)])
    Tx = np.arange(12, n_t - 6)
    nojam = (cs[Tx + 6] - cs[Tx - 12]) == 0; jam30 = q[Tx + 6]
    ccs = np.concatenate([[0], np.cumsum((um['nst'][:, tm] > 0).sum(1))])
    cov = (ccs[Tx] - ccs[Tx - 12]) / (12.0 * nT)
    cand = Tx[nojam & jam30 & (cov >= 0.7)]
    same = part_of_day[(cand - 12) // 288] == part_of_day[(cand + 6) // 288]
    cand = cand[same]; pc = part_of_day[cand // 288]
    rng = np.random.default_rng(2026 + PC[panel]); rows_w = []
    for part, cap in CAP.items():
        ids = cand[pc == part]
        if len(ids) > cap:
            ids = np.sort(rng.choice(ids, cap, replace=False))
        for i in ids:
            rows_w.append((t0 + int(i) * 5, 'proxy_onset', part))
    og = cat[(cat.panel == panel) & (cat.condition == 'proxy_ongoing')]
    for _, r in og.iterrows():
        if (r.Tmin - t0) // 5 >= 12:
            rows_w.append((int(r.Tmin), 'proxy_ongoing', r.partition))
    idx = T.read_window_index(rel, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(rel, panel, 'train'), idx, panel)
    origins, _ = T.resolve_origins(idx, tpl, None, panel)
    cmap = idx.set_index('window_id').condition.to_dict(); nofit = 0
    for wid, tmn in origins.items():
        if fit_days[(int(tmn) - t0) // 1440]:
            nofit += 1
            continue
        rows_w.append((int(tmn), cmap[wid], 'official'))
    W = pd.DataFrame(rows_w, columns=['Tmin', 'condition', 'partition'])
    W = W[(W.Tmin - t0) // 5 >= 12].reset_index(drop=True)
    W['gid'] = np.arange(gid_next, gid_next + len(W)); gid_next += len(W)
    W['panel'] = panel; W['family_id'] = FAM.get(panel)
    W['window_key'] = [f'{panel}|{p}|{c}|{t}' for p, c, t in zip(W.partition, W.condition, W.Tmin)]
    ch = chain_of(ctx); tot = head_totals(um['spd'], ctx.thr, ch, day_rows)
    hr = tot / max(int(fit_days.sum()), 1); bn_set = set(np.argsort(-tot)[:4].tolist())
    Tm = W.Tmin.to_numpy(np.int64); Ti = (Tm - t0) // 5
    tensor = T.tensor_from_series(um, hm, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, t0); clim = T.Climatology.from_days(P, C, d0, fit_days)
    w, l, h = T.target_rows_for_generated(len(W), ctx)
    X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=True)
    for k, v in nf.items():
        X[k] = v[w, l]
    hh = (h + 1) * 5 / 30.0
    X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
    X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
    lab, _ = T.labels_from_series(um, Ti, ctx.thr); y = lab[w, l, h]
    isfit = W.partition.to_numpy()[w] == 'fit'; g = W.gid.to_numpy()[w]
    XF.append(X[isfit]); YF.append(y[isfit]); XE.append(X[~isfit]); YE.append(y[~isfit]); GE.append(g[~isfit]); META.append(W)
    print(panel, 'windows', W.groupby(['partition', 'condition']).size().to_dict(), 'official in fit days skipped', nofit, 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl, nf, X
META = pd.concat(META, ignore_index=True); XF = pd.concat(XF, ignore_index=True); XE = pd.concat(XE, ignore_index=True)
yF = np.concatenate(YF); yE = np.concatenate(YE); gE = np.concatenate(GE)
hcE = XE['hist_cond'].to_numpy(); hcF = XF['hist_cond'].to_numpy()
BASE = list(T.FEATURES); NEWF = [c for c in XF.columns if c not in BASE]; ALL = BASE + NEWF
print('train rows', len(XF), 'positive share', round(float(np.nanmean(yF)), 4), 'eval rows', len(XE), flush=True)
parts = {}
for p in EVAL:
    wt = META[META.partition == p]; loc = np.full(len(META), -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
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
def fit_model(name, cols):
    ts = time.time(); cat_f = [c for c in cols if c == 'panel_code']
    dtr = lgb.Dataset(XF.loc[knownF, cols], label=yF[knownF], categorical_feature=cat_f)
    v = tuneM & knownE
    dva = lgb.Dataset(XE.loc[v, cols], label=yE[v], reference=dtr)
    bst = lgb.train(params, dtr, 1200, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    print(name, 'rounds', bst.best_iteration, 'sec', round(time.time() - ts), flush=True)
    return bst, bst.predict(XE[cols].to_numpy(np.float32), num_iteration=bst.best_iteration, num_threads=4)
p_v2 = lgb.Booster(model_file=f'{B}/lgbm_eval_v2.txt').predict(XE[BASE].to_numpy(np.float32), num_threads=4)
bR, p_R = fit_model('R_base_feats_new_windows', BASE)
bA, p_A = fit_model('A_new_feats_new_windows', ALL)
cands = {'V2 old windows, his cutoffs': (p_v2, TH), 'V2 old windows, retuned': (p_v2, None), 'R base feats, new windows': (p_R, None),
         'A new feats, new windows': (p_A, None), 'avg(R, A)': ((p_R + p_A) / 2, None)}
rows = []
for name, (pr, th) in cands.items():
    th = th or tune_th(pr); pred = T.thresholded(pr, hcE, th)
    r = {p: round(score(p, pred), 4) for p in EVAL}
    det = pd.concat([score(p, pred, True)['window'] for p in ('test', 'holdout', 'offtest')])
    bc = det.groupby('condition').proxy_IoU.mean()
    do = score('official', pred, True)['window']; oc = do.groupby('condition').proxy_IoU.mean()
    r.update(onset3=round(bc.get('proxy_onset', np.nan), 4), ongoing3=round(bc.get('proxy_ongoing', np.nan), 4),
             off_onset=round(oc.get('queue_onset', np.nan), 4), off_ongoing=round(oc.get('queue_ongoing', np.nan), 4), th=str(th))
    r['name'] = name; rows.append(r)
print(pd.DataFrame(rows).set_index('name').to_string(), flush=True)
print('official windows scored:', parts['official'][0].condition.value_counts().to_dict())
imp = pd.Series(bA.feature_importance('gain'), index=ALL).sort_values(ascending=False)
print('TOP 25 gain (model A):'); print((imp / imp.sum()).head(25).round(4).to_string())
print('share of gain from new features', round(float(imp[NEWF].sum() / imp.sum()), 3), flush=True)
