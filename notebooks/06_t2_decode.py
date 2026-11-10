import glob, json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
B = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)[0]
sys.path.insert(0, B)
import t2lib as T
cfg = json.load(open(f'{B}/config.json')); sd = json.load(open(f'{B}/split_definitions.json'))
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
FAM = T.load_family_map(REL); PC = cfg['panel_codes']; USED = set(cfg['used_channels']); ELIG = cfg['elig_mode']
TH = cfg['evaluation_models']['v2']['thresholds']
first = sd['dates_used'][0]; n_days = sd['dates_used'][2]; n_t = n_days * 288
t0 = T.text_to_minutes(first + 'T00:00:00Z')
dates_by_part = {p['name']: p['dates'] for p in sd['partitions']}
fit_days = np.zeros(n_days, bool)
for d in dates_by_part['fit']:
    fit_days[(pd.Timestamp(d) - pd.Timestamp(first)).days] = True
EVAL = ['tune', 'test', 'holdout', 'offtest']
cat = pd.read_csv(f'{B}/metrics/generated_windows_catalog.csv')
cat = cat[cat.partition.isin(EVAL)].reset_index(drop=True)
cat['Tmin'] = [T.text_to_minutes(x) for x in cat['T']]
cat = cat[(cat.Tmin - t0) // 5 >= 12].reset_index(drop=True)
cat['gid'] = np.arange(len(cat))
print('eval windows', len(cat), cat.groupby('partition').size().to_dict(), flush=True)
booster = lgb.Booster(model_file=f'{B}/lgbm_eval_v2.txt')
store = {'u': [], 'm': []}; Y = []; G = []; HC = []
for panel in cfg['panels']:
    ts = time.time()
    rel = Path(REL)
    fm = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked')
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    cm = T.resolve_colmap(T.parquet_columns(fm[0]), 'm'); cu = T.resolve_colmap(T.parquet_columns(fu[0]), 'u')
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(fm, cm, USED, ELIG, ctx, t0, n_t, 'masked')
    um, _ = T.load_series(fu, cu, USED, ELIG, ctx, t0, n_t, 'unmasked')
    sel = cat[cat.panel == panel]
    Tm = sel.Tmin.to_numpy(np.int64); Ti = (Tm - t0) // 5
    tensor = T.tensor_from_series(hm, hm, Ti, True)
    wl = T.build_wl_features(tensor, ctx, Tm)
    w, l, h = T.target_rows_for_generated(len(sel), ctx)
    lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    Y.append(lab[w, l, h]); G.append(sel.gid.to_numpy()[w])
    for tag, src in (('u', um), ('m', hm)):
        P, C, d0 = T.Climatology.day_arrays(src['spd'], ctx.thr, t0)
        clim = T.Climatology.from_days(P, C, d0, fit_days)
        X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0])
        store[tag].append(booster.predict(X[T.FEATURES].to_numpy(np.float32), num_threads=4))
        if tag == 'u':
            HC.append(X['hist_cond'].to_numpy())
    print(panel, 'windows', len(sel), 'rows', len(w), 'sec', round(time.time() - ts), flush=True)
y = np.concatenate(Y); gid = np.concatenate(G); hc = np.concatenate(HC)
probs = {k: np.concatenate(v) for k, v in store.items()}
parts = {}
for p in EVAL:
    wt = cat[cat.partition == p]
    loc = np.full(len(cat), -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
    m = np.isin(gid, wt.gid.to_numpy())
    parts[p] = (wt[['window_key', 'panel', 'family_id', 'condition']].reset_index(drop=True), m, loc[gid[m]])
def score(p, pred, detail=False):
    wt, m, rw = parts[p]
    r = T.score_proxy(wt, rw, y[m], pred[m])
    return r if detail else r['overall']
REF = {'tune': 0.763602, 'test': 0.764779, 'holdout': 0.779091, 'offtest': 0.789228}
for tag in ('u', 'm'):
    d0 = T.thresholded(probs[tag], hc, TH)
    print('BASELINE clim from', tag, {p: round(score(p, d0), 4) for p in EVAL}, 'ref', REF, flush=True)
tag = 'u' if abs(score('test', T.thresholded(probs['u'], hc, TH)) - REF['test']) <= abs(score('test', T.thresholded(probs['m'], hc, TH)) - REF['test']) else 'm'
print('USING clim variant', tag, flush=True)
prob = probs[tag]; order = np.lexsort((-prob, gid)); ps = prob[order]; ws = gid[order]
starts = np.r_[0, np.flatnonzero(np.diff(ws)) + 1]; counts = np.diff(np.r_[starts, len(ws)])
rank = np.arange(len(ws)) - np.repeat(starts, counts) + 1
def put(sorted_mask):
    out = np.zeros(len(prob), np.int8); out[order[sorted_mask]] = 1; return out
def jdecode(pw):
    p = ps ** pw if pw != 1 else ps
    cs = np.cumsum(p); base = np.repeat(np.r_[0.0, cs[starts[1:] - 1]], counts)
    A = cs - base; tot = np.repeat(np.add.reduceat(p, starts), counts)
    R = A / (rank + tot - A); mx = np.repeat(np.maximum.reduceat(R, starts), counts)
    kstar = np.minimum.reduceat(np.where(R >= mx - 1e-12, rank, 1 << 30), starts)
    return put(rank <= np.repeat(kstar, counts))
def fallback(d0, kmin, floor=0.0):
    zero = np.repeat(np.add.reduceat(d0[order], starts) == 0, counts)
    return np.maximum(d0, put((rank <= kmin) & zero & (ps >= floor)))
D0 = T.thresholded(prob, hc, TH)
dec = {'D0 current': D0, 'J jaccard': jdecode(1.0), 'J|D0': np.maximum(jdecode(1.0), D0)}
for kmin in (1, 2, 3, 5, 8): dec[f'D0+top{kmin}'] = fallback(D0, kmin)
for floor in (0.02, 0.05, 0.1): dec[f'D0+top3 p>={floor}'] = fallback(D0, 3, floor)
rows = []
for name, pr in dec.items():
    s = {p: score(p, pr) for p in EVAL}; s['avg_test_hold_off'] = np.mean([s['test'], s['holdout'], s['offtest']]); s['name'] = name; rows.append(s)
print(pd.DataFrame(rows).set_index('name').round(4).to_string(), flush=True)
det = score('test', D0, True)['window']
det['zero_pred'] = det.n_pred == 0
print('TEST windows with zero predicted cells by condition:'); print(det.groupby('condition').agg(n=('proxy_IoU', 'size'), zero=('zero_pred', 'sum'), iou=('proxy_IoU', 'mean')).to_string(), flush=True)
print('mean IoU of zero-pred windows', det[det.zero_pred].proxy_IoU.mean(), 'n_true mean', det[det.zero_pred].n_true.mean(), flush=True)
