import glob, sys, time, numpy as np, pandas as pd, lightgbm as lgb
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
VT0 = T.text_to_minutes('2031-03-01T00:00:00Z'); VNT = 31 * 288
WCFG = {'min_history_coverage': 0.7, 'max_persistence_iou': 0.9, 'min_spacing_minutes': 360, 'max_windows_per_day_condition': 2}
PARAMS = {'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255, 'num_threads': 4, 'seed': 2026,
          'deterministic': True, 'force_col_wise': True, 'verbose': -1}
ROUNDS = 320
CUTS = {'cutoffs 0.30/0.25': {'onset_like': 0.3, 'other': 0.25}, 'cutoffs 0.40/0.45': {'onset_like': 0.4, 'other': 0.45}}
ONSET_CAP = 600

def make_X(hist_s, t_s, Ti, Tm, ctx, ch, bn_set, hr, clim, lodo):
    tensor = T.tensor_from_series(hist_s, t_s, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    w, l, h = T.target_rows_for_generated(len(Ti), ctx)
    nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
    X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=lodo)
    for k, v in nf.items():
        X[k] = v[w, l]
    hh = (h + 1) * 5 / 30.0
    X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
    X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
    return X.astype(np.float32), w

FIT = {'u': [], 'm': []}; FY = []
EV = {'u': [], 'm': []}; EY = []; EG = []
VX = []; VY = []; VG = []
METAS = []; CHK = []
gid_next = 0
for panel in PANELS:
    ts = time.time(); rel = Path(REL)
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    hm, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states_masked'), LC, USED, ELIG, ctx, T0, N_T, 'masked')
    um, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states'), LC, USED, ELIG, ctx, T0, N_T, 'unmasked')
    hv, _ = T.load_series(T.parquet_files(rel / 'corridors' / panel / 'validation' / 'mainline_states_masked'), LC, USED, ELIG, ctx, VT0, VNT, 'masked')
    tm = ctx.target_mask; nT = int(tm.sum())
    idx = T.read_window_index(rel, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(rel, panel, 'train'), idx, panel)
    org, _ = T.resolve_origins(idx, tpl, None, panel); cmap = idx.set_index('window_id').condition.to_dict()
    off = [((int(t) - T0) // 5, cmap[w]) for w, t in org.items() if 12 <= (int(t) - T0) // 5 < N_T - 6]
    evday = np.zeros(N_DAYS, bool); evday[np.arange(N_DAYS) % 7 == 3] = True
    for i, _ in off:
        evday[[(i - 12) // 288, i // 288, (i + 6) // 288]] = True
    idv = T.read_window_index(rel, panel, 'validation', FAM)
    tpv = T.restrict_template(T.read_template(rel, panel, 'validation'), idv, panel)
    orv, _ = T.resolve_origins(idv, tpv, None, panel); cmv = idv.set_index('window_id').condition.to_dict()
    vw = [((int(t) - VT0) // 5, cmv[w]) for w, t in orv.items() if 12 <= (int(t) - VT0) // 5 < VNT - 6]
    oi = np.array([i for i, _ in off]); vi = np.array([i for i, _ in vw])
    hs = oi[:, None] + np.arange(-12, 0); hsv = vi[:, None] + np.arange(-12, 0); fs = vi[:, None] + np.arange(1, 7)
    CHK.append(dict(panel=panel, train_hist_nan_unmasked=round(float(np.isnan(um['spd'][hs][:, :, tm]).mean()), 4),
                    train_hist_nan_masked=round(float(np.isnan(hm['spd'][hs][:, :, tm]).mean()), 4),
                    val_hist_nan_masked=round(float(np.isnan(hv['spd'][hsv][:, :, tm]).mean()), 4),
                    val_future_present_masked=round(float((~np.isnan(hv['spd'][fs][:, :, tm])).mean()), 4)))
    q = (um['qoff'][:, tm] > 0).any(1); cs = np.concatenate([[0], np.cumsum(q)])
    Tx = np.arange(12, N_T - 6)
    ccs = np.concatenate([[0], np.cumsum((um['nst'][:, tm] > 0).sum(1))])
    cov = (ccs[Tx] - ccs[Tx - 12]) / (12.0 * nT)
    cand = Tx[((cs[Tx + 6] - cs[Tx - 12]) == 0) & q[Tx + 6] & (cov >= 0.7)]
    rng = np.random.default_rng(2026 + PC[panel])
    og_idx, og_cond, _ = T.select_proxy_windows(um, ctx, 0, N_T, WCFG, rng)
    og = og_idx[np.array(og_cond) == 'proxy_ongoing']

    def part(i):
        a, b = evday[(i - 12) // 288], evday[(i + 6) // 288]
        return np.where(a & b, 'proxy_eval', np.where(~a & ~b, 'proxy_fit', 'drop'))
    rows_w = []
    for arr, c in ((cand, 'proxy_onset'), (og, 'proxy_ongoing')):
        p = part(arr)
        for k in ('proxy_fit', 'proxy_eval'):
            s = arr[p == k]
            cap = ONSET_CAP if k == 'proxy_fit' else ONSET_CAP // 3
            if c == 'proxy_onset' and len(s) > cap:
                s = np.sort(rng.choice(s, cap, replace=False))
            rows_w += [(int(i), c, k) for i in s]
    rows_w += [(int(i), c, 'off_train') for i, c in off]
    W = pd.DataFrame(rows_w, columns=['Ti', 'condition', 'kind']).drop_duplicates().reset_index(drop=True)
    V = pd.DataFrame([(int(i), c, 'off_val') for i, c in vw], columns=['Ti', 'condition', 'kind'])
    for D in (W, V):
        D['gid'] = np.arange(gid_next, gid_next + len(D)); gid_next += len(D)
        D['panel'] = panel; D['family_id'] = FAM.get(panel)
        D['window_key'] = [f'{panel}|{k}|{c}|{t}' for k, c, t in zip(D.kind, D.condition, D.Ti)]
        METAS.append(D)
    ch = chain_of(ctx)
    fitday = ~evday
    tot = head_totals(um['spd'], ctx.thr, ch, np.repeat(fitday, 288))
    hr = tot / int(fitday.sum()); bn_set = set(np.argsort(-tot)[:4].tolist())
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, T0); clim = T.Climatology.from_days(P, C, d0, fitday)
    Ti = W.Ti.to_numpy(np.int64); Tm = T0 + Ti * 5
    lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    for key, src in (('u', um), ('m', hm)):
        X, w = make_X(src, hm, Ti, Tm, ctx, ch, bn_set, hr, clim, True)
        if key == 'u':
            _, l, h = T.target_rows_for_generated(len(Ti), ctx); y = lab[w, l, h]
            kind = W.kind.to_numpy()[w]; g = W.gid.to_numpy()[w]
            fi = kind == 'proxy_fit'; ev = (kind == 'proxy_eval') | (kind == 'off_train')
            FY.append(y[fi]); EY.append(y[ev]); EG.append(g[ev])
        FIT[key].append(X[fi]); EV[key].append(X[ev])
        del X
    Tv = V.Ti.to_numpy(np.int64)
    Xv, wv = make_X(hv, hv, Tv, VT0 + Tv * 5, ctx, ch, bn_set, hr, clim, False)
    _, lv, hvv = T.target_rows_for_generated(len(Tv), ctx)
    labv, _ = T.labels_from_series(hv, Tv, ctx.thr)
    VX.append(Xv); VY.append(labv[wv, lv, hvv]); VG.append(V.gid.to_numpy()[wv])
    print(panel, W.groupby(['kind', 'condition']).size().to_dict(), 'val windows', len(V), 'sec', round(time.time() - ts), flush=True)
    del hm, um, hv
print('INPUT CHECK'); print(pd.DataFrame(CHK).to_string(index=False), flush=True)
META = pd.concat(METAS, ignore_index=True)
fy = np.concatenate(FY); ey = np.concatenate(EY); eg = np.concatenate(EG)
vy = np.concatenate(VY); vg = np.concatenate(VG); Xv = pd.concat(VX, ignore_index=True); del VX

def evaluate(kind, pred, y, g):
    wt = META[META.kind == kind].reset_index(drop=True)
    loc = np.full(gid_next, -1); loc[wt.gid.to_numpy()] = np.arange(len(wt))
    m = np.isin(g, wt.gid.to_numpy())
    return T.score_proxy(wt[['window_key', 'panel', 'family_id', 'condition']], loc[g[m]], y[m], pred[m])['window']

rows, MODELS = [], {}
for key in ('u', 'm'):
    ts = time.time()
    Xf = pd.concat(FIT[key], ignore_index=True); FIT[key] = None
    known = ~np.isnan(fy)
    MODELS[key] = lgb.train(PARAMS, lgb.Dataset(Xf.loc[known], label=fy[known], categorical_feature=['panel_code']), ROUNDS)
    cols = list(Xf.columns); del Xf
    print('model trained on', 'clean' if key == 'u' else 'blanked', 'history', 'sec', round(time.time() - ts), flush=True)
EVX = {k: pd.concat(EV[k], ignore_index=True) for k in ('u', 'm')}
for mk, mname in (('u', 'trained on clean history'), ('m', 'trained on blanked history')):
    for iname, X, y, g, kinds in (('clean inputs', EVX['u'], ey, eg, ('proxy_eval', 'off_train')),
                                  ('blanked inputs', EVX['m'], ey, eg, ('proxy_eval', 'off_train')),
                                  ('blanked inputs', Xv, vy, vg, ('off_val',))):
        prob = MODELS[mk].predict(X[cols].to_numpy(np.float32))
        hc = X['hist_cond'].to_numpy()
        for cname, th in CUTS.items():
            pred = T.thresholded(prob, hc, th)
            for kind in kinds:
                det = evaluate(kind, pred, y, g)
                r = dict(model=mname, inputs=iname, cutoffs=cname, windows=kind)
                for cnd, v in det.groupby('condition').proxy_IoU.mean().items():
                    c = 'onset' if 'onset' in cnd else 'ongoing'
                    r[c] = round(float(v), 4); r['n_' + c] = int((det.condition == cnd).sum())
                r['score'] = round((r.get('onset', np.nan) + r.get('ongoing', np.nan)) / 2, 4)
                r['labelled_cells'] = int(det.n_eligible.sum())
                rows.append(r)
R = pd.DataFrame(rows)
print('RESULTS (off_val = official validation windows, labels from the masked layer where present)')
print(R.to_string(index=False), flush=True)
