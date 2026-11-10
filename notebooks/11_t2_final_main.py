import json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, '/kaggle/working')
import t2lib as T
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
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
WCFG = {'min_history_coverage': 0.7, 'max_persistence_iou': 0.9, 'min_spacing_minutes': 360, 'max_windows_per_day_condition': 2}
PARAMS = {'objective': 'binary', 'learning_rate': 0.1, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255, 'num_threads': 4, 'seed': 2026,
          'deterministic': True, 'force_col_wise': True, 'verbose': -1}
ROUNDS = {'R': 340, 'A': 320}
THRESHOLDS = {'onset_like': 0.3, 'other': 0.25}
ONSET_CAP = 700
ALL_DAYS = np.ones(N_DAYS, bool)
day_rows = np.repeat(ALL_DAYS, 288)
STATE, XT, YT = {}, [], []
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
    onset = Tx[quiet & first_jam & (cov >= 0.7)]
    rng = np.random.default_rng(2026 + PC[panel])
    if len(onset) > ONSET_CAP:
        onset = np.sort(rng.choice(onset, ONSET_CAP, replace=False))
    og_idx, og_cond, _ = T.select_proxy_windows(um, ctx, 0, N_T, WCFG, rng)
    ongoing = og_idx[np.array(og_cond) == 'proxy_ongoing']
    Ti = np.concatenate([onset, ongoing]).astype(np.int64)
    Tm = T0 + Ti * 5
    ch = chain_of(ctx); tot = head_totals(um['spd'], ctx.thr, ch, day_rows)
    hr = tot / N_DAYS; bn_set = set(np.argsort(-tot)[:4].tolist())
    P, C, d0 = T.Climatology.day_arrays(um['spd'], ctx.thr, T0); clim = T.Climatology.from_days(P, C, d0, ALL_DAYS)
    tensor = T.tensor_from_series(um, hm, Ti, True); wl = T.build_wl_features(tensor, ctx, Tm)
    nf = new_features(wl, tensor, ctx, ch, bn_set, hr, T._last_valid)
    w, l, h = T.target_rows_for_generated(len(Ti), ctx)
    X = T.rows_frame(wl, w, l, (h + 1) * 5, ctx.panel_code, clim=clim, T_min=Tm, seq_dn1=ctx.seq_dn[:, 0], lodo=True)
    for k, v in nf.items():
        X[k] = v[w, l]
    hh = (h + 1) * 5 / 30.0
    X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
    X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
    lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    XT.append(X); YT.append(lab[w, l, h])
    STATE[panel] = {'ctx': ctx, 'ch': ch, 'bn': bn_set, 'hr': hr, 'clim': clim}
    print(panel, 'onset windows', len(onset), 'ongoing windows', len(ongoing), 'rows', len(X), 'sec', round(time.time() - ts), flush=True)
    del hm, um, tensor, wl, nf
XF = pd.concat(XT, ignore_index=True); yF = np.concatenate(YT); del XT, YT
BASE = list(T.FEATURES); NEWF = [c for c in XF.columns if c not in BASE]; ALL = BASE + NEWF
known = ~np.isnan(yF)
print('train rows', int(known.sum()), 'positive share', round(float(yF[known].mean()), 4), 'features', len(BASE), '+', len(NEWF), flush=True)
def dataset(cols):
    return lgb.Dataset(XF.loc[known, cols], label=yF[known], categorical_feature=['panel_code'])
t1 = time.time()
model_r = lgb.train(PARAMS, dataset(BASE), ROUNDS['R'])
model_a = lgb.train(PARAMS, dataset(ALL), ROUNDS['A'])
print('trained two models in', round(time.time() - t1), 'sec', flush=True)
MODELS = OUT / 'task2_models'; MODELS.mkdir(exist_ok=True)
model_r.save_model(str(MODELS / 'lgbm_base_features.txt')); model_a.save_model(str(MODELS / 'lgbm_structure_features.txt'))
(MODELS / 'config.json').write_text(json.dumps({'thresholds': THRESHOLDS, 'rounds': ROUNDS, 'params': PARAMS, 'features': ALL, 'base_features': BASE}, indent=2))
del XF
def predict_split(split):
    outs, templates, meta = [], [], []
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
        for k, v in nf.items():
            X[k] = v[o['w_idx'], o['l_idx']]
        hh = (o['h_idx'] + 1) * 5 / 30.0
        X['jd_dn_proj'] = X['jd_dn_now'] + X['jd_dn_d30'] * hh
        X['jd_up_proj'] = X['jd_up_now'] + X['jd_up_d30'] * hh
        prob = (model_r.predict(X[BASE].to_numpy(np.float32)) + model_a.predict(X[ALL].to_numpy(np.float32))) / 2
        if not np.isfinite(prob).all():
            raise ValueError(f'{panel}/{split}: non-finite probabilities')
        pred = T.thresholded(prob, X['hist_cond'].to_numpy(), THRESHOLDS)
        base = tpl[['window_id', 'timestamp', 'link_id']].reset_index(drop=True)
        outs.append(base.assign(queue_pred=pred)); templates.append(base)
        meta.append(pd.DataFrame({'panel': panel, 'horizon': (o['h_idx'] + 1) * 5, 'pred': pred, 'onset_like': (X['hist_cond'].to_numpy() == 0).astype(int)}))
        print(split, panel, 'rows', len(base), 'predicted queued', int(pred.sum()), flush=True)
    out = pd.concat(outs, ignore_index=True); out['queue_pred'] = out['queue_pred'].astype(np.int64)
    tpl_all = pd.concat(templates, ignore_index=True)
    T.validate_queue_output(out, tpl_all, split)
    return out, tpl_all, pd.concat(meta, ignore_index=True)
RESULT = {}
for split in ('validation', 'private'):
    out, tpl_all, meta = predict_split(split)
    RESULT[split] = out
    out.to_csv(OUT / f'queue_{split}.csv', index=False)
    print(f'{split}: rows {len(out):,} positives {int(out.queue_pred.sum()):,} share {out.queue_pred.mean():.4f}')
    print('queued share by horizon, onset-like windows:', meta[meta.onset_like == 1].groupby('horizon').pred.mean().round(4).to_dict())
    print('queued share by horizon, other windows:', meta[meta.onset_like == 0].groupby('horizon').pred.mean().round(4).to_dict(), flush=True)
combined = pd.concat([RESULT['validation'], RESULT['private']], ignore_index=True)
if set(RESULT['validation'].window_id) & set(RESULT['private'].window_id):
    raise ValueError('validation and private share window ids')
combined.to_csv(OUT / 'queue.csv', index=False)
back = pd.read_csv(OUT / 'queue.csv', dtype=str, keep_default_na=False)
assert list(back.columns) == T.OUTPUT_COLUMNS and len(back) == 174000 and back.queue_pred.isin(['0', '1']).all()
print('queue.csv rows', len(back))
