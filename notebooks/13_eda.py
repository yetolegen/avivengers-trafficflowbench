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
import glob, json, subprocess, sys, numpy as np, pandas as pd
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 120)
TP = glob.glob('/kaggle/input/**/t2lib.py', recursive=True)[0]
sys.path.insert(0, str(Path(TP).parent))
import t2lib as T
PANELS = ['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
PC = {p: i for i, p in enumerate(PANELS)}
USED = {'speed', 'flow', 'occ', 'pct', 'elig'}
ELIG = 'column'
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
print('=================== TASK 4 ===================', flush=True)
try:
    r = subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jacky850/trafficflowbench-public.git', '/tmp/tfb'], capture_output=True, text=True)
    print(r.stderr[-200:])
    root = Path('/tmp/tfb')
    print('commit', subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
    for p in sorted(root.rglob('*')):
        s = str(p.relative_to(root)).replace('\\', '/')
        if p.is_file() and not s.startswith('.git/') and ('task4' in s.lower() or s.endswith('.md')):
            print('FILE', s, p.stat().st_size)
    for p in sorted(root.rglob('*.md')):
        txt = p.read_text(errors='ignore')
        if 'ODME' in txt or 'Task 4' in txt or 'task 4' in txt:
            print('\n######## DOC', p.relative_to(root)); print(txt[:6000])
    for name, cap in (('src/task4/score_task4.py', 16000), ('src/task4/build_task4_odme_artifacts.py', 10000)):
        f = root / name
        if f.exists():
            print('\n######## SRC', name); print(f.read_text()[:cap])
except Exception as e:
    print('task4 repo part failed', repr(e))
try:
    for panel in ('D12_I5_N',):
        p = Path(REL) / 'task4' / panel / 'validation'
        for f in sorted(p.glob('*.csv')):
            d = pd.read_csv(f); print('\n#### DATA', panel, f.name, d.shape); print(d.head(5).to_string())
    for panel in sorted(x.name for x in (Path(REL) / 'task4').iterdir() if x.is_dir()):
        sizes = {}
        for split in ('validation', 'private'):
            f = Path(REL) / 'task4' / panel / split / 'sample_submission_path_flow.csv'
            sizes[split] = len(pd.read_csv(f)) if f.exists() else None
        print(panel, sizes)
except Exception as e:
    print('task4 data part failed', repr(e))
print('\n=================== TASK 2 EDA ===================', flush=True)
FAM = T.load_family_map(REL)
first = '2030-06-01'; n_days = 273; n_t = n_days * 288; t0 = T.text_to_minutes(first + 'T00:00:00Z')
rows = []
for panel in PANELS:
    for split in ('train', 'validation', 'private'):
        idx = T.read_window_index(REL, panel, split, FAM)
        tpl = T.restrict_template(T.read_template(REL, panel, split), idx, 'x')
        org, _ = T.resolve_origins(idx, tpl, None, 'x')
        cm = idx.set_index('window_id').condition.to_dict()
        for w, m in org.items():
            rows.append((panel, split, cm.get(w), m))
W = pd.DataFrame(rows, columns=['panel', 'split', 'cond', 'm'])
W['ts'] = pd.to_datetime(W.m * 60, unit='s', utc=True); W['hour'] = W.ts.dt.hour + W.ts.dt.minute / 60; W['dow'] = W.ts.dt.dayofweek
print('\nWINDOWS BY SPLIT AND CONDITION')
print(W.groupby(['split', 'cond']).agg(n=('m', 'size'), days=('ts', lambda s: s.dt.date.nunique()), hour_min=('hour', 'min'), hour_med=('hour', 'median'), hour_max=('hour', 'max')).round(2).to_string())
print(W.groupby(['split', 'cond']).dow.value_counts().unstack(fill_value=0).to_string())
off_days = set(((W[W.split == 'train'].m - t0) // 1440).astype(int).tolist())
last_fit = (pd.Timestamp('2030-12-06') - pd.Timestamp(first)).days
pod = np.array(['fit' if (d <= last_fit and d not in off_days) else 'other' for d in range(n_days)], dtype=object)
ONS, ONG, EV_HOUR = [], [], []
for panel in PANELS:
    rel = Path(REL)
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    cu = T.resolve_colmap(T.parquet_columns(fu[0]), 'u')
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    um, _ = T.load_series(fu, cu, USED, ELIG, ctx, t0, n_t, 'u')
    tm = ctx.target_mask; ch = chain_of(ctx); chT = np.array([c for c in ch if tm[c]]); nT = len(chT)
    Q = um['qoff'][:, chT] > 0
    q_any = Q.any(1); cs = np.concatenate([[0], np.cumsum(q_any)])
    Tx = np.arange(12, n_t - 13)
    cand = Tx[((cs[Tx + 6] - cs[Tx - 12]) == 0) & q_any[Tx + 6]]
    cand = cand[pod[(cand - 12) // 288] == pod[(cand + 6) // 288]]
    fitc = cand[pod[cand // 288] == 'fit']
    prior = Q[fitc + 6].sum(0).astype(float)
    order = np.argsort(-prior); pct = pd.Series(prior).rank(ascending=False, method='average').to_numpy() / nT
    for i in fitc:
        EV_HOUR.append(((i % 288) * 5) / 60.0)
    idx = T.read_window_index(rel, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(rel, panel, 'train'), idx, 'x')
    org, _ = T.resolve_origins(idx, tpl, None, 'x')
    cm = idx.set_index('window_id').condition.to_dict()
    for wid, m in org.items():
        Ti = (m - t0) // 5; cond = cm[wid]
        if 'onset' in cond:
            truth = np.flatnonzero(Q[Ti + 6])
            rec = {'panel': panel, 'n_true': len(truth), 'hour': ((Ti % 288) * 5) / 60.0, 'growth_T60': int(Q[Ti + 12].sum()), 'growth_T45': int(Q[Ti + 9].sum()),
                   'prior_pct_mean': float(pct[truth].mean()) if len(truth) else np.nan, 'prior_zero_share': float((prior[truth] == 0).mean()) if len(truth) else np.nan,
                   'span': int(truth.max() - truth.min() + 1) if len(truth) else 0, 'cands_fit': len(fitc)}
            for k in (1, 2, 3, 5, 8):
                top = set(order[:k].tolist()); tr = set(truth.tolist())
                rec[f'iou_top{k}'] = len(top & tr) / max(len(top | tr), 1)
            top = set(order[:max(len(truth), 1)].tolist()); tr = set(truth.tolist())
            rec['iou_topn'] = len(top & tr) / max(len(top | tr), 1)
            ONS.append(rec)
        else:
            sizes = [int(Q[Ti + h].sum()) for h in range(-1, 7) if h != 0]
            p0 = np.flatnonzero(Q[Ti - 1]); p6 = np.flatnonzero(Q[Ti + 6]); p1 = np.flatnonzero(Q[Ti + 1])
            ONG.append({'panel': panel, 'n_Tm5': sizes[0], 'n_T5': sizes[1], 'n_T15': sizes[3], 'n_T30': sizes[6],
                        'tail_shift': (p6.min() - p0.min()) if len(p0) and len(p6) else np.nan, 'head_shift': (p6.max() - p0.max()) if len(p0) and len(p6) else np.nan,
                        'iou_persist_T30': len(set(p0) & set(p6)) / max(len(set(p0) | set(p6)), 1), 'iou_persist_T5': len(set(p0) & set(p1)) / max(len(set(p0) | set(p1)), 1)})
    print(panel, 'links', nT, 'train onset events (fit days)', len(fitc), flush=True)
O = pd.DataFrame(ONS); G = pd.DataFrame(ONG)
print('\nSHIPPED ONSET WINDOWS (40): the cells that are queued at T+30')
print(O[['n_true', 'span', 'growth_T45', 'growth_T60', 'prior_pct_mean', 'prior_zero_share', 'iou_top1', 'iou_top2', 'iou_top3', 'iou_top5', 'iou_top8', 'iou_topn']].describe().round(3).to_string())
print('\nper panel mean:'); print(O.groupby('panel')[['n_true', 'span', 'growth_T60', 'prior_pct_mean', 'iou_top3', 'iou_topn']].mean().round(3).to_string())
print('\nHOUR OF DAY: shipped onset windows', np.histogram(O.hour, bins=[0, 3, 6, 9, 12, 15, 18, 21, 24])[0].tolist(), '| all train-day onset events', np.histogram(EV_HOUR, bins=[0, 3, 6, 9, 12, 15, 18, 21, 24])[0].tolist())
print('\nSHIPPED ONGOING WINDOWS (40): queue size and movement')
print(G[['n_Tm5', 'n_T5', 'n_T15', 'n_T30', 'tail_shift', 'head_shift', 'iou_persist_T5', 'iou_persist_T30']].describe().round(3).to_string())
print('share growing / same / shrinking at T+30 vs T-5:', round((G.n_T30 > G.n_Tm5).mean(), 3), round((G.n_T30 == G.n_Tm5).mean(), 3), round((G.n_T30 < G.n_Tm5).mean(), 3))
print('ongoing with tail moving upstream (shift<0):', round((G.tail_shift < 0).mean(), 3), 'head moving downstream (shift>0):', round((G.head_shift > 0).mean(), 3))
