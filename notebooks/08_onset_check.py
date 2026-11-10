import glob, json, sys, numpy as np, pandas as pd
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
B = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)[0]
sys.path.insert(0, B)
import t2lib as T
cfg = json.load(open(f'{B}/config.json')); sd = json.load(open(f'{B}/split_definitions.json'))
REL = T.find_release_root('/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
FAM = T.load_family_map(REL); PC = cfg['panel_codes']; USED = set(cfg['used_channels']); ELIG = cfg['elig_mode']
first = sd['dates_used'][0]; n_t = sd['dates_used'][2] * 288; t0 = T.text_to_minutes(first + 'T00:00:00Z')
cat = pd.read_csv(f'{B}/metrics/generated_windows_catalog.csv'); cat['Tmin'] = [T.text_to_minutes(x) for x in cat['T']]
HN = ['T+5', 'T+10', 'T+15', 'T+20', 'T+25', 'T+30']
def counts(Ti, um, ctx):
    tm = ctx.target_mask; lab, _ = T.labels_from_series(um, Ti, ctx.thr)
    a = ((lab > 0.5) & tm[None, :, None]).sum(1)
    unk = (np.isnan(lab) & tm[None, :, None]).sum(1)
    qi = np.clip(Ti[:, None] + np.arange(1, 7)[None, :], 0, n_t - 1)
    b = (um['qoff'][qi][:, :, tm] > 0).sum(2)
    hi = np.clip(Ti[:, None] + np.arange(-12, 0)[None, :], 0, n_t - 1)
    hist = (um['qoff'][hi][:, :, tm] > 0).sum((1, 2))
    return a, b, unk, hist
def firsth(c):
    return np.where((c > 0).any(1), (c > 0).argmax(1), -1)
rec = []; gen = []
for panel in cfg['panels']:
    rel = Path(REL)
    idx = T.read_window_index(rel, panel, 'train', FAM)
    tpl = T.restrict_template(T.read_template(rel, panel, 'train'), idx, panel)
    origins, _ = T.resolve_origins(idx, tpl, None, panel)
    ctx = T.PanelContext(rel, panel, FAM.get(panel), T.collect_target_links(rel, panel, FAM), PC[panel])
    fu = T.parquet_files(rel / 'corridors' / panel / 'train' / 'mainline_states')
    cu = T.resolve_colmap(T.parquet_columns(fu[0]), 'u')
    um, _ = T.load_series(fu, cu, USED, ELIG, ctx, t0, n_t, 'u')
    ids = [w for w in idx.window_id if w in origins]
    Tm = np.array([origins[w] for w in ids], np.int64); Ti = (Tm - t0) // 5
    cond = idx.set_index('window_id').loc[ids, 'condition'].to_numpy()
    a, b, unk, hist = counts(Ti, um, ctx)
    fa, fb = firsth(a), firsth(b)
    for i, w in enumerate(ids):
        rec.append(dict(panel=panel, cond=cond[i], hist_jam_cells=int(hist[i]), fa=int(fa[i]), fb=int(fb[i]), a=a[i].tolist(), unk=unk[i].tolist()))
    g = cat[(cat.panel == panel) & (cat.partition == 'fit') & (cat.Tmin - t0 >= 60)]
    gi = (g.Tmin.to_numpy(np.int64) - t0) // 5
    ga, gb, _, gh = counts(gi, um, ctx)
    fga, fgb = firsth(ga), firsth(gb)
    for i, c in enumerate(g.condition.to_numpy()):
        gen.append(dict(panel=panel, cond=c, fa=int(fga[i]), fb=int(fgb[i])))
    print(panel, 'official windows', len(ids), 'generated fit windows', len(g), flush=True)
R = pd.DataFrame(rec); G = pd.DataFrame(gen)
print('official condition values', R.cond.value_counts().to_dict())
def hist_table(df, col):
    t = df.groupby('cond')[col].value_counts().unstack(fill_value=0).reindex(columns=[-1, 0, 1, 2, 3, 4, 5], fill_value=0)
    t.columns = ['none'] + HN
    return t
for rule, col in (('link-mean speed<=cutoff (labels_from_series)', 'fa'), ('any eligible station<=cutoff (official persistence rule)', 'fb')):
    print('\n=== FIRST JAMMED HORIZON, OFFICIAL SHIPPED TRAIN WINDOWS --', rule); print(hist_table(R, col).to_string())
    print('=== same for OUR GENERATED fit windows --', rule); print(hist_table(G, col).to_string())
on = R[R.cond.str.contains('onset')].copy()
print('\nonset windows', len(on), '| jam cells in the 60 min history (should be 0):', int(on.hist_jam_cells.sum()))
print('onset, first jam exactly at T+30: rule A', int((on.fa == 5).sum()), 'rule B', int((on.fb == 5).sum()), 'of', len(on))
print('\nPER ONSET WINDOW: jammed cells at T+5..T+30 (rule A) | first-horizon A, B | unknown cells at T+30')
for _, r in on.iterrows():
    print(r.panel, r.a, 'fa', r.fa, 'fb', r.fb, 'unk30', r.unk[5])
og = R[R.cond.str.contains('ongoing')]
print('\nONGOING windows first-jam horizon (rule A):', og.fa.value_counts().sort_index().to_dict())
