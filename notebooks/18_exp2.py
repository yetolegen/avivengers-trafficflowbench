import bisect
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public")
OUT = Path("/kaggle/working")
REPO = Path("/tmp/trafficflowbench-public")
if not REPO.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/jacky850/trafficflowbench-public.git", str(REPO)], check=True)
print("organizers repo", subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip())
sys.path.insert(0, str(REPO / "src"))
from task1 import score_task1 as s1
from task3 import score_task3 as s3

RUN_EVAL = True
PANELS = None
HISTORY_DAYS = 56
GRID_EVERY = 7
ALL_DAYS = False
COMPARE = False
EXPECTED_ROWS = {"validation": 3_451_376, "private": 3_284_419}

KEYS = ["timestamp", "station_id", "link_id"]
VALUES = ["speed_kmh", "flow_vph"]
READ_COLS = KEYS + VALUES + ["pct_observed"]
REGIMES = ["R1", "R2", "R3"]

OLD = dict(max_gap=60, max_span=None, joint=True, cross_day=False)
VARIANTS = {
    "old": OLD,
    "separate": dict(OLD, joint=False),
    "separate+xday": dict(OLD, joint=False, cross_day=True),
    "separate+xday gap45": dict(OLD, joint=False, cross_day=True, max_gap=45),
    "separate+xday gap90": dict(OLD, joint=False, cross_day=True, max_gap=90),
    "separate+xday span90": dict(OLD, joint=False, cross_day=True, max_span=90),
}

panels = PANELS or sorted(p.name for p in (REL / "corridors").iterdir() if p.is_dir())
families = {p["corridor_id"]: p["family_id"] for p in json.loads((REPO / "config" / "corridors.json").read_text())["panels"]}
modes = json.loads((REPO / "config" / "task3_lwr_modes.json").read_text())
print("roads:", panels)

def file_date(path):
    return pd.Timestamp(Path(path).stem[-10:].replace("_", "-"))


@lru_cache(maxsize=4)
def _read(path):
    df = pd.read_parquet(path, columns=READ_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["station_id"] = df["station_id"].astype(str)
    df["link_id"] = df["link_id"].astype(str)
    return df


def read_day(path):
    return _read(path).copy()


def masked_days(panel, split):
    days = {}
    for regime in REGIMES:
        for f in sorted((REL / "corridors" / panel / split / "mainline_states_masked" / f"mask_regime={regime}").glob("*.parquet")):
            assert file_date(f) not in days, f"{f.name} in two regimes"
            days[file_date(f)] = (regime, f)
    return dict(sorted(days.items()))


def daily_profiles(panel):
    aggs = {}
    for f in sorted((REL / "corridors" / panel / "train" / "mainline_states").rglob("*.parquet")):
        d = pd.read_parquet(f, columns=READ_COLS)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d["link_id"] = d["link_id"].astype(str)
        ok = d["pct_observed"] >= 75
        d["sv"] = d["speed_kmh"].where(ok)
        d["fv"] = d["flow_vph"].where(ok)
        d["weekday"] = d["timestamp"].dt.dayofweek
        d["hour"] = d["timestamp"].dt.hour
        aggs[file_date(f)] = d.groupby(["link_id", "weekday", "hour"]).agg(
            speed_sum=("sv", "sum"), speed_n=("sv", "count"), flow_sum=("fv", "sum"), flow_n=("fv", "count"))
    return aggs


def hist_before(aggs, date):
    dates = sorted(aggs)
    used = dates[:bisect.bisect_left(dates, date)][-HISTORY_DAYS:]
    if not used:
        return None
    return pd.concat([aggs[d] for d in used]).groupby(level=[0, 1, 2]).sum()


def profile_values(rows, hist):
    n = len(rows)
    if hist is None:
        return np.full(n, np.nan), np.full(n, np.nan)
    h = hist.reset_index()
    keyed = pd.DataFrame({"link_id": rows["link_id"].to_numpy(),
                          "weekday": rows["timestamp"].dt.dayofweek.to_numpy(),
                          "hour": rows["timestamp"].dt.hour.to_numpy()})
    out = []
    for v in ("speed", "flow"):
        by_day = h.assign(m=h[f"{v}_sum"] / h[f"{v}_n"].replace(0, np.nan))[["link_id", "weekday", "hour", "m"]]
        by_hour = h.groupby(["link_id", "hour"])[[f"{v}_sum", f"{v}_n"]].sum()
        by_hour = (by_hour[f"{v}_sum"] / by_hour[f"{v}_n"].replace(0, np.nan)).rename("f").reset_index()
        overall = h[f"{v}_sum"].sum() / max(h[f"{v}_n"].sum(), 1)
        k = keyed.merge(by_day, on=["link_id", "weekday", "hour"], how="left").merge(by_hour, on=["link_id", "hour"], how="left")
        out.append(k["m"].fillna(k["f"]).fillna(overall).to_numpy())
    return out


def straight_line(pv, nv, dp, dn, cfg):
    pv, nv, dp, dn = (np.asarray(a, dtype=float) for a in (pv, nv, dp, dn))
    prev_ok = ~np.isnan(pv) & (dp <= cfg["max_gap"])
    next_ok = ~np.isnan(nv) & (dn <= cfg["max_gap"])
    both = prev_ok & next_ok
    if cfg["max_span"]:
        both &= (dp + dn) <= cfg["max_span"]
    res = np.full(len(pv), np.nan)
    res[prev_ok] = pv[prev_ok]
    res[next_ok & ~prev_ok] = nv[next_ok & ~prev_ok]
    nearer_next = prev_ok & next_ok & ~both & (dn < dp)
    res[nearer_next] = nv[nearer_next]
    res[both] = (pv[both] * dn[both] + nv[both] * dp[both]) / (dp[both] + dn[both])
    return res


def predict_day(day_files, date, targets, hist, cfg):
    dates = [date - pd.Timedelta(days=1), date, date + pd.Timedelta(days=1)] if cfg["cross_day"] else [date]
    df = pd.concat([read_day(day_files[d][1]) for d in dates if d in day_files], ignore_index=True)
    df = df.sort_values(["link_id", "station_id", "timestamp"])
    keys = [df["link_id"], df["station_id"]]
    ok = df["pct_observed"] >= 75
    have = {c: df[c].notna() for c in VALUES}
    both_present = have["speed_kmh"] & have["flow_vph"]
    ctx = df[KEYS].copy()
    for c in VALUES:
        good = ok & (both_present if cfg["joint"] else have[c])
        seen, vals = df["timestamp"].where(good), df[c].where(good)
        ctx["pt_" + c] = seen.groupby(keys).ffill()
        ctx["nt_" + c] = seen.groupby(keys).bfill()
        ctx["pv_" + c] = vals.groupby(keys).ffill()
        ctx["nv_" + c] = vals.groupby(keys).bfill()
    out = targets.merge(ctx, on=KEYS, how="left", validate="one_to_one")
    h_speed, h_flow = profile_values(out, hist)
    result = {}
    for c, h in (("speed_kmh", h_speed), ("flow_vph", h_flow)):
        dp = (out["timestamp"] - out["pt_" + c]).dt.total_seconds().to_numpy() / 60
        dn = (out["nt_" + c] - out["timestamp"]).dt.total_seconds().to_numpy() / 60
        line = straight_line(out["pv_" + c], out["nv_" + c], dp, dn, cfg)
        good = ok & have[c]
        link_mean = out["link_id"].map(df[good].groupby("link_id")[c].mean()).to_numpy()
        filled = pd.Series(line).fillna(pd.Series(h)).fillna(pd.Series(link_mean)).fillna(df.loc[good, c].mean())
        result[c] = filled.to_numpy()
        if c == "speed_kmh":
            result["span_min"] = np.nan_to_num(dp, nan=999) + np.nan_to_num(dn, nan=999)
    return np.clip(result["speed_kmh"], 1, None), np.clip(result["flow_vph"], 0, None), result["span_min"]


def load_sheet(panel, split, dates=None):
    path = REL / "task1" / panel / split / "sample_submission_state.csv"
    parts = []
    for chunk in pd.read_csv(path, dtype=str, chunksize=500_000):
        if dates is not None:
            chunk = chunk[chunk["timestamp"].str[:10].isin(dates)]
        if len(chunk):
            parts.append(chunk)
    sheet = pd.concat(parts, ignore_index=True)
    sheet["ts"] = pd.to_datetime(sheet["timestamp"], utc=True)
    sheet["date"] = sheet["timestamp"].str[:10]
    return sheet


def predict_split(panel, split, cfg, aggs, only_dates=None, hist_fixed=None):
    days = masked_days(panel, split)
    if only_dates is not None:
        days_used = {d: v for d, v in days.items() if d in only_dates}
    else:
        days_used = days
    sheet = load_sheet(panel, split, {d.strftime("%Y-%m-%d") for d in days_used} if only_dates is not None else None)
    pieces = []
    for date, (regime, _) in days_used.items():
        rows = sheet[(sheet["date"] == date.strftime("%Y-%m-%d")) & (sheet["mask_regime"] == regime)]
        if rows.empty:
            continue
        targets = rows[["station_id", "link_id"]].copy()
        targets["timestamp"] = rows["ts"].to_numpy()
        hist = hist_fixed if hist_fixed is not None else hist_before(aggs, date)
        speed, flow, span = predict_day(days, date, targets, hist, cfg)
        piece = rows[["panel", "timestamp", "ts", "station_id", "link_id", "mask_regime"]].copy()
        piece["speed_kmh"], piece["flow_vph"], piece["span_min"] = speed, flow, span
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True), sheet

def official_task1(panel, preds, names=None):
    original = s1.masked_files
    if names is not None:
        s1.masked_files = lambda rel, p, split, regime: [f for f in original(rel, p, split, regime) if f.name in names]
    try:
        sub = preds.drop(columns=["timestamp"]).rename(columns={"ts": "timestamp"})
        return pd.DataFrame(s1.score_panel(panel, REL, "train", sub))
    finally:
        s1.masked_files = original


def overall(detail):
    p = detail[detail["level"] == "panel"].assign(family=lambda d: d["panel"].map(families))
    return p.groupby("family")["S_state"].mean().mean()


def sample_days(days):
    return {d for r in REGIMES for d in [d for d, (rr, _) in days.items() if rr == r][::GRID_EVERY]}


def official_fd(panel, preds, names=None):
    sub = preds.drop(columns=["timestamp"]).rename(columns={"ts": "timestamp"})
    original = s3.files
    if names is not None:
        s3.files = lambda panel_dir, split: [f for f in original(panel_dir, split) if f.name in names]
    try:
        rows = []
        for r in REGIMES:
            d, missing = s3.derive_physics(panel, REL, "train", sub, r)
            res = s3.score_panel(panel, d, REL, modes, r)
            rows.append({"panel": panel, "regime": r, "S_FD": res.get("S_FD", 0.0), "missing_targets": missing})
        return pd.DataFrame(rows)
    finally:
        s3.files = original


def error_breakdown(panel, preds):
    truth_files = {f.name: f for f in (REL / "corridors" / panel / "train" / "mainline_states").rglob("*.parquet")}
    days = masked_days(panel, "train")
    lanes_map = s1.station_lanes(REL / "corridors" / panel)
    topo = pd.read_csv(REL / "corridors" / panel / "network" / "lwr_mainline_topology.csv", dtype={"link_id": str})
    free = topo.drop_duplicates("link_id").set_index("link_id")["free_speed_kmh"]
    preds = preds.assign(day=preds["ts"].dt.normalize().dt.tz_localize(None))
    parts = []
    for date, (regime, f) in days.items():
        p = preds[preds["day"] == date]
        if p.empty:
            continue
        t = read_day(truth_files[f.name])[KEYS + VALUES].rename(columns={"speed_kmh": "s_true", "flow_vph": "f_true"})
        m = p.drop(columns=["timestamp"]).rename(columns={"ts": "timestamp"}).merge(t, on=KEYS, how="inner")
        parts.append(m)
    m = pd.concat(parts, ignore_index=True).dropna(subset=["s_true", "f_true"])
    lanes = s1.lane_vector(m["station_id"], m["link_id"], lanes_map)
    m["se_speed"] = (m["speed_kmh"] - m["s_true"]) ** 2
    m["se_flow_lane"] = ((m["flow_vph"] - m["f_true"]) / lanes) ** 2
    m["jam"] = m["s_true"] <= 0.6 * m["link_id"].map(free)
    m["gap"] = pd.cut(m["span_min"], [0, 15, 30, 60, 120, 1e9], labels=["<=15", "15-30", "30-60", "60-120", ">120 or one side"])
    g = m.groupby(["jam", "gap"], observed=True).agg(cells=("se_speed", "size"), mse_speed=("se_speed", "mean"),
                                                     mse_flow_lane=("se_flow_lane", "mean"))
    g["rmse_speed"], g["rmse_flow_lane"] = np.sqrt(g.pop("mse_speed")), np.sqrt(g.pop("mse_flow_lane"))
    return g.reset_index().assign(panel=panel)

import time
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
EXP = ['D12_I5_N', 'D7_I210_W', 'D7_I405_S']
PARAMS = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'num_threads': 4, 'verbose': -1, 'seed': 2026}
ROUNDS = 400
FE = ['speed_kmh', 'flow_vph', 'span_min', 'f_u1', 'f_d1', 'f_u2', 'f_d2', 's_u1', 's_d1', 's_u2', 's_d2', 'nb_flow', 'nb_speed',
      'flow_minus_nb', 'speed_minus_nb', 'hour', 'dow', 'lanes', 'free', 'regime']

def station_order(panel):
    f = sorted((REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet'))[0]
    d = pd.read_parquet(f, columns=['station_id', 'milepost']).drop_duplicates('station_id')
    d['station_id'] = d['station_id'].astype(str)
    return d.sort_values('milepost')['station_id'].tolist()

def free_map(panel):
    topo = pd.read_csv(REL / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv', dtype={'link_id': str})
    return topo.drop_duplicates('link_id').set_index('link_id')['free_speed_kmh']

def table_for(panel, dates, aggs):
    days = masked_days(panel, 'train')
    preds, _ = predict_split(panel, 'train', OLD, aggs, only_dates=set(dates))
    truth_files = {f.name: f for f in (REL / 'corridors' / panel / 'train' / 'mainline_states').rglob('*.parquet')}
    order = station_order(panel); pos = {s: i for i, s in enumerate(order)}; nS = len(order)
    lanes_map = s1.station_lanes(REL / 'corridors' / panel); free = free_map(panel)
    preds = preds.assign(day=preds['ts'].dt.normalize().dt.tz_localize(None))
    out = []
    for date in dates:
        p = preds[preds['day'] == date].copy()
        if p.empty:
            continue
        regime, f = days[date]
        df = read_day(f); df = df[df['pct_observed'] >= 75]
        F = df.pivot_table(index='timestamp', columns='station_id', values='flow_vph', aggfunc='mean').reindex(columns=order)
        S = df.pivot_table(index='timestamp', columns='station_id', values='speed_kmh', aggfunc='mean').reindex(columns=order)
        Fv, Sv = F.to_numpy(), S.to_numpy()
        ti = F.index.get_indexer(p['ts']); si = p['station_id'].map(pos).fillna(-100).astype(int).to_numpy()
        def take(M, k):
            j = si + k
            okb = (j >= 0) & (j < nS) & (ti >= 0)
            res = np.full(len(p), np.nan)
            res[okb] = M[ti[okb], j[okb]]
            return res
        for k, name in ((-1, 'u1'), (1, 'd1'), (-2, 'u2'), (2, 'd2')):
            p['f_' + name] = take(Fv, k); p['s_' + name] = take(Sv, k)
        near_f = np.nanmean(np.vstack([p['f_u1'], p['f_d1']]), axis=0)
        far_f = np.nanmean(np.vstack([p['f_u2'], p['f_d2']]), axis=0)
        near_s = np.nanmean(np.vstack([p['s_u1'], p['s_d1']]), axis=0)
        far_s = np.nanmean(np.vstack([p['s_u2'], p['s_d2']]), axis=0)
        p['nb_flow'] = np.where(np.isnan(near_f), far_f, near_f); p['nb_speed'] = np.where(np.isnan(near_s), far_s, near_s)
        p['flow_minus_nb'] = p['flow_vph'] - p['nb_flow']; p['speed_minus_nb'] = p['speed_kmh'] - p['nb_speed']
        t = read_day(truth_files[f.name])[KEYS + VALUES].rename(columns={'speed_kmh': 's_true', 'flow_vph': 'f_true'})
        m = p.drop(columns=['timestamp']).rename(columns={'ts': 'timestamp'}).merge(t, on=KEYS, how='left')
        m['hour'] = m['timestamp'].dt.hour; m['dow'] = m['timestamp'].dt.dayofweek
        m['lanes'] = s1.lane_vector(m['station_id'], m['link_id'], lanes_map)
        m['free'] = m['link_id'].map(free).to_numpy(); m['regime'] = REGIMES.index(regime)
        out.append(m)
    return pd.concat(out, ignore_index=True)

def s_state(m, speed, flow):
    rows = []
    for r in range(3):
        k = (m['regime'] == r) & m['s_true'].notna() & m['f_true'].notna()
        rs = np.sqrt(np.mean((speed[k] - m['s_true'][k]) ** 2))
        rf = np.sqrt(np.mean(((flow[k] - m['f_true'][k]) / m['lanes'][k]) ** 2))
        rows.append(0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rf / 600))
    return float(np.mean(rows))

def score_sub(panel, sub, names):
    original = s1.masked_files
    s1.masked_files = lambda rel, p, split, regime: [f for f in original(rel, p, split, regime) if f.name in names]
    try:
        return pd.DataFrame(s1.score_panel(panel, REL, 'train', sub))
    finally:
        s1.masked_files = original

tables, evaldates = {}, {}
for panel in EXP:
    ts0 = time.time()
    aggs = daily_profiles(panel)
    days = masked_days(panel, 'train')
    dates = sorted(sample_days(days)); cut = int(len(dates) * 0.7)
    tables[panel] = table_for(panel, dates, aggs); evaldates[panel] = dates[cut:]
    tables[panel]['train_part'] = tables[panel]['timestamp'].dt.tz_localize(None).dt.normalize().isin(dates[:cut])
    tables[panel]['panel_code'] = EXP.index(panel)
    print(panel, 'days', len(dates), 'rows', len(tables[panel]), 'sec', round(time.time() - ts0), flush=True)
    del aggs
T_ALL = pd.concat(tables.values(), ignore_index=True)
tr = T_ALL[T_ALL['train_part'] & T_ALL['s_true'].notna() & T_ALL['f_true'].notna()]
ev = T_ALL[~T_ALL['train_part']]
FEATS = FE + ['panel_code']
t1 = time.time()
mf = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr['f_true'] - tr['flow_vph']), ROUNDS)
ms = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr['s_true'] - tr['speed_kmh']), ROUNDS)
print('trained on', len(tr), 'rows in', round(time.time() - t1), 'sec', flush=True)
ev = ev.assign(flow_fix=np.clip(ev['flow_vph'] + mf.predict(ev[FEATS]), 0, None), speed_fix=np.clip(ev['speed_kmh'] + ms.predict(ev[FEATS]), 1, None))
rows = []
for panel in EXP + ['ALL']:
    e = ev if panel == 'ALL' else ev[ev['panel'] == panel]
    rows.append({'road': panel, 'cells': len(e), 'S_state_now': s_state(e, e['speed_kmh'], e['flow_vph']),
                 'S_state_flow_only': s_state(e, e['speed_kmh'], e['flow_fix']), 'S_state_speed_only': s_state(e, e['speed_fix'], e['flow_vph']),
                 'S_state_both': s_state(e, e['speed_fix'], e['flow_fix'])})
print(pd.DataFrame(rows).round(4).to_string(), flush=True)
rm = lambda a, b, l=1: float(np.sqrt(np.mean(((a - b) / l) ** 2)))
k = ev['s_true'].notna() & ev['f_true'].notna(); e = ev[k]
print('RMSE speed now', round(rm(e['speed_kmh'], e['s_true']), 3), 'fixed', round(rm(e['speed_fix'], e['s_true']), 3),
      '| RMSE flow/lane now', round(rm(e['flow_vph'], e['f_true'], e['lanes']), 2), 'fixed', round(rm(e['flow_fix'], e['f_true'], e['lanes']), 2))
print('flow model top features:', dict(sorted(zip(FEATS, mf.feature_importance('gain')), key=lambda x: -x[1])[:8]))
print('speed model top features:', dict(sorted(zip(FEATS, ms.feature_importance('gain')), key=lambda x: -x[1])[:8]))
pn = EXP[0]
sub0 = ev[ev['panel'] == pn][['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']]
sub1 = ev[ev['panel'] == pn][['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].assign(speed_kmh=ev[ev['panel'] == pn]['speed_fix'], flow_vph=ev[ev['panel'] == pn]['flow_fix'])
names = {f.name for d, (_, f) in masked_days(pn, 'train').items() if d in set(evaldates[pn])}
for label, sub in (('official, now', sub0), ('official, fixed', sub1)):
    det = score_sub(pn, sub, names)
    print(label, pn, 'S_state', round(float(det[det['level'] == 'panel']['S_state'].iloc[0]), 4), 'missing', int(det['n_missing_predictions'].sum()), flush=True)


print('==================== IDEA 2: DENSITY FIRST ====================', flush=True)
def safe_k(f, s):
    return f / np.clip(s, 1, None)
trk = tr.assign(k_now=safe_k(tr['flow_vph'], tr['speed_kmh']), k_true=safe_k(tr['f_true'], tr['s_true']))
t2 = time.time()
mk = lgb.train(PARAMS, lgb.Dataset(trk[FEATS], trk['k_true'] - trk['k_now']), ROUNDS)
print('density model trained in', round(time.time() - t2), 'sec', flush=True)
ev2 = ev.assign(k_now=safe_k(ev['flow_vph'], ev['speed_kmh']), k_true=safe_k(ev['f_true'], ev['s_true']))
ev2['k_fix'] = np.clip(ev2['k_now'] + mk.predict(ev2[FEATS]), 0, None)
ev2['flow_B'] = ev2['k_fix'] * ev2['speed_fix']
ev2['flow_H'] = 0.5 * (ev2['flow_fix'] + ev2['flow_B'])
rows = []
for panel in EXP + ['ALL']:
    e = ev2 if panel == 'ALL' else ev2[ev2['panel'] == panel]
    rows.append({'road': panel, 'now': s_state(e, e['speed_kmh'], e['flow_vph']), 'A_current_fix': s_state(e, e['speed_fix'], e['flow_fix']),
                 'B_density_first': s_state(e, e['speed_fix'], e['flow_B']), 'H_average_of_A_and_B': s_state(e, e['speed_fix'], e['flow_H'])})
print(pd.DataFrame(rows).round(4).to_string(), flush=True)
k = ev2['s_true'].notna() & ev2['f_true'].notna() & (ev2['k_true'] > 1)
e = ev2[k].copy()
e['k_A'] = safe_k(e['flow_fix'], e['speed_fix']); e['k_B'] = e['k_fix']; e['k_H'] = safe_k(e['flow_H'], e['speed_fix'])
print('DENSITY ERROR (median relative, %):', {n: round(float(np.median(np.abs(e[c] - e['k_true']) / e['k_true'])) * 100, 2) for n, c in (('now', 'k_now'), ('A', 'k_A'), ('B', 'k_B'), ('H', 'k_H'))})
e = e.sort_values(['panel', 'station_id', 'link_id', 'timestamp'])
grp = e.groupby(['panel', 'station_id', 'link_id'], sort=False)
gap = (e['timestamp'] - grp['timestamp'].shift(1)).dt.total_seconds() / 60
pair = gap == 5
print('consecutive 5-min pairs of blanked cells:', int(pair.sum()))
den = float(np.mean(np.abs(e['k_true'][pair])))
res = {}
for n, c in (('now', 'k_now'), ('A', 'k_A'), ('B', 'k_B'), ('H', 'k_H')):
    d_est = e[c] - grp[c].shift(1); d_true = e['k_true'] - grp['k_true'].shift(1)
    res[n] = round(float(np.mean(np.abs(d_est[pair] - d_true[pair]))) / den * 100, 3)
print('STEP-TO-STEP DENSITY CHANGE ERROR (% of mean density; smaller is better):', res, flush=True)
print('==================== IDEA 3: RAMP FILES ====================', flush=True)
pn = EXP[0]
for name in ('ramp_attachment_map.csv', 'ramp_station_map.csv', 'links.csv'):
    f = REL / 'corridors' / pn / 'network' / name
    if f.exists():
        d = pd.read_csv(f)
        print('NETWORK FILE', name, d.shape, list(d.columns)); print(d.head(4).to_string())
rf = sorted((REL / 'corridors' / pn / 'train' / 'ramp_states').rglob('*.parquet'))
print('ramp files', len(rf))
if rf:
    d = pd.read_parquet(rf[0])
    print('RAMP STATES', d.shape, list(d.columns)); print(d.head(6).to_string())
    print(d.describe().round(2).to_string())
