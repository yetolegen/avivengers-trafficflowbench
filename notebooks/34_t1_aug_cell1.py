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


AUG = {}  # str(path) -> MultiIndex of (station_id, timestamp) readings to hide (mask augmentation)


def read_day(path):
    df = _read(path).copy()
    drop = AUG.get(str(path))
    if drop is not None:
        hit = pd.MultiIndex.from_arrays([df["station_id"], df["timestamp"]]).isin(drop)
        df.loc[hit, ["speed_kmh", "flow_vph"]] = np.nan
    return df


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
    result, gap = {}, {}
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
        k = c[0]  # gap edges: minutes to the last/next good reading, their values and times
        gap[f"g_dp_{k}"], gap[f"g_dn_{k}"] = dp.astype(np.float32), dn.astype(np.float32)
        gap[f"g_pv_{k}"] = out["pv_" + c].to_numpy(np.float32); gap[f"g_nv_{k}"] = out["nv_" + c].to_numpy(np.float32)
        gap[f"g_pt_{k}"] = out["pt_" + c].values; gap[f"g_nt_{k}"] = out["nt_" + c].values  # naive UTC datetime64
    return np.clip(result["speed_kmh"], 1, None), np.clip(result["flow_vph"], 0, None), result["span_min"], gap


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
        speed, flow, span, gap = predict_day(days, date, targets, hist, cfg)
        piece = rows[["panel", "timestamp", "ts", "station_id", "link_id", "mask_regime"]].copy()
        piece["speed_kmh"], piece["flow_vph"], piece["span_min"] = speed, flow, span
        for k, v in gap.items():
            piece[k] = v
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