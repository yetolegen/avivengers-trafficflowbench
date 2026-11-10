# Task 2 Notebook 1 v2 (train) -- ONE CELL. Paste everything into a single Kaggle code cell and run.

# ======================================================================
# Cell 2 - configuration
import os
from pathlib import Path

import numpy as np

RELEASE_ROOT = os.environ.get(
    "T2_RELEASE_ROOT",
    "/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public")
SEARCH_ROOT = os.environ.get("T2_SEARCH_ROOT", "/kaggle/input")   # used only if RELEASE_ROOT is wrong
WORK_DIR = Path(os.environ.get("T2_WORK_DIR", "/kaggle/working"))
BUNDLE_DIR = WORK_DIR / "task2_bundle"
CACHE_DIR = WORK_DIR / "task2_cache"            # scratch, not part of the bundle

MODE = os.environ.get("T2_MODE", "full")        # "full" (default) or "smoke"
SMOKE_PANEL = None                              # None -> first participating panel
SMOKE_DAYS = 30                                 # smoke: first 30 train days, 5-day blocks

SEED = 2026
NUM_THREADS = max(1, min(4, os.cpu_count() or 1))

BLOCK_DAYS = 28 if MODE == "full" else 5        # length of the test / tune / holdout blocks
EXPECTED_TRAIN_FIRST, EXPECTED_TRAIN_LAST, EXPECTED_TRAIN_DAYS = "2030-06-01", "2031-02-28", 273

# Proxy-window generation mirrors the release policy (>=360 min spacing, history coverage >= 0.7,
# ongoing windows with persistence IoU > 0.9 removed) so generated windows resemble official ones.
WINDOW_CFG = dict(min_history_coverage=0.7, max_persistence_iou=0.9,
                  min_spacing_minutes=360, max_windows_per_day_condition=2)
A_EQUIV = dict(max_age_min=5, include_t=False, agg="any_station")   # official persistence rule (reference row only)

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=NUM_THREADS, seed=SEED, deterministic=True,
                  force_col_wise=True, verbose=-1)
LGB_MAX_ROUNDS, LGB_EARLY_STOP = 2000, 100
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
SELECTION_MIN_GAIN = 0.0      # V2 must beat REF by MORE than this on tune to be selected
TRY_OFFICIAL_CLONE = True     # cross-check thresholds against the pinned official code if internet is on
print("MODE", MODE, "| threads", NUM_THREADS, "| bundle", BUNDLE_DIR)

# ======================================================================
# Cell 3 - environment, seeds, versions, directories
import gc
import importlib
import importlib.util
import json
import platform
import random
import shutil
import subprocess
import sys
import time

import lightgbm as lgb
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
try:
    display
except NameError:          # plain python (not IPython)
    display = print

PKG_VERSIONS = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__, "lightgbm": lgb.__version__}
print(PKG_VERSIONS)
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM total {vm.total / 1e9:.1f} GB, available {vm.available / 1e9:.1f} GB")
except ImportError:
    print("psutil not installed; memory not reported")
print("CPU count", os.cpu_count(), "| LightGBM threads", NUM_THREADS)

if BUNDLE_DIR.exists():
    print("Removing previous bundle:", sorted(p.name for p in BUNDLE_DIR.iterdir()))
    shutil.rmtree(BUNDLE_DIR)
(BUNDLE_DIR / "metrics").mkdir(parents=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()

# ======================================================================
# Cell 4 - write the shared module into the bundle (Notebook 2 imports this exact file)
MODULE_SRC = r'''"""t2lib - shared TrafficFlowBench Task 2 pipeline.

Written by task2_train.ipynb into the bundle directory and imported by BOTH
notebooks, so training and inference share one feature implementation.

Leakage contract (enforced by construction and by the tests in Notebook 1):
  * window inputs are a tensor of 13 slots per link: T-60 ... T-5 (history)
    plus slot 12 = the published masked-layer observation stamped exactly at T;
  * nothing stamped after T can enter a tensor;
  * labels are produced by separate functions and never passed to the feature
    builder;
  * every window is built from its own rows only; every panel has its own
    context (links, thresholds, adjacency) and no state is cached between calls.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

T2LIB_VERSION = "2.0.0"
BUNDLE_FORMAT = "task2_bundle_v2"
SEQ_K = 5                       # corridor-sequence neighbours per direction
CLIM_HALF = 2                   # climatology smoothing: +-2 slots (+-10 min)
CLIM_ALPHA = 5.0                # pseudo-count shrinking each cell toward the link's overall rate
OFFICIAL_REPO = "https://github.com/jacky850/trafficflowbench-public"
OFFICIAL_COMMIT = "f9e3fbc35c066419726987e470448f33d4c72fd9"

STEP_MIN = 5
N_HIST = 12                     # slots 0..11 = T-60 .. T-5
SLOT_T = 12                     # slot 12 = T (masked layer)
N_SLOTS = 13
HORIZONS = (5, 10, 15, 20, 25, 30)
N_H = len(HORIZONS)
SLOTS_PER_DAY = 24 * 60 // STEP_MIN
EXCLUDED_PANELS = ("D12_I405_N", "D12_I405_S")
OFFICIAL_UNKNOWN_LINK_THRESHOLD = 63.0   # fillna(63.0) in the official baseline
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
SPLITS = ("train", "validation", "private")

CHANNEL_CANDIDATES = {
    "speed": ["speed_kmh"],
    "flow": ["flow_vph"],
    "occ": ["occupancy", "occupancy_pct", "occ", "occupancy_rate", "occ_pct", "occupancy_frac"],
    "pct": ["pct_observed"],
    "elig": ["is_score_eligible"],
}
NUMERIC_CHANNELS = ("speed", "flow", "occ", "pct")
SERIES_CHANNELS = ("spd", "spd_any", "flow", "occ", "pct", "nst", "qoff")
ORIGIN_NAME_HINTS = ("forecast_origin", "origin_timestamp", "origin_time", "origin", "t0",
                     "forecast_time", "issue_time", "window_origin", "t_origin")


# --------------------------------------------------------------------------- utils
def log(*args):
    print(*args, flush=True)


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_bool(series):
    """Robust boolean parse (nulls -> False). Used for features and labels."""
    s = pd.Series(series)
    if s.dtype == bool:
        return s.to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    m = s.astype(str).str.strip().str.lower()
    return m.isin(["true", "1", "t", "yes", "y"]).to_numpy()


def to_minutes(values):
    """Timestamp text/datetimes -> (int64 epoch minutes, valid mask, whole-minute mask)."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = s.dt.tz_convert("UTC") if s.dt.tz is not None else s.dt.tz_localize("UTC")
    else:
        codes, uniq = pd.factorize(s.astype(object).where(s.notna(), None), use_na_sentinel=True)
        u = pd.Series(np.asarray(uniq, dtype=object)).astype(str)
        try:
            parsed = pd.to_datetime(u, format=TS_FMT, utc=True)
        except (ValueError, TypeError):
            parsed = pd.to_datetime(u, utc=True, errors="coerce", format="ISO8601")
        pidx = pd.DatetimeIndex(parsed)
        if len(pidx):
            ts = pd.Series(pidx[np.clip(codes, 0, None)])
            ts[codes < 0] = pd.NaT
        else:
            ts = pd.Series(pd.to_datetime(pd.Series([pd.NaT] * len(s)), utc=True))
    ts = pd.to_datetime(ts, utc=True)
    naive = ts.dt.tz_localize(None).to_numpy("datetime64[ns]")
    valid = ~np.isnat(naive)
    ns = naive.astype("int64")
    minutes = np.where(valid, ns // 60_000_000_000, 0).astype(np.int64)
    whole = valid & (ns % 60_000_000_000 == 0)
    return minutes, valid, whole


def minutes_to_text(m):
    return pd.Timestamp(int(m) * 60_000_000_000, tz="UTC").strftime(TS_FMT)


def text_to_minutes(text):
    m, v, _ = to_minutes([text])
    if not v[0]:
        raise ValueError(f"unparseable timestamp {text!r}")
    return int(m[0])


# ------------------------------------------------------------------ discovery
def find_release_root(explicit=None, search_root="/kaggle/input", max_depth=6):
    """Return the release root (directory holding task2/ and corridors/). Rejects ambiguity."""
    if explicit:
        p = Path(explicit)
        if (p / "task2").is_dir() and (p / "corridors").is_dir():
            return p
        log(f"[discovery] RELEASE_ROOT {p} lacks task2/ or corridors/; searching {search_root}")
    matches = []
    base = Path(search_root)
    if not base.is_dir():
        raise FileNotFoundError(f"search root {search_root} does not exist")
    base_depth = len(base.parts)
    for dirpath, dirnames, _ in os.walk(base):
        depth = len(Path(dirpath).parts) - base_depth
        if "task2" in dirnames and "corridors" in dirnames:
            matches.append(Path(dirpath))
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames
                           if not d.startswith((".", "mainline", "ramp", "year_month", "mask_regime"))]
    if not matches:
        raise FileNotFoundError(f"no directory containing task2/ and corridors/ under {search_root}")
    if len(matches) > 1:
        raise RuntimeError("ambiguous release roots, set RELEASE_ROOT explicitly:\n  "
                           + "\n  ".join(map(str, matches)))
    return matches[0]


def task2_path(rel, panel, split, name):
    return Path(rel) / "task2" / panel / split / name


def discover_task2_panels(rel):
    """Table of which Task 2 files exist per panel and split."""
    rows = []
    t2 = Path(rel) / "task2"
    for d in sorted(p for p in t2.iterdir() if p.is_dir()):
        for split in SPLITS:
            rows.append({
                "panel": d.name, "split": split,
                "window_index": (d / split / "window_index.csv").exists(),
                "window_history": (d / split / "window_history.parquet").exists(),
                "template": (d / split / "sample_submission_queue.csv").exists(),
            })
    return pd.DataFrame(rows)


def _data_files_in(d):
    """Data files of a Spark/Dask-style parquet DIRECTORY (skips _SUCCESS, _metadata, .crc ...)."""
    return sorted(f for f in Path(d).rglob("*") if f.is_file() and not f.name.startswith(("_", "."))
                  and not f.name.endswith((".crc", ".json", ".txt")))


def parquet_files(directory):
    """Every parquet DATA FILE below `directory`. A directory named *.parquet (written by
    Spark/Dask) is expanded into its part files; directories are never returned."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out = set()
    for f in d.rglob("*.parquet"):
        if f.is_file():
            out.add(f)
        elif f.is_dir():
            out.update(_data_files_in(f))
    return sorted(out)


def parquet_parts(path):
    """A parquet path (single file or *.parquet directory) as a list of data files."""
    p = Path(path)
    if p.is_dir():
        files = _data_files_in(p)
        if not files:
            raise FileNotFoundError(f"parquet directory {p} contains no data files")
        return files
    return [p]


def parquet_schema(path):
    return pq.ParquetFile(parquet_parts(path)[0]).schema_arrow


def parquet_columns(path):
    return list(parquet_schema(path).names)


def read_parquet_columns(path, columns=None):
    """Read each data file on its own (no pyarrow dataset discovery, so no schema merging) and
    decode dictionary-encoded columns to plain values, so files may disagree on dictionary vs
    plain encoding (e.g. mask_regime) without failing."""
    frames = []
    for f in parquet_parts(path):
        t = pq.ParquetFile(f).read(columns=columns)
        for i, field in enumerate(t.schema):
            if pa.types.is_dictionary(field.type):
                col = t.column(i)
                dec = pa.chunked_array([c.dictionary_decode() for c in col.chunks], type=field.type.value_type)
                t = t.set_column(i, field.name, dec)
        frames.append(t.to_pandas())
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def resolve_colmap(names, where=""):
    names = list(names)
    m = {}
    for canon, cands in CHANNEL_CANDIDATES.items():
        for c in cands:
            if c in names:
                m[canon] = c
                break
    if "occ" not in m:
        occ_like = [n for n in names if "occ" in n.lower()]
        if len(occ_like) == 1:
            m["occ"] = occ_like[0]
        elif len(occ_like) > 1:
            log(f"[schema] {where}: several occupancy-like columns {occ_like}; occupancy not used")
    for key in ("timestamp", "link_id"):
        if key not in names:
            raise KeyError(f"{where}: required column {key!r} missing; columns = {names}")
        m[key] = key
    if "station_id" in names:
        m["station_id"] = "station_id"
    if "window_id" in names:
        m["window_id"] = "window_id"
    if "speed" not in m:
        raise KeyError(f"{where}: no speed column among {names}")
    return m


def load_family_map(rel):
    p = Path(rel) / "config" / "corridors.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {str(x["corridor_id"]): str(x["family_id"]) for x in data.get("panels", [])}


# --------------------------------------------------------------- task2 readers
def read_window_index(rel, panel, split, family_map=None):
    p = task2_path(rel, panel, split, "window_index.csv")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "window_id" not in df.columns:
        raise KeyError(f"{p}: no window_id column; columns = {list(df.columns)}")
    if df.window_id.duplicated().any():
        raise ValueError(f"{p}: duplicated window_id values")
    if "panel" in df.columns:
        bad = df.panel != panel
        if bad.any():
            raise ValueError(f"{p}: {int(bad.sum())} rows whose panel != directory {panel}")
    else:
        df["panel"] = panel
    if "split" in df.columns:
        bad = df.split != split
        if bad.any():
            raise ValueError(f"{p}: {int(bad.sum())} rows whose split != {split}")
    if "condition" not in df.columns:
        log(f"[index] {p}: no condition column; using 'unknown'")
        df["condition"] = "unknown"
    if "family_id" not in df.columns or (df.family_id == "").any():
        fam = (family_map or {}).get(panel)
        if fam is None:
            raise KeyError(f"{p}: no family_id and none in config/corridors.json for {panel}")
        df["family_id"] = fam
    return df


def read_template(rel, panel, split):
    p = task2_path(rel, panel, split, "sample_submission_queue.csv")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    need = ["window_id", "timestamp", "link_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"{p}: missing {missing}; columns = {list(df.columns)}")
    dup = int(df.duplicated(need).sum())
    if dup:
        raise ValueError(f"{p}: {dup} duplicated (window_id, timestamp, link_id) keys")
    return df


def restrict_template(template, index, where=""):
    """Keep template rows of windows listed in this split's index; report extras."""
    known = set(index.window_id)
    in_idx = template.window_id.isin(known)
    extra = template.loc[~in_idx, "window_id"].nunique()
    if extra:
        log(f"[template] {where}: {extra} template windows are not in the split index; excluded")
    out = template.loc[in_idx].reset_index(drop=True)
    no_rows = known - set(out.window_id)
    if no_rows:
        log(f"[template] {where}: {len(no_rows)} indexed windows have no template rows")
    return out


def resolve_origins(index, template, history_max_min=None, where=""):
    """Return (dict window_id -> T in epoch minutes, source description).

    The template-derived origin (earliest target timestamp - 5 min) is always
    computed. An origin-like index column is used only if it agrees with it on
    every window. All six horizons are verified per window."""
    tmin, tvalid, _ = to_minutes(template.timestamp)
    if not tvalid.all():
        raise ValueError(f"{where}: {int((~tvalid).sum())} unparseable template timestamps")
    t = pd.DataFrame({"window_id": template.window_id.to_numpy(), "m": tmin})
    derived = (t.groupby("window_id").m.min() - STEP_MIN).to_dict()
    source = "template_min_minus_5"
    cands = [c for c in index.columns if c.lower() in ORIGIN_NAME_HINTS or "origin" in c.lower()]
    chosen, origins = None, None
    for c in cands:
        m, v, _ = to_minutes(index[c])
        if not v.all():
            log(f"[origin] {where}: column {c!r} has unparseable values; ignored")
            continue
        vals = dict(zip(index.window_id, m))
        common = [w for w in index.window_id if w in derived]
        agree = float(np.mean([vals[w] == derived[w] for w in common])) if common else 0.0
        log(f"[origin] {where}: candidate column {c!r}: agreement with template-derived T = {agree:.4f}")
        if agree == 1.0 and chosen is None:
            chosen = c
            origins = {w: int(vals[w]) for w in index.window_id}
    if chosen is not None:
        source = f"index column {chosen!r} (verified against template)"
    else:
        if cands:
            log(f"[origin] {where}: no origin-like column agreed on every window; using template-derived T")
        origins = {w: int(derived[w]) for w in index.window_id if w in derived}
    horizon = t.m.to_numpy() - t.window_id.map(origins).to_numpy(dtype=np.int64)
    bad = ~np.isin(horizon, HORIZONS)
    if bad.any():
        raise ValueError(f"{where}: {int(bad.sum())} template rows are not at T+5..T+30")
    per_w = pd.DataFrame({"w": t.window_id, "h": horizon}).groupby("w").h.nunique()
    if (per_w != N_H).any():
        raise ValueError(f"{where}: windows without all six horizons: {per_w[per_w != N_H].index[:5].tolist()}")
    if history_max_min is not None:
        diffs = [origins[w] - history_max_min[w] for w in origins if w in history_max_min]
        vc = pd.Series(diffs).value_counts().to_dict() if diffs else {}
        log(f"[origin] {where}: T - latest history timestamp (minutes) -> window counts: {vc}")
    return origins, source


# ----------------------------------------------------------------- network
def official_thresholds(panel_dir):
    """Exact copy of src/task2/queue_utils.thresholds at the pinned commit
    (station rows of fd_parameters overwrite in file order: last row per link wins)."""
    links = pd.read_csv(Path(panel_dir) / "network" / "links.csv")
    links["link_id"] = links["link_id"].astype(str)
    fallback = pd.to_numeric(links["free_speed_kmh"], errors="coerce").fillna(105.0) * 0.60
    values = dict(zip(links["link_id"].astype(str), fallback.astype(float)))
    fd = Path(panel_dir) / "network" / "fd_parameters.csv"
    if fd.exists():
        frame = pd.read_csv(fd)
        if "v_cut" in frame.columns:
            for row in frame.dropna(subset=["link_id", "v_cut"]).itertuples(index=False):
                values[str(row.link_id)] = float(row.v_cut)
            return values, "released_fd_parameters_with_network_fallback"
    return values, "0.6_times_network_free_speed_fallback"


def network_params(panel_dir, link_ids):
    """Link-level static parameters. Precedence: fd_parameters (mean over the link's
    stations) > links.csv > NaN. capacity_vph is total-link vph, lanes the link mean,
    so flow/capacity is unit-consistent and flow/lanes is per lane."""
    idx = pd.Index(list(link_ids), name="link_id")
    out = pd.DataFrame(index=idx)
    links = pd.read_csv(Path(panel_dir) / "network" / "links.csv", dtype={"link_id": str})
    links["link_id"] = links.link_id.astype(str)
    links = links.drop_duplicates("link_id").set_index("link_id")
    fd_path = Path(panel_dir) / "network" / "fd_parameters.csv"
    fd = None
    if fd_path.exists():
        fd = pd.read_csv(fd_path, dtype={"link_id": str})
        fd["link_id"] = fd.link_id.astype(str)
    src = np.full(len(idx), 2, np.float32)   # 0 = fd_parameters, 1 = links.csv, 2 = none
    for name, col in (("net_free_speed", "free_speed_kmh"), ("net_capacity", "capacity_vph"), ("net_lanes", "lanes")):
        if col in links.columns:
            v_links = pd.to_numeric(links[col], errors="coerce").reindex(idx)
        else:
            v_links = pd.Series(np.nan, index=idx)
        if fd is not None and col in fd.columns:
            v_fd = fd.assign(_v=pd.to_numeric(fd[col], errors="coerce")).groupby("link_id")["_v"].mean().reindex(idx)
        else:
            v_fd = pd.Series(np.nan, index=idx)
        out[name] = v_fd.fillna(v_links).astype(np.float32).to_numpy()
        if name == "net_free_speed":
            src = np.where(v_fd.notna(), 0, np.where(v_links.notna(), 1, 2)).astype(np.float32)
    if "length_km" in links.columns:
        out["net_length_km"] = pd.to_numeric(links["length_km"], errors="coerce").reindex(idx).astype(np.float32).to_numpy()
    else:
        out["net_length_km"] = np.float32(np.nan)
    out["net_param_source"] = src
    return out


def neighbor_index(panel_dir, link_ids):
    """Padded one-hop neighbour indices from incoming_link_ids/outgoing_link_ids
    (';'-separated). order_index is NOT used for adjacency, only as relative position."""
    L = len(link_ids)
    pos = {l: i for i, l in enumerate(link_ids)}
    p = Path(panel_dir) / "network" / "lwr_mainline_topology.csv"
    info = {"topology_file": p.exists()}
    up = [[] for _ in range(L)]
    dn = [[] for _ in range(L)]
    rel_pos = np.full(L, np.nan, np.float32)
    if p.exists():
        topo = pd.read_csv(p, dtype=str, keep_default_na=False)
        info["topology_columns"] = list(topo.columns)
        info["has_incoming"] = "incoming_link_ids" in topo.columns
        info["has_outgoing"] = "outgoing_link_ids" in topo.columns
        unknown = 0
        tu = topo.drop_duplicates("link_id")
        for r in tu.to_dict("records"):
            i = pos.get(str(r["link_id"]))
            if i is None:
                continue
            for attr, target in (("incoming_link_ids", up), ("outgoing_link_ids", dn)):
                if attr not in r:
                    continue
                for x in str(r[attr]).split(";"):
                    x = x.strip()
                    if not x or x.lower() == "nan":
                        continue
                    j = pos.get(x)
                    if j is None:
                        unknown += 1
                    elif j != i and j not in target[i]:
                        target[i].append(j)
        info["neighbour_ids_not_in_universe"] = unknown
        if "order_index" in topo.columns:
            oi = pd.to_numeric(tu.set_index("link_id").order_index, errors="coerce")
            oi = oi[oi.index.isin(list(pos))]
            if oi.notna().sum() > 1:
                r = oi.rank(method="average")
                rr = ((r - r.min()) / max(r.max() - r.min(), 1)).astype(np.float32)
                for l, v in rr.items():
                    if not np.isnan(v):
                        rel_pos[pos[l]] = v
        info["non_reciprocal_out_edges"] = int(sum(1 for i in range(L) for j in dn[i] if i not in up[j]))
    k_up = max([len(x) for x in up] + [1])
    k_dn = max([len(x) for x in dn] + [1])
    up_idx = np.full((L, k_up), -1, np.int64)
    dn_idx = np.full((L, k_dn), -1, np.int64)
    for i in range(L):
        up_idx[i, :len(up[i])] = up[i]
        dn_idx[i, :len(dn[i])] = dn[i]
    info["mean_up_degree"] = float(np.mean([len(x) for x in up])) if L else 0.0
    info["mean_dn_degree"] = float(np.mean([len(x) for x in dn])) if L else 0.0
    info["links_without_any_neighbour"] = int(sum(1 for i in range(L) if not up[i] and not dn[i]))
    return up_idx, dn_idx, rel_pos, info


def link_mileposts(panel_dir, link_ids):
    """Mean station milepost per link (and the modal direction letter) from the first unmasked
    train observation file. Static release data, read identically by both notebooks."""
    out = np.full(len(link_ids), np.nan, np.float32)
    files = parquet_files(Path(panel_dir) / "train" / "mainline_states")
    if not files:
        return out, None
    cols = parquet_columns(files[0])
    if "milepost" not in cols or "link_id" not in cols:
        return out, None
    df = read_parquet_columns(files[0], ["link_id", "milepost"] + (["direction"] if "direction" in cols else []))
    df["link_id"] = df.link_id.astype(str)
    mp = pd.to_numeric(df.milepost, errors="coerce").groupby(df.link_id).mean()
    out = mp.reindex(list(link_ids)).to_numpy(np.float32)
    direction = None
    if "direction" in df.columns and df.direction.notna().any():
        direction = str(df.direction.dropna().astype(str).mode().iloc[0]).strip().upper()[:1]
    return out, direction


def sequence_index(panel, panel_dir, link_ids, target_mask, rel_pos, dn_idx, k=SEQ_K):
    """Corridor-order neighbours: k-th upstream / downstream link along the corridor.
    Order key: topology order_index (if it covers >= 90% of target links), else station
    mileposts. Downstream direction: majority vote of the topology's outgoing edges that are
    usable, else the direction letter (N/E -> increasing milepost), else increasing order.
    Needed because the topology's incoming/outgoing ids often reference links outside the
    panel universe, leaving many links without any immediate neighbour."""
    L = len(link_ids)
    tm = np.asarray(target_mask, bool)
    info = {}
    cov_order = float((~np.isnan(rel_pos[tm])).mean()) if tm.any() else 0.0
    mp, direction = link_mileposts(panel_dir, link_ids)
    cov_mp = float((~np.isnan(mp[tm])).mean()) if tm.any() else 0.0
    if cov_order >= 0.9:
        key, src = rel_pos.astype(np.float64), "topology_order_index"
    elif cov_mp > 0:
        key, src = mp.astype(np.float64), "station_milepost"
    else:
        key, src = np.full(L, np.nan), "none"
    info.update(seq_order_source=src, seq_order_coverage_targets=float((~np.isnan(key[tm])).mean()) if tm.any() else 0.0,
                order_index_coverage_targets=cov_order, milepost_coverage_targets=cov_mp)
    ok = ~np.isnan(key)
    votes = 0
    for i in range(L):
        for j in dn_idx[i]:
            if j >= 0 and ok[i] and ok[j] and key[j] != key[i]:
                votes += 1 if key[j] > key[i] else -1
    letter = direction or str(panel).rsplit("_", 1)[-1][:1].upper()
    if votes != 0:
        sign, osrc = (1 if votes > 0 else -1), "topology_edges"
    elif src == "station_milepost" and letter in ("N", "E", "S", "W"):
        sign, osrc = (1 if letter in ("N", "E") else -1), f"direction_letter_{letter}"
    else:
        sign, osrc = 1, "assumed_increasing"
    info.update(seq_orientation=osrc, seq_orientation_votes=int(votes))
    order = np.flatnonzero(ok)
    order = order[np.argsort(sign * key[order], kind="stable")]      # upstream -> downstream
    seq_up = np.full((L, k), -1, np.int64)
    seq_dn = np.full((L, k), -1, np.int64)
    n = len(order)
    for r, i in enumerate(order):
        for kk in range(1, k + 1):
            if r + kk < n:
                seq_dn[i, kk - 1] = order[r + kk]
            if r - kk >= 0:
                seq_up[i, kk - 1] = order[r - kk]
    has = (seq_up[:, 0] >= 0) | (seq_dn[:, 0] >= 0)
    info["seq_targets_without_neighbour"] = int((~has & tm).sum())
    return seq_up, seq_dn, info


class PanelContext:
    """Static, release-derived description of one panel. Contains no fitted statistics,
    so both notebooks rebuild it identically from the release (checked by fingerprint)."""

    def __init__(self, rel, panel, family_id, target_links, panel_code):
        self.panel = panel
        self.family_id = family_id
        self.panel_code = int(panel_code)
        self.panel_dir = Path(rel) / "corridors" / panel
        links = pd.read_csv(self.panel_dir / "network" / "links.csv", dtype={"link_id": str})
        universe = set(links.link_id.astype(str)) | set(map(str, target_links))
        self.link_ids = np.array(sorted(universe), dtype=object)
        self.L = len(self.link_ids)
        self.index = pd.Index(list(self.link_ids))
        self.target_mask = np.asarray(self.index.isin(list(map(str, target_links))), bool)
        self.thr_dict, self.thr_source = official_thresholds(self.panel_dir)
        self.thr = np.array([self.thr_dict.get(l, OFFICIAL_UNKNOWN_LINK_THRESHOLD) for l in self.link_ids], np.float32)
        self.thr_fallback_63 = int(sum(l not in self.thr_dict for l in self.link_ids))
        self.net = network_params(self.panel_dir, self.link_ids)
        self.up_idx, self.dn_idx, self.rel_pos, self.topo_info = neighbor_index(self.panel_dir, list(self.link_ids))
        self.seq_up, self.seq_dn, seq_info = sequence_index(panel, self.panel_dir, list(self.link_ids),
                                                            self.target_mask, self.rel_pos, self.dn_idx)
        self.topo_info.update(seq_info)

    def link_pos(self, link_series):
        return np.asarray(self.index.get_indexer(pd.Series(link_series).astype(str).to_numpy()), np.int64)

    def fingerprint(self):
        h = hashlib.sha256()
        h.update("|".join(self.link_ids).encode())
        h.update(self.thr.tobytes())
        h.update(self.up_idx.tobytes())
        h.update(self.dn_idx.tobytes())
        h.update(np.nan_to_num(self.net.to_numpy(np.float32), nan=-999.0).tobytes())
        h.update(np.nan_to_num(self.rel_pos, nan=-999.0).tobytes())
        h.update(self.target_mask.tobytes())
        h.update(self.seq_up.tobytes())
        h.update(self.seq_dn.tobytes())
        return h.hexdigest()


# ------------------------------------------------------------ observation IO
def read_obs(path, colmap, used, keep_minutes=None):
    """Read one observation parquet into canonical columns:
    timestamp (raw), link_id, station_id (str), speed/flow/occ/pct (float64), elig_raw,
    plus _min (epoch minutes), _valid_ts, _whole."""
    cols = [colmap["timestamp"], colmap["link_id"]]
    for k in ("station_id", "window_id"):
        if k in colmap:
            cols.append(colmap[k])
    for c in list(NUMERIC_CHANNELS) + ["elig"]:
        if c in used and c in colmap:
            cols.append(colmap[c])
    df = read_parquet_columns(path, cols)
    inv = {v: k for k, v in colmap.items() if v in df.columns}
    df = df.rename(columns=inv)
    df["link_id"] = df["link_id"].astype(str)
    df["station_id"] = df["station_id"].astype(str) if "station_id" in df.columns else df["link_id"]
    if "window_id" in df.columns:
        df["window_id"] = df["window_id"].astype(str)
    for c in NUMERIC_CHANNELS:
        if c in df.columns and c in used:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        else:
            df[c] = np.nan
    if "elig" in df.columns:
        df = df.rename(columns={"elig": "elig_raw"})
    m, v, w = to_minutes(df["timestamp"])
    df["_min"], df["_valid_ts"], df["_whole"] = m, v, w
    if keep_minutes is not None:
        df = df[v & np.isin(m, np.asarray(sorted(keep_minutes), np.int64))]
    return df.reset_index(drop=True)


def eligibility(df, elig_mode):
    """Station-row eligibility used for features and proxy labels."""
    if elig_mode == "column":
        return to_bool(df["elig_raw"])
    if elig_mode == "derived_pct75":
        return (df["pct"].to_numpy() >= 75) & ~np.isnan(df["speed"].to_numpy())
    raise ValueError(f"unknown elig_mode {elig_mode}")


def aggregate_cells(key, n_keys, df, elig, thr_row):
    """Station rows -> link-time cells (rows with key < 0 are ignored).
    spd     mean speed over eligible stations with speed (primary rule, documented approximation)
    spd_any mean speed over all stations with speed (includes ineligible/imputed)
    flow    mean flow over eligible stations (a mean, never a sum of detectors)
    occ     mean occupancy over eligible stations
    pct     mean pct_observed over stations
    nst     number of eligible stations with speed
    qoff    1 if ANY eligible station speed <= link v_cut (official persistence rule)"""
    key = np.asarray(key, np.int64)
    ok = key >= 0
    k = key[ok]
    sp = df["speed"].to_numpy(np.float64)[ok]
    fl = df["flow"].to_numpy(np.float64)[ok]
    oc = df["occ"].to_numpy(np.float64)[ok]
    pc = df["pct"].to_numpy(np.float64)[ok]
    el = np.asarray(elig, bool)[ok]
    th = np.asarray(thr_row, np.float64)[ok]
    out = {}

    def mean_of(mask, vals):
        c = np.bincount(k[mask], minlength=n_keys)
        s = np.bincount(k[mask], weights=vals[mask], minlength=n_keys)
        m = np.full(n_keys, np.nan)
        np.divide(s, c, out=m, where=c > 0)
        return m.astype(np.float32), c

    e = el & ~np.isnan(sp)
    out["spd"], cnt_e = mean_of(e, sp)
    out["spd_any"], _ = mean_of(~np.isnan(sp), sp)
    out["flow"], _ = mean_of(el & ~np.isnan(fl), fl)
    out["occ"], _ = mean_of(el & ~np.isnan(oc), oc)
    out["pct"], _ = mean_of(~np.isnan(pc), pc)
    out["nst"] = cnt_e.astype(np.float32)
    slow = e.copy()
    slow[e] = sp[e] <= th[e]
    out["qoff"] = (np.bincount(k[slow], minlength=n_keys) > 0).astype(np.float32)
    return out


def scan_dates(files, colmap):
    """dict file -> sorted list of UTC dates (YYYY-MM-DD) present in the file."""
    res = {}
    for f in files:
        t = read_parquet_columns(f, [colmap["timestamp"]])[colmap["timestamp"]]
        m, v, _ = to_minutes(t.drop_duplicates())
        days = np.unique(m[v] // (24 * 60))
        res[str(f)] = [pd.Timestamp(int(d) * 86400, unit="s", tz="UTC").strftime("%Y-%m-%d") for d in days]
    return res


def load_series(files, colmap, used, elig_mode, ctx, t0_min, n_t, audit_name=""):
    """Dense (n_t, L) link-level arrays for one panel and one observation layer."""
    L = ctx.L
    arrs = {c: np.full((n_t, L), np.nan, np.float32) for c in ("spd", "spd_any", "flow", "occ", "pct")}
    arrs["nst"] = np.zeros((n_t, L), np.float32)
    arrs["qoff"] = np.zeros((n_t, L), np.float32)
    filled = np.zeros(n_t, bool)
    audit = {"layer": audit_name, "files": 0, "rows": 0, "rows_bad_ts": 0, "rows_off_grid": 0,
             "rows_out_of_range": 0, "rows_unknown_link": 0, "dup_keys": 0,
             "null_speed": 0, "null_flow": 0, "null_occ": 0, "null_pct": 0, "eligible_rows": 0,
             "max_stations_per_link": 0, "links_multi_station": 0}
    for f in files:
        df = read_obs(f, colmap, used)
        audit["files"] += 1
        audit["rows"] += len(df)
        audit["dup_keys"] += int(df.duplicated(["timestamp", "station_id", "link_id"]).sum())
        for c in NUMERIC_CHANNELS:
            audit[f"null_{c}"] += int(np.isnan(df[c].to_numpy()).sum())
        el = eligibility(df, elig_mode)
        audit["eligible_rows"] += int(el.sum())
        spl = df.groupby("link_id").station_id.nunique()
        audit["max_stations_per_link"] = max(audit["max_stations_per_link"], int(spl.max()) if len(spl) else 0)
        audit["links_multi_station"] = max(audit["links_multi_station"], int((spl > 1).sum()))
        m = df["_min"].to_numpy()
        vt = df["_valid_ts"].to_numpy() & df["_whole"].to_numpy()
        rel = m - t0_min
        on_grid = vt & (rel % STEP_MIN == 0)
        ti = rel // STEP_MIN
        inr = on_grid & (ti >= 0) & (ti < n_t)
        li = ctx.link_pos(df.link_id)
        keep = inr & (li >= 0)
        audit["rows_bad_ts"] += int((~vt).sum())
        audit["rows_off_grid"] += int((vt & ~on_grid).sum())
        audit["rows_out_of_range"] += int((on_grid & ~inr).sum())
        audit["rows_unknown_link"] += int((inr & (li < 0)).sum())
        if not keep.any():
            continue
        a, b = int(ti[keep].min()), int(ti[keep].max())
        span = b - a + 1
        key = np.where(keep, (ti - a) * L + li, -1).astype(np.int64)
        thr_row = np.where(li >= 0, ctx.thr[np.clip(li, 0, None)], np.nan)
        agg = aggregate_cells(key, span * L, df, el, thr_row)
        present = np.zeros(span, bool)
        present[np.unique(ti[keep]) - a] = True
        if (filled[a:b + 1] & present).any():
            raise ValueError(f"{audit_name}: file {f} overlaps timestamps already loaded from another file")
        filled[a:b + 1] |= present
        for c in SERIES_CHANNELS:
            blk = agg[c].reshape(span, L)
            view = arrs[c][a:b + 1]
            view[present] = blk[present]
    audit["timestamps_filled"] = int(filled.sum())
    audit["timestamps_expected"] = int(n_t)
    return arrs, audit


# ------------------------------------------------------------------ tensors
def tensor_from_series(hist, tlayer, T_idx, use_t=True):
    """(W, L, 13) tensors. Slots 0..11 from `hist` at T-60..T-5, slot 12 from the masked
    layer `tlayer` at T. No index greater than T is ever read."""
    T_idx = np.asarray(T_idx, np.int64)
    off = np.arange(-N_HIST, 0, dtype=np.int64)
    idx = T_idx[:, None] + off[None, :]
    if len(T_idx) and idx.min() < 0:
        raise IndexError("window history starts before the series")
    out = {}
    for c in SERIES_CHANNELS:
        h = hist[c][idx]                                   # (W, 12, L)
        if use_t and tlayer is not None:
            t = tlayer[c][T_idx][:, None, :]
        else:
            fill = 0.0 if c in ("nst", "qoff") else np.nan
            t = np.full((len(T_idx), 1, h.shape[2]), fill, np.float32)
        out[c] = np.ascontiguousarray(np.concatenate([h, t], axis=1).transpose(0, 2, 1))
    return out


def tensor_from_records(hist_df, t_df, ctx, window_ids, T_min, elig_mode, use_t=True):
    """Tensors for official windows from canonical station rows.
    hist_df: window_history rows (window_id required). Only rows at T-60..T-5 of their
    OWN window are used. t_df: masked-layer rows; only rows stamped exactly at T are used."""
    W, L, S = len(window_ids), ctx.L, N_SLOTS
    T_min = np.asarray(T_min, np.int64)
    diag = {}
    wpos = np.asarray(pd.Index(list(window_ids)).get_indexer(hist_df.window_id.astype(str).to_numpy()), np.int64)
    m = hist_df["_min"].to_numpy()
    okt = hist_df["_valid_ts"].to_numpy() & hist_df["_whole"].to_numpy()
    Tw = np.where(wpos >= 0, T_min[np.clip(wpos, 0, None)] if W else 0, 0)
    off = m - Tw
    on_grid = okt & (off % STEP_MIN == 0)
    slot = off // STEP_MIN + N_HIST
    li = ctx.link_pos(hist_df.link_id)
    known_w = wpos >= 0
    keep = known_w & on_grid & (slot >= 0) & (slot < N_HIST) & (li >= 0)
    diag["hist_rows"] = int(len(hist_df))
    diag["hist_rows_unknown_window"] = int((~known_w).sum())
    diag["hist_rows_at_T_dropped"] = int((known_w & on_grid & (off == 0)).sum())
    diag["hist_rows_after_T_dropped"] = int((known_w & okt & (off > 0)).sum())
    diag["hist_rows_before_T60_dropped"] = int((known_w & on_grid & (slot < 0)).sum())
    diag["hist_rows_off_grid"] = int((known_w & okt & ~on_grid).sum())
    diag["hist_rows_unknown_link"] = int((known_w & (li < 0)).sum())
    key_h = np.where(keep, (np.clip(wpos, 0, None) * S + slot) * L + li, -1).astype(np.int64)
    cols = ["speed", "flow", "occ", "pct", "link_id"] + (["elig_raw"] if elig_mode == "column" else [])
    parts = [hist_df[cols].assign(_key=key_h)]
    diag["t_rows_used"] = 0
    if use_t and t_df is not None and len(t_df):
        wins = pd.DataFrame({"_w": np.arange(W, dtype=np.int64), "_min": T_min})
        tt = t_df[t_df["_valid_ts"].to_numpy() & t_df["_whole"].to_numpy()]
        tt = tt.merge(wins, on="_min", how="inner")
        lt = ctx.link_pos(tt.link_id)
        key_t = np.where(lt >= 0, (tt["_w"].to_numpy() * S + SLOT_T) * L + lt, -1).astype(np.int64)
        parts.append(tt[cols].assign(_key=key_t))
        diag["t_rows_used"] = int((key_t >= 0).sum())
    allr = pd.concat(parts, ignore_index=True)
    el = eligibility(allr, elig_mode)
    lk = ctx.link_pos(allr.link_id)
    thr_row = np.where(lk >= 0, ctx.thr[np.clip(lk, 0, None)], np.nan)
    agg = aggregate_cells(allr["_key"].to_numpy(), W * S * L, allr, el, thr_row)
    out = {c: np.ascontiguousarray(agg[c].reshape(W, S, L).transpose(0, 2, 1)) for c in SERIES_CHANNELS}
    if not use_t:
        for c in SERIES_CHANNELS:
            out[c][..., SLOT_T] = 0.0 if c in ("nst", "qoff") else np.nan
    return out, diag


def labels_from_series(hist, T_idx, thr):
    """Proxy labels at T+5..T+30 (W, L, 6): mean eligible observed link speed <= v_cut.
    NaN = unknown (no eligible observation; NOT class 0). Also returns speed - v_cut,
    kept for label-uncertainty diagnostics only (never a feature)."""
    T_idx = np.asarray(T_idx, np.int64)
    idx = T_idx[:, None] + np.arange(1, N_H + 1)[None, :]
    n_t = hist["spd"].shape[0]
    inside = idx < n_t
    spd = hist["spd"][np.clip(idx, 0, n_t - 1)]             # (W, 6, L)
    spd = np.where(inside[:, :, None], spd, np.nan).transpose(0, 2, 1)
    lab = np.full(spd.shape, np.nan, np.float32)
    known = ~np.isnan(spd)
    lab[known] = (spd <= thr[None, :, None])[known]
    margin = (spd - thr[None, :, None]).astype(np.float32)
    return lab, margin


# ----------------------------------------------------------------- features
WL_FEATURES = [
    "spd_T", "has_T", "spd_m5", "spd_m10", "spd_m15", "spd_m30", "spd_m60",
    "last_spd", "last_age_min", "has_any_obs", "last_spd_hist", "last_age_hist_min",
    "last_ratio", "last_minus_thr", "last_over_vf", "last_any_ratio",
    "trend_5", "trend_15", "trend_30",
    "mean_15", "min_15", "std_15", "n_15", "mean_30", "min_30", "std_30",
    "mean_60", "min_60", "std_60", "n_60", "frac_valid_60", "ratio_min_30", "frac_slow_60",
    "qoff_last", "qoff_T", "qoff_any_hist", "qoff_cnt_hist",
    "last_flow", "flow_age_min", "flow_mean_15", "flow_mean_60", "flow_trend_15",
    "flow_per_lane_last", "flow_over_cap_last", "flow_missing_all",
    "last_occ", "occ_mean_15", "occ_trend_15", "occ_missing_all",
    "pct_mean_60", "nst_max",
    "up_n", "up_nv", "up_ratio_mean", "up_ratio_min", "up_q_any", "up_trend15_mean",
    "dn_n", "dn_nv", "dn_ratio_mean", "dn_ratio_min", "dn_q_any", "dn_trend15_mean",
    "net_thr", "net_free_speed", "net_capacity", "net_lanes", "net_length_km", "net_param_source", "rel_pos",
    "win_frac_q", "win_min_ratio", "win_frac_qoff_hist",
    "tod_sin", "tod_cos", "tod_hour", "weekday", "is_weekend",
]
# v1 feature set (kept unchanged as the comparison reference model)
REF_FEATURES = WL_FEATURES + ["horizon", "panel_code"]
# v2 additions, per (window, link): corridor-sequence context, queue dynamics, history condition
SEQ_FEATURES = [
    "sq_dn1_r", "sq_dn2_r", "sq_dn3_r", "sq_up1_r", "sq_up2_r", "sq_up3_r",
    "sq_dn_min5", "sq_up_min5", "sq_dn_fq5", "sq_up_fq5", "sq_dn_qdist", "sq_up_qdist",
    "sq_dn1_t15", "sq_up1_t15", "sq_dn_qhist3", "sq_up_qhist3", "sq_dn_rmin30_3",
    "sq_n_dn", "sq_n_up", "r_filled", "run_state_min", "win_frac_q_m30", "win_dfrac_q30", "hist_cond",
]
WL_FEATURES = WL_FEATURES + SEQ_FEATURES
# v2 additions, per row (depend on the horizon): link climatology of proxy queue frequency
CLIM_FEATURES = ["clim_q_h", "clim_q_T", "clim_dq", "clim_n_h", "clim_q_dn1_h"]
FEATURES = WL_FEATURES + CLIM_FEATURES + ["horizon", "panel_code"]
CATEGORICAL_FEATURES = ["panel_code"]
HIST_COND_ONSET = 0             # hist_cond codes: 0 onset_like, 1 ongoing_like, 2 other


def _last_valid(x, valid):
    S = x.shape[-1]
    idx = np.where(valid, np.arange(S), -1)
    ff = np.maximum.accumulate(idx, axis=-1)
    ls = ff[..., -1]
    val = np.take_along_axis(x, np.clip(ls, 0, None)[..., None], axis=-1)[..., 0]
    val = np.where(ls >= 0, val, np.nan).astype(np.float32)
    return ff, ls, val


def _trend(x, valid, k_slots):
    """(latest valid - latest valid at least k slots earlier) / actual elapsed minutes."""
    ff, ls, last = _last_valid(x, valid)
    pos = ls - k_slots
    ref = np.take_along_axis(ff, np.clip(pos, 0, None)[..., None], axis=-1)[..., 0]
    ok = (ls >= 0) & (pos >= 0) & (ref >= 0)
    refv = np.take_along_axis(x, np.clip(ref, 0, None)[..., None], axis=-1)[..., 0]
    elapsed = np.where(ok, (ls - ref) * STEP_MIN, 1)
    tr = np.where(ok, (last - refv) / elapsed, np.nan)
    return tr.astype(np.float32)


def _stats(x, valid, n_last):
    xs, vs = x[..., -n_last:], valid[..., -n_last:]
    cnt = vs.sum(-1)
    z = np.where(vs, xs, 0.0).astype(np.float64)
    den = np.maximum(cnt, 1)
    mean = np.where(cnt > 0, z.sum(-1) / den, np.nan)
    sq = np.where(cnt > 0, (z * z).sum(-1) / den, np.nan)
    var = np.where(cnt > 1, np.maximum(np.nan_to_num(sq - mean * mean), 0.0), np.nan)
    std = np.sqrt(var)
    mn = np.where(vs, xs, np.inf).min(-1)
    mn = np.where(np.isfinite(mn), mn, np.nan)
    return mean.astype(np.float32), mn.astype(np.float32), std.astype(np.float32), cnt.astype(np.float32)


def _neigh(v, nb_idx):
    """v (W, L) with NaN -> (count valid, mean, min) over padded neighbour indices (L, K)."""
    valid_nb = nb_idx >= 0
    g = v[:, np.clip(nb_idx, 0, None)]                     # (W, L, K)
    g = np.where(valid_nb[None], g, np.nan)
    ok = ~np.isnan(g)
    nv = ok.sum(-1).astype(np.float32)
    mean = np.where(nv > 0, np.where(ok, g, 0).sum(-1) / np.maximum(nv, 1), np.nan)
    mn = np.where(ok, g, np.inf).min(-1)
    mn = np.where(np.isfinite(mn), mn, np.nan)
    return nv, mean.astype(np.float32), mn.astype(np.float32)


def _safe_div(a, b):
    a = np.asarray(a, np.float64)
    b = np.broadcast_to(np.asarray(b, np.float64), a.shape)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=(b != 0) & ~np.isnan(b))
    return out.astype(np.float32)


def build_wl_features(tensor, ctx, T_min):
    """Per (window, link) features from a (W, L, 13) tensor -> dict name -> (W, L) float32.
    Inputs: the tensor (observations up to and including T) and static panel context only."""
    spd = tensor["spd"]
    W, L, S = spd.shape
    if S != N_SLOTS:
        raise ValueError(f"tensor has {S} slots, expected {N_SLOTS}")
    T_min = np.asarray(T_min, np.int64)
    thr2 = np.broadcast_to(ctx.thr[None, :], (W, L))
    vf = ctx.net["net_free_speed"].to_numpy(np.float32)[None, :]
    cap = ctx.net["net_capacity"].to_numpy(np.float32)[None, :]
    lanes = ctx.net["net_lanes"].to_numpy(np.float32)[None, :]
    f = {}
    v = ~np.isnan(spd)
    f["spd_T"] = spd[..., SLOT_T]
    f["has_T"] = v[..., SLOT_T].astype(np.float32)
    for name, s in (("spd_m5", 11), ("spd_m10", 10), ("spd_m15", 9), ("spd_m30", 6), ("spd_m60", 0)):
        f[name] = spd[..., s]
    _, ls, last = _last_valid(spd, v)
    f["last_spd"] = last
    f["last_age_min"] = np.where(ls >= 0, (SLOT_T - ls) * STEP_MIN, np.nan).astype(np.float32)
    f["has_any_obs"] = (ls >= 0).astype(np.float32)
    _, lsh, lasth = _last_valid(spd[..., :N_HIST], v[..., :N_HIST])
    f["last_spd_hist"] = lasth
    f["last_age_hist_min"] = np.where(lsh >= 0, (SLOT_T - lsh) * STEP_MIN, np.nan).astype(np.float32)
    f["last_ratio"] = _safe_div(last, thr2)
    f["last_minus_thr"] = (last - thr2).astype(np.float32)
    f["last_over_vf"] = _safe_div(last, vf)
    sa = tensor["spd_any"]
    f["last_any_ratio"] = _safe_div(_last_valid(sa, ~np.isnan(sa))[2], thr2)
    for k in (5, 15, 30):
        f[f"trend_{k}"] = _trend(spd, v, k // STEP_MIN)
    for n_min, n_slots in ((15, 4), (30, 7), (60, 13)):
        mean, mn, sd, cnt = _stats(spd, v, n_slots)
        f[f"mean_{n_min}"], f[f"min_{n_min}"], f[f"std_{n_min}"] = mean, mn, sd
        if n_min in (15, 60):
            f[f"n_{n_min}"] = cnt
    f["frac_valid_60"] = (f["n_60"] / N_SLOTS).astype(np.float32)
    f["ratio_min_30"] = _safe_div(f["min_30"], thr2)
    slow = np.zeros(spd.shape, bool)
    slow[v] = (spd <= ctx.thr[None, :, None])[v] if v.any() else False
    f["frac_slow_60"] = _safe_div(slow.sum(-1), np.where(f["n_60"] > 0, f["n_60"], np.nan))
    q = tensor["qoff"]
    f["qoff_last"] = q[..., N_HIST - 1].astype(np.float32)
    f["qoff_T"] = q[..., SLOT_T].astype(np.float32)
    f["qoff_any_hist"] = q[..., :N_HIST].max(-1).astype(np.float32)
    f["qoff_cnt_hist"] = q[..., :N_HIST].sum(-1).astype(np.float32)
    fl = tensor["flow"]
    vfl = ~np.isnan(fl)
    _, lsf, lastf = _last_valid(fl, vfl)
    f["last_flow"] = lastf
    f["flow_age_min"] = np.where(lsf >= 0, (SLOT_T - lsf) * STEP_MIN, np.nan).astype(np.float32)
    f["flow_mean_15"] = _stats(fl, vfl, 4)[0]
    f["flow_mean_60"] = _stats(fl, vfl, 13)[0]
    f["flow_trend_15"] = _trend(fl, vfl, 3)
    f["flow_per_lane_last"] = _safe_div(lastf, lanes)
    f["flow_over_cap_last"] = _safe_div(lastf, cap)
    f["flow_missing_all"] = (~vfl.any(-1)).astype(np.float32)
    oc = tensor["occ"]
    voc = ~np.isnan(oc)
    f["last_occ"] = _last_valid(oc, voc)[2]
    f["occ_mean_15"] = _stats(oc, voc, 4)[0]
    f["occ_trend_15"] = _trend(oc, voc, 3)
    f["occ_missing_all"] = (~voc.any(-1)).astype(np.float32)
    pc = tensor["pct"]
    f["pct_mean_60"] = _stats(pc, ~np.isnan(pc), 13)[0]
    f["nst_max"] = tensor["nst"].max(-1).astype(np.float32)
    r = f["last_ratio"]
    t15 = f["trend_15"]
    for pre, nb in (("up", ctx.up_idx), ("dn", ctx.dn_idx)):
        deg = (nb >= 0).sum(1).astype(np.float32)
        f[f"{pre}_n"] = np.broadcast_to(deg[None, :], (W, L))
        nv, mean, mn = _neigh(r, nb)
        f[f"{pre}_nv"] = nv
        f[f"{pre}_ratio_mean"] = mean
        f[f"{pre}_ratio_min"] = mn
        qa = np.full((W, L), np.nan, np.float32)
        qa[nv > 0] = (mn[nv > 0] <= 1.0)
        f[f"{pre}_q_any"] = qa
        f[f"{pre}_trend15_mean"] = _neigh(t15, nb)[1]
    for col in ("net_free_speed", "net_capacity", "net_lanes", "net_length_km", "net_param_source"):
        f[col] = np.broadcast_to(ctx.net[col].to_numpy(np.float32)[None, :], (W, L))
    f["net_thr"] = thr2
    f["rel_pos"] = np.broadcast_to(ctx.rel_pos[None, :], (W, L))
    tm = ctx.target_mask[None, :]
    rv = ~np.isnan(r) & tm
    cntv = rv.sum(1)
    rq = np.zeros(r.shape, bool)
    rq[rv] = r[rv] <= 1.0
    frac = np.where(cntv > 0, rq.sum(1) / np.maximum(cntv, 1), np.nan)
    mnr = np.where(rv, r, np.inf).min(1)
    mnr = np.where(np.isfinite(mnr), mnr, np.nan)
    fq = (f["qoff_any_hist"] * tm).sum(1) / max(int(ctx.target_mask.sum()), 1)
    for name, val in (("win_frac_q", frac), ("win_min_ratio", mnr), ("win_frac_qoff_hist", fq)):
        f[name] = np.broadcast_to(np.asarray(val, np.float32)[:, None], (W, L))
    mod = (T_min % (24 * 60)).astype(np.float64)
    dow = ((T_min // (24 * 60)) + 3) % 7        # 1970-01-01 was a Thursday; Monday = 0
    for name, val in (("tod_sin", np.sin(2 * np.pi * mod / 1440)), ("tod_cos", np.cos(2 * np.pi * mod / 1440)),
                      ("tod_hour", mod / 60.0), ("weekday", dow), ("is_weekend", dow >= 5)):
        f[name] = np.broadcast_to(np.asarray(val, np.float32)[:, None], (W, L))
    _seq_features(f, tensor, ctx, spd, v, r, t15)
    missing, extra = set(WL_FEATURES) - set(f), set(f) - set(WL_FEATURES)
    if missing or extra:
        raise RuntimeError(f"feature builder mismatch: missing={missing} extra={extra}")
    return {k: np.ascontiguousarray(f[k], dtype=np.float32) for k in WL_FEATURES}


def _take_link(v, idx_col):
    """v (W, L), idx_col (L,) link index or -1 -> (W, L) values of the indexed link (NaN for -1)."""
    g = v[:, np.clip(idx_col, 0, None)]
    return np.where(idx_col[None, :] >= 0, g, np.nan).astype(np.float32)


def _seq_features(f, tensor, ctx, spd, v, r, t15):
    """v2 features from the same (W, L, 13) tensor (inputs at or before T only)."""
    W, L = r.shape
    sq = {"dn": ctx.seq_dn, "up": ctx.seq_up}
    for pre, idx in sq.items():
        for k in (1, 2, 3):
            f[f"sq_{pre}{k}_r"] = _take_link(r, idx[:, k - 1])
        st = np.stack([_take_link(r, idx[:, k]) for k in range(idx.shape[1])], -1)     # (W, L, K)
        okk = ~np.isnan(st)
        mn = np.where(okk, st, np.inf).min(-1)
        f[f"sq_{pre}_min5"] = np.where(np.isfinite(mn), mn, np.nan).astype(np.float32)
        nv = okk.sum(-1)
        qq = (okk & (np.nan_to_num(st, nan=9.0) <= 1.0)).sum(-1)
        f[f"sq_{pre}_fq5"] = np.where(nv > 0, qq / np.maximum(nv, 1), np.nan).astype(np.float32)
        dist = np.full((W, L), np.nan, np.float32)
        for k in range(idx.shape[1] - 1, -1, -1):          # nearest currently queued link (hops)
            dist = np.where(okk[..., k] & (np.nan_to_num(st[..., k], nan=9.0) <= 1.0), np.float32(k + 1), dist)
        f[f"sq_{pre}_qdist"] = dist
        f[f"sq_{pre}1_t15"] = _take_link(t15, idx[:, 0])
        qh = np.stack([_take_link(f["qoff_any_hist"], idx[:, k]) for k in range(3)], -1)
        qh_ok = ~np.isnan(qh)
        f[f"sq_{pre}_qhist3"] = np.where(qh_ok.any(-1), np.where(qh_ok, qh, -1.0).max(-1), np.nan).astype(np.float32)
        f[f"sq_n_{pre}"] = np.broadcast_to((idx >= 0).sum(1).astype(np.float32)[None, :], (W, L))
    rm = np.stack([_take_link(f["ratio_min_30"], ctx.seq_dn[:, k]) for k in range(3)], -1)
    rm_ok = ~np.isnan(rm)
    mn = np.where(rm_ok, rm, np.inf).min(-1)
    f["sq_dn_rmin30_3"] = np.where(np.isfinite(mn), mn, np.nan).astype(np.float32)
    def _nmean(a):
        ok = ~np.isnan(a)
        n = ok.sum(-1)
        return np.where(n > 0, np.where(ok, a, 0.0).sum(-1) / np.maximum(n, 1), np.nan)

    m1 = _nmean(np.stack([f["sq_dn1_r"], f["sq_up1_r"]], -1))
    m2 = _nmean(np.stack([f["sq_dn2_r"], f["sq_up2_r"]], -1))
    f["r_filled"] = np.where(~np.isnan(r), r, np.where(~np.isnan(m1), m1, m2)).astype(np.float32)
    # duration of the current own state (minutes over valid slots): + queued, - free
    thr = ctx.thr[None, :]
    state = np.full((W, L), np.nan, np.float32)
    run = np.zeros((W, L), np.float32)
    done = np.zeros((W, L), bool)
    for s in range(N_SLOTS - 1, -1, -1):
        vs = v[..., s]
        qs = np.nan_to_num(spd[..., s], nan=np.inf) <= thr
        first = vs & np.isnan(state)
        cont = vs & ~np.isnan(state) & ~first & ~done
        same = cont & (qs == (state == 1))
        run[same] += 1
        done |= cont & ~same
        state[first] = qs[first]
        run[first] = 1
    f["run_state_min"] = np.where(np.isnan(state), np.nan, np.where(state == 1, run, -run) * STEP_MIN).astype(np.float32)
    # window-level queue extent 30 minutes ago and its change (growth / retreat)
    tm = ctx.target_mask[None, :]
    s30 = spd[..., N_HIST - 6]                               # slot 6 = T-30
    v30 = ~np.isnan(s30) & tm
    c30 = v30.sum(1)
    q30 = (v30 & (np.nan_to_num(s30, nan=np.inf) <= thr)).sum(1)
    frac30 = np.where(c30 > 0, q30 / np.maximum(c30, 1), np.nan)
    f["win_frac_q_m30"] = np.broadcast_to(np.asarray(frac30, np.float32)[:, None], (W, L))
    f["win_dfrac_q30"] = (f["win_frac_q"] - f["win_frac_q_m30"]).astype(np.float32)
    # history-only condition (0 onset_like, 1 ongoing_like, 2 other), same rule as the audit
    q = tensor["qoff"][:, ctx.target_mask, :]
    any_hist = q[..., :N_HIST].max((1, 2)) > 0 if q.size else np.zeros(W, bool)
    at_last = q[..., N_HIST - 1].max(1) > 0 if q.size else np.zeros(W, bool)
    hc = np.where(~any_hist, 0, np.where(at_last, 1, 2)).astype(np.float32)
    f["hist_cond"] = np.broadcast_to(hc[:, None], (W, L))


# ------------------------------------------------------------- climatology
def _is_weekend_day(epoch_day):
    return (((np.asarray(epoch_day, np.int64) + 3) % 7) >= 5).astype(np.int64)


class Climatology:
    """Per-link proxy-queue frequency by day type (weekday / weekend) and 5-minute slot of day,
    smoothed over +-CLIM_HALF slots and shrunk toward the link's overall rate.
    Built ONLY from the train days in `used`; validation/private days are never part of it.
    Per-day arrays (training only) allow leave-own-day-out queries, so a training row never
    sees its own day's future through the climatology."""

    def __init__(self, tp, tc, prior, day0=None, P=None, C=None, used=None):
        self.tp, self.tc, self.prior = tp, tc, prior
        self.day0, self.P, self.C, self.used = day0, P, C, used

    @staticmethod
    def day_arrays(spd, thr, t0_min):
        """Smoothed per-day counts (D, 288, L) uint8 of known and queued proxy cells."""
        if t0_min % (24 * 60):
            raise ValueError("series must start at midnight UTC")
        D = spd.shape[0] // SLOTS_PER_DAY
        x = spd[:D * SLOTS_PER_DAY].reshape(D, SLOTS_PER_DAY, -1)
        known = ~np.isnan(x)
        pos = known & (np.nan_to_num(x, nan=np.inf) <= thr[None, None, :])
        P = np.zeros(x.shape, np.uint8)
        C = np.zeros(x.shape, np.uint8)
        for k in range(-CLIM_HALF, CLIM_HALF + 1):
            P += np.roll(pos, k, axis=1)
            C += np.roll(known, k, axis=1)
        return P, C, int(t0_min // (24 * 60))

    @classmethod
    def from_days(cls, P, C, day0, used):
        used = np.asarray(used, bool)
        dt = _is_weekend_day(day0 + np.arange(P.shape[0]))
        L = P.shape[2]
        tp = np.zeros((2, SLOTS_PER_DAY, L), np.float32)
        tc = np.zeros((2, SLOTS_PER_DAY, L), np.float32)
        for t in (0, 1):
            sel = used & (dt == t)
            if sel.any():
                tp[t] = P[sel].sum(0, dtype=np.float64)
                tc[t] = C[sel].sum(0, dtype=np.float64)
        sp, sc = tp.sum((0, 1)), tc.sum((0, 1))
        glob = float(sp.sum() / sc.sum()) if sc.sum() > 0 else 0.0
        prior = np.where(sc > 0, sp / np.maximum(sc, 1), glob).astype(np.float32)
        obj = cls(tp, tc, prior, day0, P, C, used)
        obj.sp, obj.sc, obj.glob = sp.astype(np.float64), sc.astype(np.float64), glob
        obj.Pd = P.sum(1, dtype=np.float64)          # (D, L) per-day totals, for leave-own-day-out priors
        obj.Cd = C.sum(1, dtype=np.float64)
        return obj

    @classmethod
    def from_totals(cls, tp, tc, prior):
        return cls(np.asarray(tp, np.float32), np.asarray(tc, np.float32), np.asarray(prior, np.float32))

    def totals(self):
        return {"tp": self.tp, "tc": self.tc, "prior": self.prior}

    def query(self, minutes, l_idx, lodo=False):
        """(rate, support) at epoch minutes for links l_idx. lodo: drop the queried day's own
        contribution when that day is part of the climatology."""
        m = np.asarray(minutes, np.int64)
        l = np.asarray(l_idx, np.int64)
        day = m // (24 * 60)
        slot = (m % (24 * 60)) // STEP_MIN
        dt = _is_weekend_day(day)
        s = self.tp[dt, slot, l].astype(np.float64)
        c = self.tc[dt, slot, l].astype(np.float64)
        prior = self.prior[l].astype(np.float64)
        if lodo and self.P is not None:
            di = day - self.day0
            ok = (di >= 0) & (di < self.P.shape[0])
            ok[ok] = self.used[di[ok]]
            s[ok] -= self.P[di[ok], slot[ok], l[ok]]
            c[ok] -= self.C[di[ok], slot[ok], l[ok]]
            ps = self.sp[l[ok]] - self.Pd[di[ok], l[ok]]
            pc = self.sc[l[ok]] - self.Cd[di[ok], l[ok]]
            gs = self.sp.sum() - self.Pd[di[ok]].sum(1)
            gc = self.sc.sum() - self.Cd[di[ok]].sum(1)
            glob = np.where(gc > 0, gs / np.maximum(gc, 1), 0.0)
            prior[ok] = np.where(pc > 0, ps / np.maximum(pc, 1), glob)
        q = (s + CLIM_ALPHA * prior) / (c + CLIM_ALPHA)
        return q.astype(np.float32), c.astype(np.float32)


def clim_rows(clim, T_row_min, l_idx, horizon_min, seq_dn1, lodo=False):
    """CLIM_FEATURES for rows (window origin T, link, horizon). NaN when clim is None."""
    n = len(l_idx)
    if clim is None:
        return {k: np.full(n, np.nan, np.float32) for k in CLIM_FEATURES}
    T_row_min = np.asarray(T_row_min, np.int64)
    l_idx = np.asarray(l_idx, np.int64)
    tgt = T_row_min + np.asarray(horizon_min, np.int64)
    qh, nh = clim.query(tgt, l_idx, lodo)
    qT, _ = clim.query(T_row_min, l_idx, lodo)
    d1 = seq_dn1[l_idx]
    qd = np.full(n, np.nan, np.float32)
    ok = d1 >= 0
    if ok.any():
        qd[ok] = clim.query(tgt[ok], d1[ok], lodo)[0]
    return {"clim_q_h": qh, "clim_q_T": qT, "clim_dq": (qh - qT).astype(np.float32), "clim_n_h": nh,
            "clim_q_dn1_h": qd}


def rows_frame(wl, w_idx, l_idx, horizon_min, panel_code, clim=None, T_min=None, seq_dn1=None, lodo=False):
    """Per-row feature matrix in the exact FEATURES order (float32). T_min is per window."""
    w_idx = np.asarray(w_idx, np.int64)
    l_idx = np.asarray(l_idx, np.int64)
    data = {k: wl[k][w_idx, l_idx] for k in WL_FEATURES}
    T_rows = None if T_min is None else np.asarray(T_min, np.int64)[w_idx]
    if clim is not None and (T_rows is None or seq_dn1 is None):
        raise ValueError("climatology features need T_min and seq_dn1")
    data.update(clim_rows(clim, T_rows, l_idx, horizon_min, seq_dn1, lodo))
    data["horizon"] = np.asarray(horizon_min, np.float32)
    data["panel_code"] = np.full(len(w_idx), panel_code, np.float32)
    return pd.DataFrame(data, columns=FEATURES)


def target_rows_for_generated(n_windows, ctx):
    """(w_idx, l_idx, h_idx) for every target link and horizon of generated windows."""
    tl = np.flatnonzero(ctx.target_mask)
    w = np.repeat(np.arange(n_windows), len(tl) * N_H)
    l = np.tile(np.repeat(tl, N_H), n_windows)
    h = np.tile(np.arange(N_H), n_windows * len(tl))
    return w, l, h


def template_rows(template, window_ids, T_by_window, ctx):
    """Map official template rows to (w_idx, l_idx, h_idx); errors on unknown keys."""
    w_idx = np.asarray(pd.Index(list(window_ids)).get_indexer(template.window_id.to_numpy()), np.int64)
    if (w_idx < 0).any():
        raise KeyError("template rows reference windows not in the batch")
    l_idx = ctx.link_pos(template.link_id)
    if (l_idx < 0).any():
        raise KeyError(f"{int((l_idx < 0).sum())} template links missing from the panel universe")
    m, v, _ = to_minutes(template.timestamp)
    T = np.asarray([T_by_window[w] for w in template.window_id], np.int64)
    h = (m - T) // STEP_MIN - 1
    if not (v.all() and ((h >= 0) & (h < N_H)).all()):
        raise ValueError("template horizon outside 5..30")
    return w_idx, l_idx, h


# ---------------------------------------------------------------- baselines
OFFICIAL_HISTORY_COLUMNS = ["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]


def official_persistence_exact(hist_raw, template, thr_dict):
    """Vectorised copy of build_task2_persistence_submission.py (pinned commit) for one panel.
    hist_raw holds the official columns. Returns int8 predictions aligned with template
    rows and the number of template windows without history rows (the official script
    writes no rows for those; missing rows score as 0, so they are predicted 0 here)."""
    h = hist_raw[OFFICIAL_HISTORY_COLUMNS].copy()
    h["window_id"] = h.window_id.astype(str)
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    h["link_id"] = h.link_id.astype(str)
    h["speed_kmh"] = pd.to_numeric(h.speed_kmh, errors="coerce")
    last = h.groupby("window_id").timestamp.transform("max")
    at = h[h.timestamp.eq(last)].copy()
    thr = at.link_id.map(thr_dict).fillna(OFFICIAL_UNKNOWN_LINK_THRESHOLD)
    at["queue_now"] = at.speed_kmh <= thr
    at["queue_now"] &= at.is_score_eligible.astype(bool)
    q = at.groupby(["window_id", "link_id"]).queue_now.any()
    keys = pd.MultiIndex.from_arrays([template.window_id.astype(str).to_numpy(), template.link_id.astype(str).to_numpy()])
    pred = q.reindex(keys).astype("boolean").fillna(False).astype(bool).to_numpy().astype(np.int8)
    no_hist = int(len(set(template.window_id) - set(h.window_id)))
    return pred, no_hist


def persistence_b_wl(tensor, thr, max_age_min, include_t, agg):
    """Age-aware persistence per (window, link): queue state of the most recent available
    observation (optionally the published T row), if not older than max_age_min; else 0."""
    S = N_SLOTS if include_t else N_HIST
    if agg == "any_station":
        valid = tensor["nst"][..., :S] > 0
        qv = tensor["qoff"][..., :S] > 0
    elif agg == "link_mean":
        sp = tensor["spd"][..., :S]
        valid = ~np.isnan(sp)
        qv = np.zeros(sp.shape, bool)
        qv[valid] = (sp <= thr[None, :, None])[valid]
    else:
        raise ValueError(agg)
    idx = np.where(valid, np.arange(S), -1)
    ls = idx.max(-1)
    age = (SLOT_T - ls) * STEP_MIN
    q_at = np.take_along_axis(qv, np.clip(ls, 0, None)[..., None], axis=-1)[..., 0]
    return ((ls >= 0) & (age <= max_age_min) & q_at).astype(np.int8)


# ------------------------------------------------------------ proxy windows
def _roll_count(x, start_off, end_off):
    """For each t: sum(x[t+start_off .. t+end_off]) (inclusive) and in-range flag."""
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64))])
    t = np.arange(n)
    a, b = t + start_off, t + end_off
    ok = (a >= 0) & (b < n)
    s = np.where(ok, c[np.clip(b + 1, 0, n)] - c[np.clip(a, 0, n)], 0.0)
    return s, ok


def select_proxy_windows(hist, ctx, part_lo, part_hi, cfg, rng, allowed=None):
    """Competition-like proxy windows whose whole interval [T-60, T+30] lies in the
    time-index range [part_lo, part_hi). Future observations are used ONLY to select
    windows (the organisers also condition on a future queue); nothing computed here
    becomes a feature.
      proxy_onset   : no official-style queue cell (any eligible station <= v_cut) on a
                      target link in T-60..T-5, and >= 1 proxy-positive future cell
      proxy_ongoing : official-style queue cell on a target link at T-5, and >= 1
                      proxy-positive future cell
    Also: history coverage >= min_history_coverage, ongoing windows with official
    persistence proxy-IoU > max_persistence_iou removed (release policy), spacing >=
    min_spacing_minutes per condition, at most max_windows_per_day_condition per day."""
    tm = ctx.target_mask
    nT = int(tm.sum())
    q_t = hist["qoff"][:, tm].max(axis=1) > 0
    sp = hist["spd"][:, tm]
    with np.errstate(invalid="ignore"):
        fut_t = (~np.isnan(sp) & (sp <= ctx.thr[tm][None, :])).any(axis=1)
    cov_t = (hist["nst"][:, tm] > 0).sum(axis=1)
    n = len(q_t)
    h_cnt, ok1 = _roll_count(q_t, -N_HIST, -1)
    f_cnt, ok2 = _roll_count(fut_t, 1, N_H)
    cov, ok3 = _roll_count(cov_t, -N_HIST, -1)
    cov = cov / max(N_HIST * nT, 1)
    t = np.arange(n)
    inside = (t - N_HIST >= part_lo) & (t + N_H < part_hi) & ok1 & ok2 & ok3
    if allowed is not None:            # every slot of [T-60, T+30] must belong to the partition
        bad, okb = _roll_count(~np.asarray(allowed, bool), -N_HIST, N_H)
        inside &= okb & (bad == 0)
    fut_any = f_cnt > 0
    base = inside & fut_any & (cov >= cfg["min_history_coverage"])
    q_last = np.concatenate([[False], q_t[:-1]])
    cand = {"proxy_onset": np.flatnonzero(base & (h_cnt == 0)),
            "proxy_ongoing": np.flatnonzero(base & q_last)}
    stats = {"origins_inside_partition": int(inside.sum()),
             "with_future_queue": int((inside & fut_any).sum()),
             "with_future_queue_and_coverage": int(base.sum())}
    c = cand["proxy_ongoing"]
    if len(c):
        keep = np.ones(len(c), bool)
        for s0 in range(0, len(c), 2000):
            cc = c[s0:s0 + 2000]
            lab, _ = labels_from_series(hist, cc, ctx.thr)          # (n, L, 6)
            pred = (hist["qoff"][cc - 1] > 0)[:, :, None]           # official persistence at T-5
            known = ~np.isnan(lab) & tm[None, :, None]
            tru = (lab > 0.5) & known
            pp = pred & known
            inter = (tru & pp).sum((1, 2))
            union = (tru | pp).sum((1, 2))
            iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
            keep[s0:s0 + 2000] = iou <= cfg["max_persistence_iou"]
        stats["ongoing_removed_persistence_iou"] = int((~keep).sum())
        cand["proxy_ongoing"] = c[keep]
    out = []
    for cond, idx in cand.items():
        stats[f"candidates_{cond}"] = int(len(idx))
        order = rng.permutation(len(idx))
        accepted, per_day = [], {}
        min_gap = cfg["min_spacing_minutes"] / STEP_MIN
        for i in idx[order]:
            d = int(i // SLOTS_PER_DAY)
            if per_day.get(d, 0) >= cfg["max_windows_per_day_condition"]:
                continue
            if accepted and np.abs(np.asarray(accepted) - i).min() < min_gap:
                continue
            accepted.append(int(i))
            per_day[d] = per_day.get(d, 0) + 1
        stats[f"selected_{cond}"] = len(accepted)
        out.extend((i, cond) for i in accepted)
    out.sort()
    return np.array([i for i, _ in out], np.int64), [c for _, c in out], stats


# ------------------------------------------------------------------ scoring
def score_proxy(win_table, row_w, label, pred):
    """Official-style aggregation of proxy IoU (NOT the official S_queue).
    win_table: one row per window (window_key, panel, family_id, condition); row_w indexes it.
    label: float, NaN = unknown/ineligible. pred: 0/1.
    Window with no eligible cell -> 0; eligible cells but empty union -> 1.
    window -> panel-condition mean -> panel mean -> family mean -> overall mean."""
    n = len(win_table)
    row_w = np.asarray(row_w, np.int64)
    label = np.asarray(label, np.float32)
    pred = np.asarray(pred) > 0.5
    known = ~np.isnan(label)
    wk = row_w[known]
    t = label[known] > 0.5
    p = pred[known]
    elig = np.bincount(wk, minlength=n)
    inter = np.bincount(wk[t & p], minlength=n)
    union = np.bincount(wk[t | p], minlength=n)
    iou = np.where(elig == 0, 0.0, np.where(union == 0, 1.0, inter / np.maximum(union, 1)))
    det = win_table.reset_index(drop=True).copy()
    det["proxy_IoU"] = iou
    det["n_eligible"] = elig
    det["n_true"] = np.bincount(wk[t], minlength=n)
    det["n_pred"] = np.bincount(wk[p], minlength=n)
    det["n_inter"] = inter
    pc = det.groupby(["panel", "condition"], sort=True).agg(
        family_id=("family_id", "first"), proxy_IoU=("proxy_IoU", "mean"),
        n_windows=("proxy_IoU", "size"), n_eligible=("n_eligible", "sum")).reset_index()
    pm = pc.groupby("panel", sort=True).agg(
        family_id=("family_id", "first"), proxy_IoU=("proxy_IoU", "mean"),
        n_windows=("n_windows", "sum"), n_eligible=("n_eligible", "sum")).reset_index()
    fam = pm.groupby("family_id", sort=True).agg(
        proxy_IoU=("proxy_IoU", "mean"), n_panels=("panel", "size"),
        n_windows=("n_windows", "sum"), n_eligible=("n_eligible", "sum")).reset_index()
    overall = float(fam.proxy_IoU.mean()) if len(fam) else float("nan")
    return {"window": det, "panel_condition": pc, "panel": pm, "family": fam, "overall": overall}


def pooled_metrics(label, pred):
    label = np.asarray(label, np.float32)
    pred = np.asarray(pred) > 0.5
    k = ~np.isnan(label)
    t, p = label[k] > 0.5, pred[k]
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    return {"n_eligible_rows": int(k.sum()), "n_unknown_rows": int((~k).sum()),
            "true_pos_share": float(t.mean()) if k.any() else float("nan"),
            "pred_pos_share": float(p.mean()) if k.any() else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "pooled_cell_IoU": tp / (tp + fp + fn) if tp + fp + fn else float("nan")}


def age_bucket(age):
    a = np.asarray(age, np.float32)
    return np.select([np.isnan(a), a <= 0, a <= 5, a <= 15],
                     ["unavailable", "at_T", "T-5", "T-10..T-15"], "stale_>15min")


# ---------------------------------------------------------- output checks
OUTPUT_COLUMNS = ["window_id", "timestamp", "link_id", "queue_pred"]


def validate_queue_output(out, template, where=""):
    """Strict checks of a Task 2 output frame against its template rows (same order)."""
    if list(out.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{where}: columns {list(out.columns)} != {OUTPUT_COLUMNS}")
    keys = OUTPUT_COLUMNS[:3]
    if out.duplicated(keys).any():
        raise ValueError(f"{where}: duplicate keys")
    if len(out) != len(template):
        raise ValueError(f"{where}: {len(out)} rows but template has {len(template)}")
    a = out[keys].astype(str).reset_index(drop=True)
    b = template[keys].astype(str).reset_index(drop=True)
    if not (a.to_numpy() == b.to_numpy()).all():
        ka, kb = set(map(tuple, a.to_numpy())), set(map(tuple, b.to_numpy()))
        raise ValueError(f"{where}: keys/text differ from template: missing={len(kb - ka)} extra={len(ka - kb)}")
    q = pd.to_numeric(out.queue_pred, errors="coerce").to_numpy(np.float64)
    if np.isnan(q).any() or not np.isfinite(q).all():
        raise ValueError(f"{where}: NaN/inf in queue_pred")
    if not np.isin(q, [0, 1]).all():
        raise ValueError(f"{where}: non-binary queue_pred")
    return True


# ------------------------------------------------- official-window pipeline
def collect_target_links(rel, panel, family_map=None):
    """Sorted union of template link ids over every split that ships Task 2 files.
    Identifiers only (no labels); both notebooks call this so the panel universe matches."""
    links = set()
    for split in SPLITS:
        if not (task2_path(rel, panel, split, "window_index.csv").exists()
                and task2_path(rel, panel, split, "sample_submission_queue.csv").exists()):
            continue
        idx = read_window_index(rel, panel, split, family_map)
        tpl = restrict_template(read_template(rel, panel, split), idx, f"{panel}/{split}")
        links |= set(tpl.link_id.astype(str))
    return sorted(links)


def load_official_history(rel, panel, split, hist_colmap, used):
    """(canonical rows for features, raw official columns for exact persistence)."""
    p = task2_path(rel, panel, split, "window_history.parquet")
    if "window_id" not in hist_colmap:
        raise KeyError(f"{p}: window_history has no window_id column")
    hist = read_obs(p, hist_colmap, used)
    names = parquet_columns(p)
    missing = [c for c in OFFICIAL_HISTORY_COLUMNS if c not in names]
    if missing:
        raise KeyError(f"{p}: official persistence needs {missing}; columns = {names}")
    raw = read_parquet_columns(p, OFFICIAL_HISTORY_COLUMNS)
    return hist, raw


def files_covering(date_map, minutes):
    """Files (from a scan_dates map) whose dates include any of the given epoch minutes."""
    days = {pd.Timestamp(int(m) * 60_000_000_000, tz="UTC").strftime("%Y-%m-%d") for m in minutes}
    return [Path(f) for f, ds in sorted(date_map.items()) if days & set(ds)]


def read_t_rows(files, colmap, used, T_minutes):
    """Masked-layer rows stamped exactly at the given origins (nothing later is kept)."""
    keep = set(int(m) for m in T_minutes)
    parts = [read_obs(f, colmap, used, keep_minutes=keep) for f in files]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    if not np.isin(out["_min"].to_numpy(), np.asarray(sorted(keep), np.int64)).all():
        raise AssertionError("T-row reader returned rows not stamped at an origin")
    return out


def official_rows(o, ctx, clim=None, lodo=False):
    """Row features of prepared official windows with a given climatology."""
    return rows_frame(o["wl"], o["w_idx"], o["l_idx"], (o["h_idx"] + 1) * STEP_MIN, ctx.panel_code,
                      clim=clim, T_min=o["T_min"], seq_dn1=ctx.seq_dn[:, 0], lodo=lodo)


def thresholded(prob, hist_cond, thresholds):
    """Binary predictions with one threshold for onset_like windows and one for the rest."""
    hc = np.asarray(hist_cond)
    th = np.where(hc == HIST_COND_ONSET, float(thresholds["onset_like"]), float(thresholds["other"]))
    return (np.asarray(prob) >= th).astype(np.int8)


def predictions_digest(pred):
    """Fingerprint of a 0/1 prediction vector in template row order (reproducibility check)."""
    a = np.ascontiguousarray(np.asarray(pred, np.int8))
    return {"rows": int(len(a)), "positives": int(a.sum()), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def prepare_official_panel(ctx, index, template, origins, hist, raw, t_rows, elig_mode, use_t, clim=None):
    """Features and official persistence for the released windows of one panel/split.
    Used for official train windows in Notebook 1 and for validation/private in Notebook 2.
    clim: the climatology for the row features (None -> NaN climatology columns)."""
    window_ids = list(index.window_id)
    window_ids = [w for w in window_ids if w in origins]
    T_min = np.asarray([origins[w] for w in window_ids], np.int64)
    tensor, diag = tensor_from_records(hist, t_rows, ctx, window_ids, T_min, elig_mode, use_t)
    wl = build_wl_features(tensor, ctx, T_min)
    w_idx, l_idx, h_idx = template_rows(template, window_ids, origins, ctx)
    X = rows_frame(wl, w_idx, l_idx, (h_idx + 1) * STEP_MIN, ctx.panel_code,
                   clim=clim, T_min=T_min, seq_dn1=ctx.seq_dn[:, 0])
    pA, no_hist = official_persistence_exact(raw, template, ctx.thr_dict)
    diag["windows"] = len(window_ids)
    diag["windows_without_history"] = no_hist
    diag["template_rows"] = int(len(template))
    tm = ctx.target_mask
    nst = tensor["nst"]
    diag["hist_cell_coverage"] = float((nst[:, tm, :N_HIST] > 0).mean()) if len(window_ids) else float("nan")
    diag["T_cell_availability"] = float((nst[:, tm, SLOT_T] > 0).mean()) if len(window_ids) else float("nan")
    return {"window_ids": window_ids, "T_min": T_min, "tensor": tensor, "wl": wl, "X": X,
            "w_idx": w_idx, "l_idx": l_idx, "h_idx": h_idx, "pred_official": pA,
            "t_rows": t_rows, "diag": diag}


def window_condition_audit(tensor, ctx):
    """History-only condition proxy per window: 'onset_like' (no official-style queue cell in
    T-60..T-5), 'ongoing_like' (queue cell at T-5), else 'other'. Also coverage stats."""
    tm = ctx.target_mask
    q = tensor["qoff"][:, tm, :]
    any_hist = q[..., :N_HIST].max((1, 2)) > 0
    at_last = q[..., N_HIST - 1].max(1) > 0
    cond = np.where(~any_hist, "onset_like", np.where(at_last, "ongoing_like", "other"))
    cov = (tensor["nst"][:, tm, :N_HIST] > 0).mean((1, 2))
    tav = (tensor["nst"][:, tm, SLOT_T] > 0).mean(1)
    return pd.DataFrame({"history_condition": cond, "hist_coverage": cov, "T_availability": tav})
'''
(BUNDLE_DIR / "t2lib.py").write_text(MODULE_SRC, encoding="utf-8")
print("wrote", BUNDLE_DIR / "t2lib.py", len(MODULE_SRC), "chars")

# ======================================================================
# Cell 5 - import the module from the bundle
sys.dont_write_bytecode = True
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
sys.modules.pop("t2lib", None)
import t2lib as t2
assert Path(t2.__file__).resolve() == (BUNDLE_DIR / "t2lib.py").resolve(), t2.__file__
print("t2lib", t2.T2LIB_VERSION, "| official commit", t2.OFFICIAL_COMMIT)

# ======================================================================
# Cell 6 - release discovery and participating panels
REL = t2.find_release_root(RELEASE_ROOT, SEARCH_ROOT)
print("RELEASE ROOT:", REL)
avail = t2.discover_task2_panels(REL)
avail["complete"] = avail.window_index & avail.window_history & avail.template
display(avail.pivot(index="panel", columns="split", values="complete"))
complete = avail[avail.complete]
ALL_PANELS = sorted(p for p in complete.panel.unique()
                    if {"validation", "private"} <= set(complete[complete.panel == p].split))
partial = sorted(set(avail.panel) - set(ALL_PANELS))
if partial:
    print("Task 2 directories without complete validation+private files (not used):", partial)
bad = [p for p in ALL_PANELS if p in t2.EXCLUDED_PANELS]
if bad:
    raise RuntimeError(f"excluded panels have Task 2 files: {bad}")
for p in t2.EXCLUDED_PANELS:
    print(f"{p}: Task 2 directory present = {(REL / 'task2' / p).exists()} (excluded from Task 2)")
for p in ALL_PANELS:
    nd = REL / "corridors" / p / "network"
    for f in ("links.csv", "fd_parameters.csv", "lwr_mainline_topology.csv"):
        if not (nd / f).exists():
            print(f"WARNING {p}: network/{f} missing")
    if not (nd / "links.csv").exists():
        raise FileNotFoundError(nd / "links.csv")
FAMILY_MAP = t2.load_family_map(REL)
PANEL_CODES = {p: i for i, p in enumerate(ALL_PANELS)}
PANELS = ALL_PANELS if MODE == "full" else [SMOKE_PANEL or ALL_PANELS[0]]
print("participating panels:", ALL_PANELS)
print("panels processed in this run:", PANELS)
HAS_TRAIN_WINDOWS = {p: bool(((complete.panel == p) & (complete.split == "train")).any()) for p in PANELS}
print("official train windows available:", HAS_TRAIN_WINDOWS)

# Optional: cross-check thresholds() against the pinned official source.
OFFICIAL_SRC = None
if TRY_OFFICIAL_CLONE:
    dst = Path("/tmp/tfb_official") if Path("/tmp").is_dir() else CACHE_DIR / "tfb_official"   # kept out of the Output
    try:
        if not (dst / ".git").exists():
            subprocess.run(["git", "init", "-q", str(dst)], check=True, timeout=60)
            subprocess.run(["git", "-C", str(dst), "remote", "add", "origin", t2.OFFICIAL_REPO], check=True, timeout=60)
        subprocess.run(["git", "-C", str(dst), "fetch", "-q", "--depth", "1", "origin", t2.OFFICIAL_COMMIT],
                       check=True, timeout=180)
        subprocess.run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"], check=True, timeout=60)
        head = subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != t2.OFFICIAL_COMMIT:
            raise RuntimeError(f"checked out {head}")
        OFFICIAL_SRC = dst / "src"
    except Exception as exc:     # internet off or git missing: reported, not hidden
        print("Official repo not fetched -> threshold cross-check skipped:", repr(exc))
if OFFICIAL_SRC is not None:
    spec = importlib.util.spec_from_file_location("official_queue_utils", OFFICIAL_SRC / "task2" / "queue_utils.py")
    oqu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oqu)
    for p in PANELS:
        pdir = REL / "corridors" / p
        links = pd.read_csv(pdir / "network" / "links.csv")
        links["link_id"] = links.link_id.astype(str)
        ref, _ = oqu.thresholds(pdir, links)
        mine, _ = t2.official_thresholds(pdir)
        if ref != mine:
            raise AssertionError(f"{p}: t2lib thresholds differ from official queue_utils.thresholds")
    print("thresholds identical to official queue_utils.thresholds at", t2.OFFICIAL_COMMIT)

# ======================================================================
# Cell 7 - schemas of Task 2 files and observation layers
def head_parquet(path, n=3):
    pf = pq.ParquetFile(t2.parquet_parts(path)[0])
    return pf.read_row_group(0).slice(0, n).to_pandas() if pf.num_row_groups else pd.DataFrame()


for split in t2.SPLITS:
    ps = [p for p in PANELS if t2.task2_path(REL, p, split, "window_index.csv").exists()]
    if not ps:
        continue
    p = ps[0]
    print(f"\n===== {split} (example panel {p}) =====")
    wi = pd.read_csv(t2.task2_path(REL, p, split, "window_index.csv"), dtype=str)
    print("window_index columns:", list(wi.columns))
    display(wi.head(3))
    tp = pd.read_csv(t2.task2_path(REL, p, split, "sample_submission_queue.csv"), dtype=str, nrows=3)
    print("template columns:", list(tp.columns))
    display(tp)
    hp = t2.task2_path(REL, p, split, "window_history.parquet")
    print("window_history schema:")
    print(t2.parquet_schema(hp))
    display(head_parquet(hp))
    for q in ps[1:]:
        if list(pd.read_csv(t2.task2_path(REL, q, split, "window_index.csv"), nrows=0).columns) != list(wi.columns):
            print(f"WARNING: window_index columns differ for {q}/{split}")
        hq = t2.task2_path(REL, q, split, "window_history.parquet")
        if hq.exists() and t2.parquet_columns(hq) != t2.parquet_columns(hp):
            print(f"WARNING: window_history columns differ for {q}/{split}")

LAYER_DIRS = {"train_unmasked": ("train", "mainline_states"), "train_masked": ("train", "mainline_states_masked"),
              "validation_masked": ("validation", "mainline_states_masked"),
              "private_masked": ("private", "mainline_states_masked")}
LAYER_FILES = {layer: {p: t2.parquet_files(REL / "corridors" / p / s / d) for p in PANELS}
               for layer, (s, d) in LAYER_DIRS.items()}
LAYER_COLUMNS = {}
for layer, by_panel in LAYER_FILES.items():
    sets = []
    for p, files in by_panel.items():
        print(f"{layer:18s} {p:12s} files={len(files)}")
        sets += [tuple(t2.parquet_columns(f)) for f in files]
    if not sets:
        LAYER_COLUMNS[layer] = None
        continue
    if len(set(sets)) > 1:
        raise ValueError(f"{layer}: parquet files disagree on columns: {set(sets)}")
    LAYER_COLUMNS[layer] = list(sets[0])
    f0 = next(f for fs in by_panel.values() for f in fs)
    print(f"\n{layer} schema ({f0}):")
    print(t2.parquet_schema(f0))
    display(head_parquet(f0))
if not LAYER_COLUMNS["train_unmasked"]:
    raise FileNotFoundError("no unmasked train observations (corridors/<PANEL>/train/mainline_states/)")

# ======================================================================
# Cell 8 - channel decision (identical for training and inference)
LAYER_COLMAP = {k: (t2.resolve_colmap(v, k) if v else None) for k, v in LAYER_COLUMNS.items()}
HIST_COLMAP = {}
for split in t2.SPLITS:
    ps = [p for p in PANELS if t2.task2_path(REL, p, split, "window_history.parquet").exists()]
    if ps:
        HIST_COLMAP[split] = t2.resolve_colmap(
            t2.parquet_columns(t2.task2_path(REL, ps[0], split, "window_history.parquet")), f"history/{split}")
for split in ("validation", "private"):
    if split not in HIST_COLMAP:
        raise FileNotFoundError(f"no {split} window_history for the processed panels")
USE_T = all(LAYER_COLMAP.get(k) for k in ("train_masked", "validation_masked", "private_masked"))
sources = [LAYER_COLMAP["train_unmasked"]] + list(HIST_COLMAP.values())
if USE_T:
    sources += [LAYER_COLMAP[k] for k in ("train_masked", "validation_masked", "private_masked")]
USED = ["speed"] + [c for c in ("flow", "occ", "pct") if all(c in s for s in sources)]
if all("elig" in s for s in sources):
    ELIG_MODE = "column"
    USED.append("elig")
elif all("pct" in s for s in sources):
    ELIG_MODE = "derived_pct75"
else:
    raise KeyError("is_score_eligible is not in every source and pct_observed cannot replace it")
print("layer column maps:", json.dumps(LAYER_COLMAP, indent=1))
print("history column maps:", json.dumps(HIST_COLMAP, indent=1))
print("USED channels:", USED, "| eligibility:", ELIG_MODE, "| use published T row:", USE_T)
for c in ("flow", "occ", "pct"):
    if c not in USED:
        print(f"NOTE: channel {c!r} is not available in every source; its features stay NaN everywhere")

# ======================================================================
# Cell 9 - calendar dates of every layer
DATE_MAP = {}
for layer, by_panel in LAYER_FILES.items():
    if not LAYER_COLMAP.get(layer):
        continue
    for p, files in by_panel.items():
        DATE_MAP[(layer, p)] = t2.scan_dates(files, LAYER_COLMAP[layer])
rows = []
for (layer, p), dm in DATE_MAP.items():
    ds = sorted({d for v in dm.values() for d in v})
    rows.append({"layer": layer, "panel": p, "files": len(dm), "days": len(ds),
                 "first": ds[0] if ds else None, "last": ds[-1] if ds else None,
                 "files_spanning_>1_day": sum(len(v) > 1 for v in dm.values())})
DATE_TABLE = pd.DataFrame(rows)
display(DATE_TABLE)
train_days = {p: sorted({d for v in DATE_MAP[("train_unmasked", p)].values() for d in v}) for p in PANELS}
common = sorted(set.intersection(*[set(v) for v in train_days.values()]))
for p, v in train_days.items():
    if set(v) != set(common):
        print(f"WARNING {p}: {len(set(v) - set(common))} train days not shared by all panels (ignored)")
if MODE == "full":
    if (common[0], common[-1], len(common)) != (EXPECTED_TRAIN_FIRST, EXPECTED_TRAIN_LAST, EXPECTED_TRAIN_DAYS):
        print(f"WARNING: train dates {common[0]}..{common[-1]} ({len(common)} days) differ from the expected "
              f"{EXPECTED_TRAIN_FIRST}..{EXPECTED_TRAIN_LAST} ({EXPECTED_TRAIN_DAYS}); using actual coverage")
    full_range = pd.date_range(common[0], common[-1], freq="D").strftime("%Y-%m-%d").tolist()
    if full_range != common:
        raise ValueError(f"{len(full_range) - len(common)} calendar gaps in shared train coverage; "
                         "the dense time index assumes contiguous days")
else:
    common = common[:SMOKE_DAYS]
DATES = common
print(f"train dates used: {DATES[0]} .. {DATES[-1]} ({len(DATES)} days)")
for split in ("validation", "private"):
    key = (f"{split}_masked", PANELS[0])
    if key in DATE_MAP:
        ds = sorted({d for v in DATE_MAP[key].values() for d in v})
        print(f"{split} masked layer dates: {ds[0]} .. {ds[-1]} ({len(ds)} days)")

# ======================================================================
# Cell 10 - official Task 2 tables (index, template, history, origins) and panel contexts
T2 = {}
summary = []
for p in PANELS:
    for split in t2.SPLITS:
        if not all(t2.task2_path(REL, p, split, n).exists() for n in
                   ("window_index.csv", "window_history.parquet", "sample_submission_queue.csv")):
            continue
        idx = t2.read_window_index(REL, p, split, FAMILY_MAP)
        tpl_full = t2.read_template(REL, p, split)
        tpl = t2.restrict_template(tpl_full, idx, f"{p}/{split}")
        hist, raw = t2.load_official_history(REL, p, split, HIST_COLMAP[split], USED)
        hmax = hist[hist._valid_ts].groupby("window_id")._min.max().to_dict()
        origins, osrc = t2.resolve_origins(idx, tpl, hmax, f"{p}/{split}")
        T2[(p, split)] = dict(index=idx, template=tpl, hist=hist, raw=raw, origins=origins, origin_source=osrc)
        off = (hist._min - hist.window_id.map(origins)).dropna()
        summary.append({"panel": p, "split": split, "windows": len(idx),
                        **{f"n_{c}": int((idx.condition == c).sum()) for c in sorted(idx.condition.unique())},
                        "template_rows": len(tpl), "template_rows_in_file": len(tpl_full),
                        "links_per_window": tpl.groupby("window_id").link_id.nunique().median(),
                        "history_rows": len(hist), "hist_offset_min": off.min(), "hist_offset_max": off.max(),
                        "origin_source": osrc})
T2_SUMMARY = pd.DataFrame(summary)
display(T2_SUMMARY)
for split in ("validation", "private"):
    print(f"{split}: template rows over processed panels = "
          f"{int(T2_SUMMARY[T2_SUMMARY.split == split].template_rows.sum()):,}")

TARGET_LINKS = {p: t2.collect_target_links(REL, p, FAMILY_MAP) for p in PANELS}
CTX = {}
ctx_rows = []
for p in PANELS:
    fam = T2[(p, "validation")]["index"].family_id.iloc[0]
    ctx = t2.PanelContext(REL, p, fam, TARGET_LINKS[p], PANEL_CODES[p])
    CTX[p] = ctx
    per_window = {s: T2[(p, s)]["template"].groupby("window_id").link_id.apply(frozenset)
                  for s in t2.SPLITS if (p, s) in T2}
    same = all(len(set(v)) == 1 for v in per_window.values())
    fdp = ctx.panel_dir / "network" / "fd_parameters.csv"
    fd = pd.read_csv(fdp, dtype=str) if fdp.exists() else pd.DataFrame()
    st_per_link = fd.groupby("link_id").size() if "link_id" in fd.columns else pd.Series(dtype=int)
    ctx_rows.append({"panel": p, "family": fam, "links_universe": ctx.L, "target_links": int(ctx.target_mask.sum()),
                     "same_links_every_window": same, "thr_source": ctx.thr_source,
                     "links_thr_63_fallback": ctx.thr_fallback_63, "v_cut_min": float(ctx.thr.min()),
                     "v_cut_max": float(ctx.thr.max()), "fd_links_multi_station": int((st_per_link > 1).sum()),
                     "net_src_fd/links/none": tuple(np.bincount(ctx.net.net_param_source.astype(int), minlength=3)),
                     **{k: v for k, v in ctx.topo_info.items() if k != "topology_columns"}})
display(pd.DataFrame(ctx_rows))
FINGERPRINTS = {p: CTX[p].fingerprint() for p in PANELS}

# ======================================================================
# Cell 11 - date blocks (decided BEFORE any window is generated)
n_days = len(DATES)
B = BLOCK_DAYS
if n_days < 5 * B:
    B = max(3, n_days // 9)
    print(f"WARNING: only {n_days} train days; block length reduced to {B}")
T0_MIN = t2.text_to_minutes(DATES[0] + "T00:00:00Z")
N_T = n_days * t2.SLOTS_PER_DAY
DAY_PART = np.array(["fit"] * n_days, dtype=object)
DAY_PART[n_days - B:] = "holdout"
DAY_PART[n_days - 2 * B:n_days - B] = "tune"
DAY_PART[n_days - 3 * B:n_days - 2 * B] = "test"
# Days touched by the official TRAIN windows ([T-60, T+30]) leave the fit block, so those
# windows become an evaluation set the evaluation models never trained on.
OFF_DAYS = set()
for (p, split), d in T2.items():
    if split != "train":
        continue
    for T in d["origins"].values():
        for m in (T - 60, T + 30):
            di = int((m - T0_MIN) // (24 * 60))
            if 0 <= di < n_days:
                OFF_DAYS.add(di)
moved = sorted(di for di in OFF_DAYS if DAY_PART[di] == "fit")
DAY_PART[moved] = "offtest"
elsewhere = sorted(di for di in OFF_DAYS if DAY_PART[di] != "offtest")
if elsewhere:
    print("NOTE: official train windows on days of other blocks (kept there):", [DATES[i] for i in elsewhere])
PART_NAMES = np.array(["fit", "offtest", "test", "tune", "holdout"], dtype=object)
CODE = {n: i for i, n in enumerate(PART_NAMES)}
SLOT_PART = np.repeat(np.array([CODE[x] for x in DAY_PART], np.int8), t2.SLOTS_PER_DAY)
_CHG = np.concatenate([[0], np.cumsum(SLOT_PART[1:] != SLOT_PART[:-1])])


def partition_of(T_idx):
    """Partition name for origins whose whole window [T-60, T+30] lies in one block, else 'boundary'."""
    T_idx = np.asarray(T_idx, np.int64)
    lo, hi = T_idx - t2.N_HIST, T_idx + t2.N_H
    ok = (lo >= 0) & (hi < N_T)
    out = np.full(len(T_idx), "boundary", dtype=object)
    same = ok.copy()
    same[ok] = _CHG[hi[ok]] == _CHG[lo[ok]]
    out[same] = PART_NAMES[SLOT_PART[T_idx[same]]]
    return out


PARTITIONS = []
for name in PART_NAMES:
    di = np.flatnonzero(DAY_PART == name)
    PARTITIONS.append({"name": name, "days": int(len(di)),
                       "first_day": DATES[di[0]] if len(di) else None, "last_day": DATES[di[-1]] if len(di) else None,
                       "dates": [DATES[i] for i in di]})
SPLIT_DEF = {"dates_used": [DATES[0], DATES[-1], n_days], "block_days": B,
             "partitions": PARTITIONS,
             "rule": "every slot of [T-60, T+30] must lie in one block; boundary-crossing windows are removed",
             "roles": {"fit": "training of the evaluation models", "tune": "early stopping, thresholds, selection",
                       "test": "untouched evaluation (first look in v2)", "holdout": "evaluation (already inspected in v1)",
                       "offtest": "days of the official train windows, excluded from training of the evaluation models"}}
display(pd.DataFrame([{k: v for k, v in x.items() if k != "dates"} for x in PARTITIONS]))

# ======================================================================
# Cell 12 - official windows: features, exact persistence, condition and availability audit
OFF = {}
audit_rows = []
for (p, split), d in T2.items():
    ctx = CTX[p]
    T_list = [d["origins"][w] for w in d["index"].window_id if w in d["origins"]]
    t_rows = None
    if USE_T:
        files = t2.files_covering(DATE_MAP[(f"{split}_masked", p)], T_list)
        t_rows = t2.read_t_rows(files, LAYER_COLMAP[f"{split}_masked"], USED, T_list)
    o = t2.prepare_official_panel(ctx, d["index"], d["template"], d["origins"], d["hist"], d["raw"],
                                  t_rows, ELIG_MODE, USE_T)
    OFF[(p, split)] = o
    aud = t2.window_condition_audit(o["tensor"], ctx)
    aud["official_condition"] = d["index"].set_index("window_id").condition.reindex(o["window_ids"]).to_numpy()
    aud["panel"], aud["split"] = p, split
    audit_rows.append(aud)
    print(f"{p}/{split}: {o['diag']} | official persistence positives = {int(o['pred_official'].sum())}")
COND_AUDIT = pd.concat(audit_rows, ignore_index=True)
print("\nOfficial condition vs history-only proxy condition (all splits; history inputs only):")
display(pd.crosstab([COND_AUDIT.split, COND_AUDIT.official_condition], COND_AUDIT.history_condition))
print("\nReleased-window input availability (target links):")
display(COND_AUDIT.groupby(["split", "official_condition"])[["hist_coverage", "T_availability"]]
        .describe(percentiles=[0.05, 0.5]).round(3))
for split in ("validation", "private"):
    s = sum(int(o["pred_official"].sum()) for (p, sp), o in OFF.items() if sp == split)
    print(f"exact official persistence positives, {split}: {s:,}")
tr_cov = COND_AUDIT[COND_AUDIT.split == "train"].hist_coverage
if len(tr_cov) and tr_cov.min() < WINDOW_CFG["min_history_coverage"]:
    WINDOW_CFG["min_history_coverage"] = float(np.floor(tr_cov.min() * 100) / 100)
    print("min_history_coverage lowered to the official TRAIN windows' minimum:", WINDOW_CFG["min_history_coverage"])
print("WINDOW_CFG:", WINDOW_CFG)

# ======================================================================
# Cell 13 - per-panel processing: series, climatology, proxy windows, features, labels
def files_in_dates(date_map, dates):
    ds = set(dates)
    return [Path(f) for f, v in sorted(date_map.items()) if ds & set(v)]


def tensors_equal(a, b):
    out = {}
    for c in t2.SERIES_CHANNELS:
        x, y = a[c], b[c]
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        diff = np.abs(np.nan_to_num(x, nan=0) - np.nan_to_num(y, nan=0))
        out[c] = (float(same.mean()) if same.size else 1.0, float(diff.max()) if diff.size else 0.0)
    return out


def wl_equal(a, b, sel=None):
    for k in t2.WL_FEATURES:
        x = a[k] if sel is None else a[k][sel]
        if not np.array_equal(x, b[k], equal_nan=True):
            return False, k
    return True, None


def leak_tests_series(hs, ts_, ctx, T_idx, rng):
    """Tests 1+2: overwrite everything at/after T (so the labels change); features must not move."""
    res = []
    T_idx = np.asarray(T_idx[:10], np.int64)
    base = t2.build_wl_features(t2.tensor_from_series(hs, ts_, T_idx, USE_T), ctx, T0_MIN + T_idx * t2.STEP_MIN)
    for i, t in enumerate(T_idx):
        K = int(min(200, N_T - t))
        saved_h = {c: hs[c][t:t + K].copy() for c in t2.SERIES_CHANNELS}
        saved_t = {c: ts_[c][t + 1:t + K].copy() for c in t2.SERIES_CHANNELS} if ts_ is not None else None
        lab0, _ = t2.labels_from_series(hs, np.array([t]), ctx.thr)
        for c in t2.SERIES_CHANNELS:
            hs[c][t:t + K] = rng.uniform(0, 150, hs[c][t:t + K].shape).astype(np.float32)
            if ts_ is not None:
                ts_[c][t + 1:t + K] = rng.uniform(0, 150, ts_[c][t + 1:t + K].shape).astype(np.float32)
        lab1, _ = t2.labels_from_series(hs, np.array([t]), ctx.thr)
        f1 = t2.build_wl_features(t2.tensor_from_series(hs, ts_, np.array([t]), USE_T), ctx,
                                  T0_MIN + np.array([t]) * t2.STEP_MIN)
        ok, k = wl_equal(base, f1, sel=slice(i, i + 1))
        res.append({"origin_index": int(t), "features_unchanged": ok, "first_diff": k,
                    "labels_changed": not np.array_equal(lab0, lab1, equal_nan=True)})
        for c in t2.SERIES_CHANNELS:
            hs[c][t:t + K] = saved_h[c]
            if ts_ is not None:
                ts_[c][t + 1:t + K] = saved_t[c]
    return pd.DataFrame(res)


def clim_lodo_test(P, C, day0, ctx):
    """Test 5: the climatology of a fit row ignores its own day; non-fit days never enter the
    evaluation climatology; another fit day does enter it (the test is not vacuous)."""
    used = DAY_PART == "fit"
    fd = np.flatnonzero(used)
    nd = np.flatnonzero(~used)
    d, d2, du = int(fd[len(fd) // 2]), int(fd[0] if fd[0] != fd[len(fd) // 2] else fd[1]), int(nd[0])
    li = np.flatnonzero(ctx.target_mask)[:25]
    m = np.repeat((day0 + d) * 1440 + np.arange(0, 1440, 30), len(li))
    ll = np.tile(li, 48)
    base = t2.Climatology.from_days(P, C, day0, used).query(m, ll, lodo=True)[0]

    def perturbed(day):
        P2 = P.copy()
        P2[day] = C[day]                        # every known cell of that day becomes "queued"
        return t2.Climatology.from_days(P2, C, day0, used).query(m, ll, lodo=True)[0]

    return {"own_day_ignored": bool(np.array_equal(base, perturbed(d))),
            "non_fit_day_ignored": bool(np.array_equal(base, perturbed(du))),
            "other_fit_day_used": bool(not np.array_equal(base, perturbed(d2)))}


def process_panel(p, run_leak_tests=False):
    t_start = time.time()
    ctx = CTX[p]
    res = {"panel": p, "select_stats": [], "parts": {}, "audit": [], "leak": None, "clim_test": None, "official": None}
    hs, aud = t2.load_series(files_in_dates(DATE_MAP[("train_unmasked", p)], DATES), LAYER_COLMAP["train_unmasked"],
                             USED, ELIG_MODE, ctx, T0_MIN, N_T, f"{p}/train_unmasked")
    res["audit"].append(aud)
    ts_ = None
    if USE_T:
        ts_, aud_m = t2.load_series(files_in_dates(DATE_MAP[("train_masked", p)], DATES), LAYER_COLMAP["train_masked"],
                                    USED, ELIG_MODE, ctx, T0_MIN, N_T, f"{p}/train_masked")
        res["audit"].append(aud_m)
    tm = ctx.target_mask
    res["cell_missingness"] = {
        "panel": p,
        "unmasked_link_cells_with_eligible_speed": float((hs["nst"][:, tm] > 0).mean()),
        "masked_link_cells_with_eligible_speed": float((ts_["nst"][:, tm] > 0).mean()) if ts_ is not None else np.nan}
    # climatology: evaluation version (fit days only) and final version (all train days)
    P, C, day0 = t2.Climatology.day_arrays(hs["spd"], ctx.thr, T0_MIN)
    clim_eval = t2.Climatology.from_days(P, C, day0, DAY_PART == "fit")
    clim_final = t2.Climatology.from_days(P, C, day0, np.ones(n_days, bool))
    res["clim_final"] = t2.Climatology.from_totals(**clim_final.totals())
    if run_leak_tests:
        res["clim_test"] = clim_lodo_test(P, C, day0, ctx)
    dn1 = ctx.seq_dn[:, 0]
    for k, name in enumerate(PART_NAMES):
        allowed = SLOT_PART == CODE[name]
        if not allowed.any():
            continue
        rng = np.random.default_rng([SEED, ctx.panel_code, k])
        T_idx, conds, st = t2.select_proxy_windows(hs, ctx, 0, N_T, WINDOW_CFG, rng, allowed=allowed)
        st.update(panel=p, partition=name)
        res["select_stats"].append(st)
        if len(T_idx) == 0:
            continue
        assert (partition_of(T_idx) == name).all(), "window crosses a block boundary"
        T_min = T0_MIN + T_idx * t2.STEP_MIN
        ten = t2.tensor_from_series(hs, ts_, T_idx, USE_T)
        wl = t2.build_wl_features(ten, ctx, T_min)                 # inputs only
        lab, margin = t2.labels_from_series(hs, T_idx, ctx.thr)    # labels computed separately
        w, l, h = t2.target_rows_for_generated(len(T_idx), ctx)
        y = lab[w, l, h]
        known = ~np.isnan(y)
        wk, lk, hk = w[known], l[known], ((h[known] + 1) * t2.STEP_MIN).astype(np.int64)
        X = t2.rows_frame(wl, wk, lk, hk, ctx.panel_code, clim=clim_eval, T_min=T_min, seq_dn1=dn1,
                          lodo=(name == "fit"))
        cf = pd.DataFrame(t2.clim_rows(clim_final, T_min[wk], lk, hk, dn1, lodo=True), columns=t2.CLIM_FEATURES)
        win = pd.DataFrame({
            "window_key": [f"gen|{p}|{name}|{t2.minutes_to_text(m)}|{c}" for m, c in zip(T_min, conds)],
            "panel": p, "family_id": ctx.family_id, "condition": conds, "partition": name,
            "T": [t2.minutes_to_text(m) for m in T_min],
            "hist_coverage": (ten["nst"][:, tm, :t2.N_HIST] > 0).mean((1, 2)),
            "T_availability": (ten["nst"][:, tm, t2.SLOT_T] > 0).mean(1)})
        pa = t2.persistence_b_wl(ten, ctx.thr, A_EQUIV["max_age_min"], A_EQUIV["include_t"], A_EQUIV["agg"])
        rows = {"X": X, "clim_final": cf, "y": y[known].astype(np.float32), "margin": margin[w, l, h][known],
                "w_local": wk, "link": ctx.link_ids[lk], "horizon": hk.astype(np.int16), "A": pa[wk, lk]}
        res["parts"][name] = {"win": win, "rows": rows, "n_rows_total": int(len(y)),
                              "n_rows_unknown": int((~known).sum())}
        if run_leak_tests and res["leak"] is None:
            res["leak"] = leak_tests_series(hs, ts_, ctx, T_idx, np.random.default_rng(SEED))
        del ten, wl, lab, margin
    if (p, "train") in OFF:
        o = OFF[(p, "train")]
        T_idx_o = (o["T_min"] - T0_MIN) // t2.STEP_MIN
        inside = (T_idx_o - t2.N_HIST >= 0) & (T_idx_o + t2.N_H < N_T)
        cons = None
        if inside.any():
            ten_s = t2.tensor_from_series(hs, ts_, T_idx_o[inside], USE_T)
            cons = tensors_equal(ten_s, {c: v[inside] for c, v in o["tensor"].items()})
        lab_o, _ = t2.labels_from_series(hs, np.clip(T_idx_o, 0, N_T - 1), ctx.thr)
        lab_o[~inside] = np.nan
        res["official"] = {"y": lab_o[o["w_idx"], o["l_idx"], o["h_idx"]].astype(np.float32),
                           "window_partition": partition_of(T_idx_o), "consistency": cons,
                           "X_eval": t2.official_rows(o, ctx, clim_eval, lodo=True)}
    del hs, ts_, P, C, clim_eval, clim_final
    gc.collect()
    print(f"[{p}] done in {time.time() - t_start:.0f}s | RSS {t2.rss_gb():.1f} GB | "
          + ", ".join(f"{k}: {len(v['win'])} win / {len(v['rows']['y']):,} rows" for k, v in res["parts"].items()))
    return res

# ======================================================================
# Cell 14 - run all panels
RESULTS = {}
for i, p in enumerate(PANELS):
    RESULTS[p] = process_panel(p, run_leak_tests=(i == 0))
CLIM_FINAL = {p: r["clim_final"] for p, r in RESULTS.items()}
print(f"elapsed {time.time() - T_START:.0f}s")

# ======================================================================
# Cell 15 - audit, corridor-order coverage and window-count reports
LOAD_AUDIT = pd.DataFrame([a for r in RESULTS.values() for a in r["audit"]])
display(LOAD_AUDIT)
display(pd.DataFrame([r["cell_missingness"] for r in RESULTS.values()]).round(4))
SEQ_AUDIT = pd.DataFrame([{"panel": p, "target_links": int(CTX[p].target_mask.sum()),
                           "topology_targets_without_neighbour": int(((CTX[p].up_idx[:, 0] < 0) & (CTX[p].dn_idx[:, 0] < 0)
                                                                      & CTX[p].target_mask).sum()),
                           **{k: v for k, v in CTX[p].topo_info.items() if k.startswith(("seq", "order", "milepost"))}}
                          for p in PANELS])
print("Corridor-sequence neighbours (v2) vs topology neighbours (v1):")
display(SEQ_AUDIT)
SELECT_STATS = pd.DataFrame([s for r in RESULTS.values() for s in r["select_stats"]])
display(SELECT_STATS)
wins_ = [v["win"] for r in RESULTS.values() for v in r["parts"].values()]
if not wins_:
    raise RuntimeError("no proxy windows were generated; inspect SELECT_STATS")
WIN = pd.concat(wins_, ignore_index=True)
display(WIN.pivot_table(index="panel", columns=["partition", "condition"], values="window_key",
                        aggfunc="count", fill_value=0))
cov_rows = [{"panel": p, "partition": k, "rows_total": v["n_rows_total"], "rows_unknown_label": v["n_rows_unknown"],
             "label_coverage": 1 - v["n_rows_unknown"] / max(v["n_rows_total"], 1),
             "proxy_positive_share": float(v["rows"]["y"].mean()),
             "near_threshold_share_abs_lt_5kmh": float((np.abs(v["rows"]["margin"]) < 5).mean())}
            for p, r in RESULTS.items() for k, v in r["parts"].items()]
display(pd.DataFrame(cov_rows).round(4))
cons_rows = [{"panel": p, "channel": c, "equal_cell_share": v[0]}
             for p, r in RESULTS.items() if r["official"] and r["official"]["consistency"]
             for c, v in r["official"]["consistency"].items()]
if cons_rows:
    print("Official TRAIN windows: released window_history (+T row) vs the same cells rebuilt from train layers")
    display(pd.DataFrame(cons_rows).pivot(index="panel", columns="channel", values="equal_cell_share").round(4))

# ======================================================================
# Cell 16 - leakage tests
p0 = PANELS[0]
lt = RESULTS[p0]["leak"]
print("Tests 1+2 (series path, all v1 + v2 window features): overwrite all values at/after T -> labels change, features must not")
display(lt)
assert lt is not None and lt.features_unchanged.all(), "LEAK: features changed after perturbing post-T data"
assert lt.labels_changed.any(), "test ineffective: labels did not change"

key0 = next(k for k in ((p0, "train"), (p0, "validation")) if k in OFF)
d0, o0 = T2[key0], OFF[key0]
ctx0 = CTX[p0]
wid0 = o0["window_ids"][0]
T0w = d0["origins"][wid0]
hist0 = d0["hist"][d0["hist"].window_id == wid0]
fut = hist0.copy()
fut["_min"] = fut["_min"] + 120
fut["speed"] = 5.0
tl_fut = (o0["t_rows"].copy() if o0["t_rows"] is not None else hist0.drop(columns=["window_id"]).copy())
tl_fut["_min"] = T0w + 5
tl_fut["speed"] = 1.0
ten_a, dg_a = t2.tensor_from_records(hist0, o0["t_rows"], ctx0, [wid0], [T0w], ELIG_MODE, USE_T)
t_mix = pd.concat([x for x in (o0["t_rows"], tl_fut) if x is not None], ignore_index=True)
ten_b, dg_b = t2.tensor_from_records(pd.concat([hist0, fut], ignore_index=True), t_mix, ctx0, [wid0], [T0w],
                                     ELIG_MODE, USE_T)
fa = t2.build_wl_features(ten_a, ctx0, [T0w])
fb = t2.build_wl_features(ten_b, ctx0, [T0w])
ok1, k1 = wl_equal(fa, fb)
print("Test 1 (records path) post-T rows ignored:", ok1, "| history rows dropped after T:",
      dg_b["hist_rows_after_T_dropped"], "of", len(fut))
assert ok1 and dg_b["hist_rows_after_T_dropped"] == len(fut), f"LEAK via records path ({k1})"
twin = hist0.copy()
twin["window_id"] = "__twin__"
twin["speed"] = twin["speed"] + 25.0
ten_j, _ = t2.tensor_from_records(pd.concat([hist0, twin], ignore_index=True), o0["t_rows"], ctx0,
                                  [wid0, "__twin__"], [T0w, T0w], ELIG_MODE, USE_T)
fj = t2.build_wl_features(ten_j, ctx0, [T0w, T0w])
ok3, k3 = wl_equal(fj, fa, sel=slice(0, 1))
differs = not np.array_equal(fj["mean_60"][1], fa["mean_60"][0], equal_nan=True)
print("Test 3 overlapping windows isolated:", ok3, "| twin window differs:", differs)
assert ok3 and differs, f"LEAK between windows ({k3})"
fp_before = CTX[p0].fingerprint()
X_first = o0["X"].copy()
for q in PANELS[1:2]:
    kq = next(k for k in ((q, "train"), (q, "validation")) if k in T2)
    dq = T2[kq]
    _ = t2.prepare_official_panel(CTX[q], dq["index"], dq["template"], dq["origins"], dq["hist"], dq["raw"],
                                  OFF[kq]["t_rows"], ELIG_MODE, USE_T)
again = t2.prepare_official_panel(ctx0, d0["index"], d0["template"], d0["origins"], d0["hist"], d0["raw"],
                                  o0["t_rows"], ELIG_MODE, USE_T)
fresh_ctx = t2.PanelContext(REL, p0, ctx0.family_id, TARGET_LINKS[p0], PANEL_CODES[p0])
ok4 = (np.array_equal(X_first.to_numpy(), again["X"].to_numpy(), equal_nan=True)
       and fp_before == CTX[p0].fingerprint() == fresh_ctx.fingerprint())
print("Test 4 panel isolation (recomputed after another panel; fresh context identical):", ok4)
assert ok4, "state carried between panels"
ct = RESULTS[p0]["clim_test"]
print("Test 5 climatology (leave-own-day-out; evaluation climatology = fit days only):", ct)
assert ct["own_day_ignored"] and ct["non_fit_day_ignored"] and ct["other_fit_day_used"], "climatology leakage"
assert not any(k in t2.FEATURES for k in ("window_id", "link_id", "label", "y", "margin", "station_id", "condition"))
LEAK_REPORT = {"series_future_perturbation": bool(lt.features_unchanged.all()),
               "labels_changed_features_unchanged": bool(lt.labels_changed.any() and lt.features_unchanged.all()),
               "records_post_T_rows_ignored": bool(ok1), "overlapping_windows_isolated": bool(ok3),
               "panel_isolation": bool(ok4), "climatology_own_day_ignored": ct["own_day_ignored"],
               "climatology_non_fit_days_ignored": ct["non_fit_day_ignored"],
               "climatology_uses_train_layers_only": True}
print(LEAK_REPORT)

# ======================================================================
# Cell 17 - assemble blocks and window weights
def assemble(part_name):
    wins, Xs, cfs, ys, wcodes, A = [], [], [], [], [], []
    extra = {"horizon": [], "panel": [], "link": [], "margin": []}
    offset = 0
    for p, r in RESULTS.items():
        v = r["parts"].get(part_name)
        if v is None:
            continue
        wins.append(v["win"])
        Xs.append(v["rows"]["X"])
        cfs.append(v["rows"]["clim_final"])
        ys.append(v["rows"]["y"])
        A.append(v["rows"]["A"])
        wcodes.append(v["rows"]["w_local"] + offset)
        extra["horizon"].append(v["rows"]["horizon"])
        extra["panel"].append(np.full(len(v["rows"]["y"]), p, dtype=object))
        extra["link"].append(v["rows"]["link"])
        extra["margin"].append(v["rows"]["margin"])
        offset += len(v["win"])
    if not wins:
        return None
    return {"win": pd.concat(wins, ignore_index=True), "X": pd.concat(Xs, ignore_index=True),
            "clim_final": pd.concat(cfs, ignore_index=True), "y": np.concatenate(ys), "w": np.concatenate(wcodes),
            "A": np.concatenate(A), **{k: np.concatenate(v) for k, v in extra.items()}}


def window_weights(D):
    """Each window carries total weight 1/(#windows in its panel-condition group), spread over its
    labelled rows (mirrors the evaluator's equal weighting); normalised to mean 1."""
    win = D["win"]
    n_rows = np.bincount(D["w"], minlength=len(win)).astype(np.float64)
    grp = win.groupby(["panel", "condition"]).window_key.transform("count").to_numpy(np.float64)
    ww = 1.0 / (np.maximum(n_rows, 1) * grp)
    w = ww[D["w"]]
    return (w / w.mean()).astype(np.float32)


DATA = {k: assemble(k) for k in PART_NAMES}
DATA = {k: v for k, v in DATA.items() if v is not None}
for k in ("fit", "tune"):
    if k not in DATA:
        raise RuntimeError(f"block {k} has no windows")
for k, D in DATA.items():
    D["weight"] = window_weights(D)
    print(f"{k:8s} windows={len(D['win']):6,} rows={len(D['y']):11,} positive_share={D['y'].mean():.4f}")
print("features: REF", len(t2.REF_FEATURES), "| V2", len(t2.FEATURES), "| RSS", round(t2.rss_gb(), 2), "GB")

# official train windows as an evaluation set (only windows lying entirely in the offtest block)
off_parts = []
for p, r in RESULTS.items():
    ro = r["official"]
    if ro is None:
        continue
    o = OFF[(p, "train")]
    cond = T2[(p, "train")]["index"].set_index("window_id").condition
    keep_w = np.flatnonzero(ro["window_partition"] == "offtest")
    rsel = np.isin(o["w_idx"], keep_w)
    remap = -np.ones(len(o["window_ids"]), np.int64)
    remap[keep_w] = np.arange(len(keep_w))
    ids = [o["window_ids"][i] for i in keep_w]
    off_parts.append({"win": pd.DataFrame({"window_key": ids, "panel": p, "family_id": CTX[p].family_id,
                                           "condition": cond.reindex(ids).to_numpy()}),
                      "w": remap[o["w_idx"][rsel]], "y": ro["y"][rsel], "X": ro["X_eval"][rsel].reset_index(drop=True),
                      "A": o["pred_official"][rsel], "horizon": ((o["h_idx"][rsel] + 1) * t2.STEP_MIN).astype(np.int16),
                      "panel": np.full(int(rsel.sum()), p, dtype=object), "n_windows_total": len(o["window_ids"])})
OFFICIAL_EVAL = None
if off_parts and sum(len(x["win"]) for x in off_parts):
    starts = np.cumsum([0] + [len(x["win"]) for x in off_parts[:-1]])
    OFFICIAL_EVAL = {"win": pd.concat([x["win"] for x in off_parts], ignore_index=True),
                     "w": np.concatenate([x["w"] + s0 for x, s0 in zip(off_parts, starts)]),
                     **{k: np.concatenate([x[k] for x in off_parts]) for k in ("y", "A", "horizon", "panel")},
                     "X": pd.concat([x["X"] for x in off_parts], ignore_index=True)}
    print(f"official train windows usable as untouched evaluation: {len(OFFICIAL_EVAL['win'])} of "
          f"{sum(x['n_windows_total'] for x in off_parts)} (the rest cross a block boundary)")

# ======================================================================
# Cell 18 - train REF (v1 features) and V2 on fit; early stopping and thresholds on tune
def score(D, pred):
    return t2.score_proxy(D["win"], D["w"], D["y"], pred)


def summary_row(name, D, pred, extra=None):
    s = score(D, pred)
    pc = s["panel_condition"]
    row = {"method": name, "proxy_IoU": s["overall"],
           **{f"IoU_{c}": float(pc[pc.condition == c].proxy_IoU.mean()) for c in sorted(pc.condition.unique())},
           **t2.pooled_metrics(D["y"], pred), "windows": len(D["win"])}
    row.update(extra or {})
    return row, s


def best_threshold(D, prob, other=None, hc=None, fixed=None, grid=THRESHOLD_GRID):
    """Grid search of the proxy_IoU-maximising threshold (ties -> closest to 0.5).
    With hc/fixed: searches the threshold of group `other` with the other group's threshold fixed."""
    vals = []
    for th in grid:
        if hc is None:
            pred = (prob >= th).astype(np.int8)
        else:
            pred = t2.thresholded(prob, hc, {**fixed, other: th})
        vals.append(score(D, pred)["overall"])
    vals = np.asarray(vals)
    ties = np.flatnonzero(vals >= vals.max() - 1e-12)
    return float(grid[ties[np.abs(grid[ties] - 0.5).argmin()]]), pd.DataFrame({"threshold": grid, "proxy_IoU": vals})


fit, tune = DATA["fit"], DATA["tune"]
if len(np.unique(fit["y"])) < 2:
    raise RuntimeError("fit block has a single class; cannot train LightGBM")
MODELS = {}
for tag, feats in (("ref", t2.REF_FEATURES), ("v2", t2.FEATURES)):
    cat_idx = [feats.index(c) for c in t2.CATEGORICAL_FEATURES if c in feats]
    ds_fit = lgb.Dataset(fit["X"][feats].to_numpy(np.float32), label=fit["y"], weight=fit["weight"],
                         feature_name=feats, categorical_feature=cat_idx, free_raw_data=True)
    ds_tune = lgb.Dataset(tune["X"][feats].to_numpy(np.float32), label=tune["y"], weight=tune["weight"],
                          feature_name=feats, categorical_feature=cat_idx, reference=ds_fit, free_raw_data=True)
    t_fit = time.time()
    bst = lgb.train(LGB_PARAMS, ds_fit, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[ds_tune], valid_names=["tune"],
                    callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(250)])
    best_it = int(bst.best_iteration or bst.current_iteration())
    p_tune = bst.predict(tune["X"][feats].to_numpy(np.float32), num_iteration=best_it)
    MODELS[tag] = {"booster": bst, "features": feats, "best_iter": best_it, "p_tune": p_tune, "cat_idx": cat_idx}
    print(f"{tag}: {len(feats)} features, trained in {time.time() - t_fit:.0f}s, best iteration {best_it}")
    del ds_fit, ds_tune
    gc.collect()
hc_tune = tune["X"]["hist_cond"].to_numpy()
g_ref, THR_TABLE_REF = best_threshold(tune, MODELS["ref"]["p_tune"])
g_v2, THR_TABLE_V2 = best_threshold(tune, MODELS["v2"]["p_tune"])
# V2: one threshold for onset_like windows and one for the rest. The two groups occupy disjoint
# panel-condition cells, so optimising one with the other fixed is exact.
th_on, _ = best_threshold(tune, MODELS["v2"]["p_tune"], other="onset_like", hc=hc_tune,
                          fixed={"onset_like": g_v2, "other": g_v2})
th_ot, _ = best_threshold(tune, MODELS["v2"]["p_tune"], other="other", hc=hc_tune,
                          fixed={"onset_like": th_on, "other": g_v2})
THRESHOLDS = {"ref": {"onset_like": g_ref, "other": g_ref}, "v2": {"onset_like": th_on, "other": th_ot}}
print("thresholds:", THRESHOLDS, "| V2 single-threshold optimum:", g_v2)
for tag in MODELS:
    imp = pd.DataFrame({"feature": MODELS[tag]["features"],
                        "gain": MODELS[tag]["booster"].feature_importance("gain")}).sort_values("gain", ascending=False)
    MODELS[tag]["importance"] = imp
print("V2 top features by gain:")
display(MODELS["v2"]["importance"].head(30))
new_feats = set(t2.SEQ_FEATURES + t2.CLIM_FEATURES)
imp2 = MODELS["v2"]["importance"]
print(f"share of V2 gain from new features: {imp2[imp2.feature.isin(new_feats)].gain.sum() / imp2.gain.sum():.3f}")

# ======================================================================
# Cell 19 - select on tune (only tune is used for any decision)
def preds_for(D, tag, prob=None):
    if tag == "persistence_official":
        return D["A"].astype(np.int8)
    m = MODELS[tag]
    if prob is None:
        prob = m["booster"].predict(D["X"][m["features"]].to_numpy(np.float32), num_iteration=m["best_iter"])
    return t2.thresholded(prob, D["X"]["hist_cond"].to_numpy(), THRESHOLDS[tag])


tune_preds = {"persistence_official": preds_for(tune, "persistence_official"),
              "ref": preds_for(tune, "ref", MODELS["ref"]["p_tune"]),
              "v2_single_threshold": (MODELS["v2"]["p_tune"] >= g_v2).astype(np.int8),
              "v2": preds_for(tune, "v2", MODELS["v2"]["p_tune"])}
TUNE_COMPARISON = pd.DataFrame([summary_row(k, tune, v)[0] for k, v in tune_preds.items()])
display(TUNE_COMPARISON.round(4))
s_ref = float(TUNE_COMPARISON.set_index("method").loc["ref", "proxy_IoU"])
s_v2 = float(TUNE_COMPARISON.set_index("method").loc["v2", "proxy_IoU"])
SELECTED = "v2" if s_v2 > s_ref + SELECTION_MIN_GAIN else "ref"
print(f"SELECTED MODEL (tune proxy_IoU): {SELECTED}  (ref {s_ref:.4f} | v2 {s_v2:.4f})")
if SELECTED == "ref":
    print("V2 did NOT beat the v1-feature reference on tune; this is reported, not hidden.")

# ======================================================================
# Cell 20 - evaluation: test (untouched), holdout (re-used), offtest windows, official train windows
EVAL_SETS = {k: DATA[k] for k in ("test", "holdout", "offtest") if k in DATA}
if OFFICIAL_EVAL is not None:
    EVAL_SETS["official_train_windows"] = OFFICIAL_EVAL
METHODS = ["persistence_official", "ref", "v2"]
EVAL = {}
summary_rows, pc_rows, fam_rows, hz_rows, age_rows = [], [], [], [], []
for set_name, D in EVAL_SETS.items():
    EVAL[set_name] = {}
    ages = t2.age_bucket(D["X"]["last_age_min"].to_numpy())
    for mth in METHODS:
        pred = preds_for(D, mth)
        row, s = summary_row(mth, D, pred)
        EVAL[set_name][mth] = s
        summary_rows.append({"set": set_name, **row})
        pc_rows.append(s["panel_condition"].assign(set=set_name, method=mth))
        fam_rows.append(s["family"].assign(set=set_name, method=mth))
        for h in t2.HORIZONS:
            m_ = D["horizon"] == h
            hz_rows.append({"set": set_name, "method": mth, "horizon": h, **t2.pooled_metrics(D["y"][m_], pred[m_])})
        for a in np.unique(ages):
            m_ = ages == a
            age_rows.append({"set": set_name, "method": mth, "obs_age": a, "row_share": float(m_.mean()),
                             **t2.pooled_metrics(D["y"][m_], pred[m_])})
EVAL_SUMMARY = pd.DataFrame(summary_rows)
EVAL_PC = pd.concat(pc_rows, ignore_index=True)
EVAL_FAMILY = pd.concat(fam_rows, ignore_index=True)
EVAL_HORIZON = pd.DataFrame(hz_rows)
EVAL_AGE = pd.DataFrame(age_rows)


def gain_table(df, index, value):
    t = df.pivot_table(index=index, columns="method", values=value, aggfunc="first")
    t["gain_v2_vs_ref"] = t["v2"] - t["ref"]
    return t[["persistence_official", "ref", "v2", "gain_v2_vs_ref"]]


print("proxy_IoU (approximate labels; NOT the official S_queue). test = untouched; holdout = already inspected in v1;"
      " offtest / official_train_windows = days excluded from training of these evaluation models")
display(gain_table(EVAL_SUMMARY, "set", "proxy_IoU").round(4))
cond_cols = [c for c in EVAL_SUMMARY.columns if c.startswith("IoU_")]
for c in cond_cols:
    print(f"--- {c}")
    display(gain_table(EVAL_SUMMARY.dropna(subset=[c]), "set", c).round(4))
print("--- precision / recall")
display(EVAL_SUMMARY.pivot_table(index="set", columns="method", values=["precision", "recall"]).round(4))
print("--- road x condition")
display(gain_table(EVAL_PC, ["set", "panel", "condition"], "proxy_IoU").round(4))
print("--- family")
display(gain_table(EVAL_FAMILY, ["set", "family_id"], "proxy_IoU").round(4))
print("--- horizon (pooled cell IoU)")
display(gain_table(EVAL_HORIZON, ["set", "horizon"], "pooled_cell_IoU").round(4))
print("--- age of the latest own observation (pooled cell IoU)")
display(gain_table(EVAL_AGE, ["set", "obs_age"], "pooled_cell_IoU").round(4))
REGRESSIONS = gain_table(EVAL_PC, ["set", "panel", "condition"], "proxy_IoU")
REGRESSIONS = REGRESSIONS[REGRESSIONS.gain_v2_vs_ref < 0]
print(f"road x condition cells where V2 is below REF: {len(REGRESSIONS)}")
display(REGRESSIONS.round(4))

# ======================================================================
# Cell 21 - final refit of the selected model on all train blocks (frozen recipe)
SEL = MODELS[SELECTED]
SEL_FEATS = SEL["features"]
FINAL_ITER = SEL["best_iter"]
clim_pos = [SEL_FEATS.index(c) for c in t2.CLIM_FEATURES if c in SEL_FEATS]
mats, ys, all_win, all_w, off = [], [], [], [], 0
for k, D in DATA.items():
    M = D["X"][SEL_FEATS].to_numpy(np.float32, copy=True)
    for j in clim_pos:                     # final climatology (all train days, leave-own-day-out)
        M[:, j] = D["clim_final"][SEL_FEATS[j]].to_numpy(np.float32)
    mats.append(M)
    ys.append(D["y"])
    all_win.append(D["win"])
    all_w.append(D["w"] + off)
    off += len(D["win"])
X_all = np.vstack(mats)
del mats
y_all = np.concatenate(ys)
wt_all = window_weights({"win": pd.concat(all_win, ignore_index=True), "w": np.concatenate(all_w)})
gc.collect()
ds_all = lgb.Dataset(X_all, label=y_all, weight=wt_all, feature_name=SEL_FEATS, categorical_feature=SEL["cat_idx"],
                     free_raw_data=True)
t_fit = time.time()
booster_final = lgb.train(LGB_PARAMS, ds_all, num_boost_round=FINAL_ITER)
print(f"final {SELECTED} model: {FINAL_ITER} rounds on {len(y_all):,} rows ({len(DATA)} blocks) in "
      f"{time.time() - t_fit:.0f}s; thresholds reused: {THRESHOLDS[SELECTED]}. Evaluation numbers above belong to "
      "the evaluation models trained on fit only.")
del X_all, ds_all
gc.collect()

# ======================================================================
# Cell 22 - final predictions for validation / private (fingerprint for Notebook 2)
EXPECTED_PREDICTIONS, PRED_SHARES = {}, []
for split in ("validation", "private"):
    preds = []
    for p in PANELS:
        o = OFF[(p, split)]
        Xs = t2.official_rows(o, CTX[p], CLIM_FINAL[p])
        prob = booster_final.predict(Xs[SEL_FEATS].to_numpy(np.float32))
        pr = t2.thresholded(prob, Xs["hist_cond"].to_numpy(), THRESHOLDS[SELECTED])
        preds.append(pr)
        cond = T2[(p, split)]["template"].window_id.map(T2[(p, split)]["index"].set_index("window_id").condition)
        PRED_SHARES.append(pd.DataFrame({"split": split, "panel": p, "condition": cond.to_numpy(), "pred": pr,
                                         "persistence": o["pred_official"]}))
    EXPECTED_PREDICTIONS[split] = t2.predictions_digest(np.concatenate(preds))
    print(split, EXPECTED_PREDICTIONS[split])
PRED_SHARES = pd.concat(PRED_SHARES, ignore_index=True)
print("predicted positive share by split / road / condition (predictions, NOT accuracy):")
display(PRED_SHARES.groupby(["split", "panel", "condition"])[["pred", "persistence"]].mean().round(4))

# ======================================================================
# Cell 23 - write the bundle
booster_final.save_model(str(BUNDLE_DIR / "lgbm_final.txt"))
for tag, m in MODELS.items():
    m["booster"].save_model(str(BUNDLE_DIR / f"lgbm_eval_{tag}.txt"), num_iteration=m["best_iter"])
np.savez_compressed(BUNDLE_DIR / "climatology.npz",
                    **{f"{p}__{k}": v for p, c in CLIM_FINAL.items() for k, v in c.totals().items()})
METRICS = {"tune_comparison": TUNE_COMPARISON, "tune_thresholds_ref": THR_TABLE_REF, "tune_thresholds_v2": THR_TABLE_V2,
           "eval_summary": EVAL_SUMMARY, "eval_road_condition": EVAL_PC, "eval_family": EVAL_FAMILY,
           "eval_horizon": EVAL_HORIZON, "eval_obs_age": EVAL_AGE, "eval_regressions": REGRESSIONS.reset_index(),
           "importance_ref": MODELS["ref"]["importance"], "importance_v2": MODELS["v2"]["importance"],
           "window_counts": WIN.groupby(["panel", "partition", "condition"]).size().rename("windows").reset_index(),
           "selection_stats": SELECT_STATS, "load_audit": LOAD_AUDIT, "label_coverage": pd.DataFrame(cov_rows),
           "sequence_audit": SEQ_AUDIT, "released_window_audit": COND_AUDIT, "task2_tables": T2_SUMMARY,
           "date_table": DATE_TABLE, "prediction_shares": PRED_SHARES.groupby(["split", "panel", "condition"])[
               ["pred", "persistence"]].mean().reset_index()}
for name, df in METRICS.items():
    df.to_csv(BUNDLE_DIR / "metrics" / f"{name}.csv", index=False)
WIN.to_csv(BUNDLE_DIR / "metrics" / "generated_windows_catalog.csv", index=False)   # window meta, no labels
CONFIG = {
    "mode": MODE, "t2lib_version": t2.T2LIB_VERSION, "bundle_format": t2.BUNDLE_FORMAT,
    "official_commit": t2.OFFICIAL_COMMIT, "seed": SEED,
    "selected_model": SELECTED, "model_file": "lgbm_final.txt", "model_features": SEL_FEATS,
    "thresholds": THRESHOLDS[SELECTED], "final_rounds": FINAL_ITER, "params": LGB_PARAMS,
    "evaluation_models": {tag: {"file": f"lgbm_eval_{tag}.txt", "best_iteration": m["best_iter"],
                                "n_features": len(m["features"]), "thresholds": THRESHOLDS[tag]}
                          for tag, m in MODELS.items()},
    "tune_proxy_IoU": TUNE_COMPARISON.set_index("method").proxy_IoU.to_dict(),
    "eval_proxy_IoU": {s: {m: float(v["overall"]) for m, v in d.items()} for s, d in EVAL.items()},
    "features": t2.FEATURES, "ref_features": t2.REF_FEATURES, "categorical_features": t2.CATEGORICAL_FEATURES,
    "panel_codes": PANEL_CODES, "panels": PANELS, "participating_panels": ALL_PANELS,
    "used_channels": USED, "elig_mode": ELIG_MODE, "use_t": bool(USE_T),
    "layer_colmaps": LAYER_COLMAP, "hist_colmaps": HIST_COLMAP, "panel_fingerprints": FINGERPRINTS,
    "window_cfg": WINDOW_CFG, "threshold_grid": THRESHOLD_GRID.tolist(), "leakage_tests": LEAK_REPORT,
    "expected_predictions": EXPECTED_PREDICTIONS,
    "notes": ["proxy_IoU uses labels from noisy observed train speeds; it is not the official S_queue",
              "evaluation numbers belong to models trained on the fit block only; lgbm_final.txt is refit on all blocks",
              "climatology.npz holds per-link weekday/weekend slot counts from TRAIN days only"],
}
(BUNDLE_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2, default=float))
(BUNDLE_DIR / "feature_names.json").write_text(json.dumps(SEL_FEATS))
(BUNDLE_DIR / "categorical_mapping.json").write_text(json.dumps({"panel_code": PANEL_CODES}, indent=2))
(BUNDLE_DIR / "split_definitions.json").write_text(json.dumps(SPLIT_DEF, indent=2))
print("bundle files:", sorted(p.name for p in BUNDLE_DIR.iterdir()))

# ======================================================================
# Cell 24 - round-trip check (reload module, config, climatology and model from disk)
spec = importlib.util.spec_from_file_location("t2lib_roundtrip", BUNDLE_DIR / "t2lib.py")
t2r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2r)
cfg_r = json.loads((BUNDLE_DIR / "config.json").read_text())
assert cfg_r["features"] == t2r.FEATURES == t2.FEATURES and cfg_r["model_features"] == SEL_FEATS
bf = lgb.Booster(model_file=str(BUNDLE_DIR / "lgbm_final.txt"))
assert bf.feature_name() == SEL_FEATS, "model feature names differ"
npz = np.load(BUNDLE_DIR / "climatology.npz")
p_rt = PANELS[0]
clim_r = t2r.Climatology.from_totals(npz[f"{p_rt}__tp"], npz[f"{p_rt}__tc"], npz[f"{p_rt}__prior"])
key_rt = (p_rt, "train") if (p_rt, "train") in T2 else (p_rt, "validation")
d, o = T2[key_rt], OFF[key_rt]
ctx_r = t2r.PanelContext(REL, p_rt, CTX[p_rt].family_id, t2r.collect_target_links(REL, p_rt, FAMILY_MAP),
                         cfg_r["panel_codes"][p_rt])
assert ctx_r.fingerprint() == cfg_r["panel_fingerprints"][p_rt]
hist_r, raw_r = t2r.load_official_history(REL, p_rt, key_rt[1], cfg_r["hist_colmaps"][key_rt[1]], cfg_r["used_channels"])
o_r = t2r.prepare_official_panel(ctx_r, d["index"], d["template"], d["origins"], hist_r, raw_r, o["t_rows"],
                                 cfg_r["elig_mode"], cfg_r["use_t"], clim=clim_r)
X_mem = t2.official_rows(o, CTX[p_rt], CLIM_FINAL[p_rt])
assert np.array_equal(o_r["X"].to_numpy(), X_mem.to_numpy(), equal_nan=True), "features differ after reload"
p_mem = booster_final.predict(X_mem[SEL_FEATS].to_numpy(np.float32))
p_disk = bf.predict(o_r["X"][SEL_FEATS].to_numpy(np.float32))
print("round-trip max |prob diff|:", float(np.abs(p_mem - p_disk).max()))
assert np.allclose(p_mem, p_disk, atol=1e-9)
n_ref = min(5000, len(X_mem))
ref = d["template"].iloc[:n_ref][["window_id", "timestamp", "link_id"]].reset_index(drop=True)
ref.insert(0, "split", key_rt[1])
ref.insert(0, "panel", p_rt)
ref = pd.concat([ref, X_mem.iloc[:n_ref].reset_index(drop=True)], axis=1)
ref["prob_final"] = p_mem[:n_ref]
ref.to_parquet(BUNDLE_DIR / "roundtrip_reference.parquet", index=False)   # inputs + probabilities only, no labels
print("round-trip OK; reference sample saved:", n_ref, "rows from", key_rt)

# ======================================================================
# Cell 25 - manifest
files = sorted(p for p in BUNDLE_DIR.rglob("*") if p.is_file())
MANIFEST = {"bundle_format": t2.BUNDLE_FORMAT, "mode": MODE, "smoke": MODE != "full",
            "created_utc": pd.Timestamp.now(tz="UTC").strftime(t2.TS_FMT), "seed": SEED,
            "official_commit": t2.OFFICIAL_COMMIT, "t2lib_version": t2.T2LIB_VERSION,
            "package_versions": PKG_VERSIONS, "selected_model": SELECTED, "release_root_seen": str(REL),
            "files": {str(p.relative_to(BUNDLE_DIR)).replace("\\", "/"): t2.sha256_file(p) for p in files}}
(BUNDLE_DIR / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
print(json.dumps({k: v for k, v in MANIFEST.items() if k != "files"}, indent=2))
print(f"{len(files)} files hashed. Total runtime {time.time() - T_START:.0f}s")
if MODE != "full":
    print("SMOKE BUNDLE - Notebook 2 will refuse it for full inference.")
