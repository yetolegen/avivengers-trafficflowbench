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
SEQ_K = 5
CLIM_HALF = 2
CLIM_ALPHA = 5.0
OFFICIAL_REPO = "https://github.com/jacky850/trafficflowbench-public"
OFFICIAL_COMMIT = "f9e3fbc35c066419726987e470448f33d4c72fd9"

STEP_MIN = 5
N_HIST = 12
SLOT_T = 12
N_SLOTS = 13
HORIZONS = (5, 10, 15, 20, 25, 30)
N_H = len(HORIZONS)
SLOTS_PER_DAY = 24 * 60 // STEP_MIN
EXCLUDED_PANELS = ("D12_I405_N", "D12_I405_S")
OFFICIAL_UNKNOWN_LINK_THRESHOLD = 63.0
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
    s = pd.Series(series)
    if s.dtype == bool:
        return s.to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    m = s.astype(str).str.strip().str.lower()
    return m.isin(["true", "1", "t", "yes", "y"]).to_numpy()


def to_minutes(values):
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


def find_release_root(explicit=None, search_root="/kaggle/input", max_depth=6):
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
    return sorted(f for f in Path(d).rglob("*") if f.is_file() and not f.name.startswith(("_", "."))
                  and not f.name.endswith((".crc", ".json", ".txt")))


def parquet_files(directory):
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


def official_thresholds(panel_dir):
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
    src = np.full(len(idx), 2, np.float32)
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
    order = order[np.argsort(sign * key[order], kind="stable")]
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


def read_obs(path, colmap, used, keep_minutes=None):
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
    if elig_mode == "column":
        return to_bool(df["elig_raw"])
    if elig_mode == "derived_pct75":
        return (df["pct"].to_numpy() >= 75) & ~np.isnan(df["speed"].to_numpy())
    raise ValueError(f"unknown elig_mode {elig_mode}")


def aggregate_cells(key, n_keys, df, elig, thr_row):
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
    res = {}
    for f in files:
        t = read_parquet_columns(f, [colmap["timestamp"]])[colmap["timestamp"]]
        m, v, _ = to_minutes(t.drop_duplicates())
        days = np.unique(m[v] // (24 * 60))
        res[str(f)] = [pd.Timestamp(int(d) * 86400, unit="s", tz="UTC").strftime("%Y-%m-%d") for d in days]
    return res


def load_series(files, colmap, used, elig_mode, ctx, t0_min, n_t, audit_name=""):
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


def tensor_from_series(hist, tlayer, T_idx, use_t=True):
    T_idx = np.asarray(T_idx, np.int64)
    off = np.arange(-N_HIST, 0, dtype=np.int64)
    idx = T_idx[:, None] + off[None, :]
    if len(T_idx) and idx.min() < 0:
        raise IndexError("window history starts before the series")
    out = {}
    for c in SERIES_CHANNELS:
        h = hist[c][idx]
        if use_t and tlayer is not None:
            t = tlayer[c][T_idx][:, None, :]
        else:
            fill = 0.0 if c in ("nst", "qoff") else np.nan
            t = np.full((len(T_idx), 1, h.shape[2]), fill, np.float32)
        out[c] = np.ascontiguousarray(np.concatenate([h, t], axis=1).transpose(0, 2, 1))
    return out


def tensor_from_records(hist_df, t_df, ctx, window_ids, T_min, elig_mode, use_t=True):
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
    T_idx = np.asarray(T_idx, np.int64)
    idx = T_idx[:, None] + np.arange(1, N_H + 1)[None, :]
    n_t = hist["spd"].shape[0]
    inside = idx < n_t
    spd = hist["spd"][np.clip(idx, 0, n_t - 1)]
    spd = np.where(inside[:, :, None], spd, np.nan).transpose(0, 2, 1)
    lab = np.full(spd.shape, np.nan, np.float32)
    known = ~np.isnan(spd)
    lab[known] = (spd <= thr[None, :, None])[known]
    margin = (spd - thr[None, :, None]).astype(np.float32)
    return lab, margin


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
REF_FEATURES = WL_FEATURES + ["horizon", "panel_code"]
SEQ_FEATURES = [
    "sq_dn1_r", "sq_dn2_r", "sq_dn3_r", "sq_up1_r", "sq_up2_r", "sq_up3_r",
    "sq_dn_min5", "sq_up_min5", "sq_dn_fq5", "sq_up_fq5", "sq_dn_qdist", "sq_up_qdist",
    "sq_dn1_t15", "sq_up1_t15", "sq_dn_qhist3", "sq_up_qhist3", "sq_dn_rmin30_3",
    "sq_n_dn", "sq_n_up", "r_filled", "run_state_min", "win_frac_q_m30", "win_dfrac_q30", "hist_cond",
]
WL_FEATURES = WL_FEATURES + SEQ_FEATURES
CLIM_FEATURES = ["clim_q_h", "clim_q_T", "clim_dq", "clim_n_h", "clim_q_dn1_h"]
FEATURES = WL_FEATURES + CLIM_FEATURES + ["horizon", "panel_code"]
CATEGORICAL_FEATURES = ["panel_code"]
HIST_COND_ONSET = 0


def _last_valid(x, valid):
    S = x.shape[-1]
    idx = np.where(valid, np.arange(S), -1)
    ff = np.maximum.accumulate(idx, axis=-1)
    ls = ff[..., -1]
    val = np.take_along_axis(x, np.clip(ls, 0, None)[..., None], axis=-1)[..., 0]
    val = np.where(ls >= 0, val, np.nan).astype(np.float32)
    return ff, ls, val


def _trend(x, valid, k_slots):
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
    valid_nb = nb_idx >= 0
    g = v[:, np.clip(nb_idx, 0, None)]
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
    dow = ((T_min // (24 * 60)) + 3) % 7
    for name, val in (("tod_sin", np.sin(2 * np.pi * mod / 1440)), ("tod_cos", np.cos(2 * np.pi * mod / 1440)),
                      ("tod_hour", mod / 60.0), ("weekday", dow), ("is_weekend", dow >= 5)):
        f[name] = np.broadcast_to(np.asarray(val, np.float32)[:, None], (W, L))
    _seq_features(f, tensor, ctx, spd, v, r, t15)
    missing, extra = set(WL_FEATURES) - set(f), set(f) - set(WL_FEATURES)
    if missing or extra:
        raise RuntimeError(f"feature builder mismatch: missing={missing} extra={extra}")
    return {k: np.ascontiguousarray(f[k], dtype=np.float32) for k in WL_FEATURES}


def _take_link(v, idx_col):
    g = v[:, np.clip(idx_col, 0, None)]
    return np.where(idx_col[None, :] >= 0, g, np.nan).astype(np.float32)


def _seq_features(f, tensor, ctx, spd, v, r, t15):
    W, L = r.shape
    sq = {"dn": ctx.seq_dn, "up": ctx.seq_up}
    for pre, idx in sq.items():
        for k in (1, 2, 3):
            f[f"sq_{pre}{k}_r"] = _take_link(r, idx[:, k - 1])
        st = np.stack([_take_link(r, idx[:, k]) for k in range(idx.shape[1])], -1)
        okk = ~np.isnan(st)
        mn = np.where(okk, st, np.inf).min(-1)
        f[f"sq_{pre}_min5"] = np.where(np.isfinite(mn), mn, np.nan).astype(np.float32)
        nv = okk.sum(-1)
        qq = (okk & (np.nan_to_num(st, nan=9.0) <= 1.0)).sum(-1)
        f[f"sq_{pre}_fq5"] = np.where(nv > 0, qq / np.maximum(nv, 1), np.nan).astype(np.float32)
        dist = np.full((W, L), np.nan, np.float32)
        for k in range(idx.shape[1] - 1, -1, -1):
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
    tm = ctx.target_mask[None, :]
    s30 = spd[..., N_HIST - 6]
    v30 = ~np.isnan(s30) & tm
    c30 = v30.sum(1)
    q30 = (v30 & (np.nan_to_num(s30, nan=np.inf) <= thr)).sum(1)
    frac30 = np.where(c30 > 0, q30 / np.maximum(c30, 1), np.nan)
    f["win_frac_q_m30"] = np.broadcast_to(np.asarray(frac30, np.float32)[:, None], (W, L))
    f["win_dfrac_q30"] = (f["win_frac_q"] - f["win_frac_q_m30"]).astype(np.float32)
    q = tensor["qoff"][:, ctx.target_mask, :]
    any_hist = q[..., :N_HIST].max((1, 2)) > 0 if q.size else np.zeros(W, bool)
    at_last = q[..., N_HIST - 1].max(1) > 0 if q.size else np.zeros(W, bool)
    hc = np.where(~any_hist, 0, np.where(at_last, 1, 2)).astype(np.float32)
    f["hist_cond"] = np.broadcast_to(hc[:, None], (W, L))


def _is_weekend_day(epoch_day):
    return (((np.asarray(epoch_day, np.int64) + 3) % 7) >= 5).astype(np.int64)


class Climatology:

    def __init__(self, tp, tc, prior, day0=None, P=None, C=None, used=None):
        self.tp, self.tc, self.prior = tp, tc, prior
        self.day0, self.P, self.C, self.used = day0, P, C, used

    @staticmethod
    def day_arrays(spd, thr, t0_min):
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
        obj.Pd = P.sum(1, dtype=np.float64)
        obj.Cd = C.sum(1, dtype=np.float64)
        return obj

    @classmethod
    def from_totals(cls, tp, tc, prior):
        return cls(np.asarray(tp, np.float32), np.asarray(tc, np.float32), np.asarray(prior, np.float32))

    def totals(self):
        return {"tp": self.tp, "tc": self.tc, "prior": self.prior}

    def query(self, minutes, l_idx, lodo=False):
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
    tl = np.flatnonzero(ctx.target_mask)
    w = np.repeat(np.arange(n_windows), len(tl) * N_H)
    l = np.tile(np.repeat(tl, N_H), n_windows)
    h = np.tile(np.arange(N_H), n_windows * len(tl))
    return w, l, h


def template_rows(template, window_ids, T_by_window, ctx):
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


OFFICIAL_HISTORY_COLUMNS = ["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]


def official_persistence_exact(hist_raw, template, thr_dict):
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


def _roll_count(x, start_off, end_off):
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64))])
    t = np.arange(n)
    a, b = t + start_off, t + end_off
    ok = (a >= 0) & (b < n)
    s = np.where(ok, c[np.clip(b + 1, 0, n)] - c[np.clip(a, 0, n)], 0.0)
    return s, ok


def select_proxy_windows(hist, ctx, part_lo, part_hi, cfg, rng, allowed=None):
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
    if allowed is not None:
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
            lab, _ = labels_from_series(hist, cc, ctx.thr)
            pred = (hist["qoff"][cc - 1] > 0)[:, :, None]
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


def score_proxy(win_table, row_w, label, pred):
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


OUTPUT_COLUMNS = ["window_id", "timestamp", "link_id", "queue_pred"]


def validate_queue_output(out, template, where=""):
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


def collect_target_links(rel, panel, family_map=None):
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
    days = {pd.Timestamp(int(m) * 60_000_000_000, tz="UTC").strftime("%Y-%m-%d") for m in minutes}
    return [Path(f) for f, ds in sorted(date_map.items()) if days & set(ds)]


def read_t_rows(files, colmap, used, T_minutes):
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
    return rows_frame(o["wl"], o["w_idx"], o["l_idx"], (o["h_idx"] + 1) * STEP_MIN, ctx.panel_code,
                      clim=clim, T_min=o["T_min"], seq_dn1=ctx.seq_dn[:, 0], lodo=lodo)


def thresholded(prob, hist_cond, thresholds):
    hc = np.asarray(hist_cond)
    th = np.where(hc == HIST_COND_ONSET, float(thresholds["onset_like"]), float(thresholds["other"]))
    return (np.asarray(prob) >= th).astype(np.int8)


def predictions_digest(pred):
    a = np.ascontiguousarray(np.asarray(pred, np.int8))
    return {"rows": int(len(a)), "positives": int(a.sum()), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def prepare_official_panel(ctx, index, template, origins, hist, raw, t_rows, elig_mode, use_t, clim=None):
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
    tm = ctx.target_mask
    q = tensor["qoff"][:, tm, :]
    any_hist = q[..., :N_HIST].max((1, 2)) > 0
    at_last = q[..., N_HIST - 1].max(1) > 0
    cond = np.where(~any_hist, "onset_like", np.where(at_last, "ongoing_like", "other"))
    cov = (tensor["nst"][:, tm, :N_HIST] > 0).mean((1, 2))
    tav = (tensor["nst"][:, tm, SLOT_T] > 0).mean(1)
    return pd.DataFrame({"history_condition": cond, "hist_coverage": cov, "T_availability": tav})
