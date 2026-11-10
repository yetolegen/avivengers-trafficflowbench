# Task 2 Notebook 2 v2 (predict) -- ONE CELL. Paste everything into a single Kaggle code cell and run.

# ======================================================================
# Cell 2 - configuration
import os
from pathlib import Path

RELEASE_ROOT = os.environ.get(
    "T2_RELEASE_ROOT",
    "/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public")
SEARCH_ROOT = os.environ.get("T2_SEARCH_ROOT", "/kaggle/input")
# Set BUNDLE_DIR to the attached Notebook-1 output, e.g. "/kaggle/input/task2-train/task2_bundle".
# None -> search SEARCH_ROOT for exactly one task2_bundle/manifest.json (ambiguity is an error).
BUNDLE_DIR = os.environ.get("T2_BUNDLE_DIR") or None
OUT_DIR = Path(os.environ.get("T2_OUT_DIR", "/kaggle/working"))
REF_DIR = OUT_DIR / "task2_reference"           # references + diagnostics (never named queue*.csv)

RUN_OFFICIAL_SCRIPT = True     # pipeline cross-check: pinned official persistence script vs our exact copy (internet on)
EXPECTED_ROWS = {"validation": 87_000, "private": 87_000}
EXPECTED_PERSISTENCE_POSITIVES = {"validation": 4_374, "private": 4_032}   # from the earlier official run
SPLITS_TO_RUN = ("validation", "private")

# ======================================================================
# Cell 3 - imports and bundle validation
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
try:
    display
except NameError:
    display = print
T_START = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)


def find_bundle(explicit, search_root):
    if explicit:
        p = Path(explicit)
        if not (p / "manifest.json").exists():
            raise FileNotFoundError(f"BUNDLE_DIR {p} has no manifest.json")
        return p
    hits = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        if "manifest.json" in filenames and Path(dirpath).name == "task2_bundle":
            hits.append(Path(dirpath))
        dirnames[:] = [d for d in dirnames if d not in ("corridors", "task1", "task4", "metrics") and not d.startswith(".")]
    local = OUT_DIR / "task2_bundle"            # Notebook 1 run earlier in this same session
    if not hits and (local / "manifest.json").exists():
        print(f"no task2_bundle attached under {search_root}; using Notebook 1's bundle from this session: {local}")
        return local
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one task2_bundle under {search_root} (or {local}), found {len(hits)}: "
                           f"{hits}. Run Notebook 1 first, or attach its output, or set BUNDLE_DIR explicitly.")
    return hits[0]


BUNDLE = find_bundle(BUNDLE_DIR, SEARCH_ROOT)
print("BUNDLE:", BUNDLE)
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text())
if MANIFEST.get("bundle_format") != "task2_bundle_v2":
    raise ValueError(f"bundle format {MANIFEST.get('bundle_format')} - this Notebook 2 needs a v2 bundle "
                     "(re-run the v2 Notebook 1)")
if MANIFEST.get("mode") != "full" or MANIFEST.get("smoke"):
    raise RuntimeError("this bundle was produced in SMOKE mode; re-run Notebook 1 with MODE='full'")
for rel_name, digest in MANIFEST["files"].items():
    f = BUNDLE / rel_name
    if not f.exists():
        raise FileNotFoundError(f"bundle incomplete: {rel_name} missing")
    if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        raise ValueError(f"bundle file {rel_name} does not match its manifest hash")
for need in ("t2lib.py", "config.json", "feature_names.json", "categorical_mapping.json",
             "split_definitions.json", "roundtrip_reference.parquet", "climatology.npz", "lgbm_final.txt"):
    if need not in MANIFEST["files"]:
        raise FileNotFoundError(f"bundle lacks {need}")
CFG = json.loads((BUNDLE / "config.json").read_text())

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("t2lib", BUNDLE / "t2lib.py")
t2 = importlib.util.module_from_spec(spec)
sys.modules["t2lib"] = t2
spec.loader.exec_module(t2)
if t2.T2LIB_VERSION != CFG["t2lib_version"] or t2.OFFICIAL_COMMIT != CFG["official_commit"]:
    raise ValueError("bundle module version/commit mismatch with config")
MODEL_FEATS = CFG["model_features"]
if CFG["features"] != t2.FEATURES or json.loads((BUNDLE / "feature_names.json").read_text()) != MODEL_FEATS \
        or not set(MODEL_FEATS) <= set(t2.FEATURES):
    raise ValueError("feature schema mismatch between config, feature_names.json and t2lib")
VERS = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__, "lightgbm": lgb.__version__}
for k, v in VERS.items():
    trained = MANIFEST["package_versions"].get(k)
    print(f"{k:9s} now {v:10s} trained {trained}" + ("" if v == trained else "   <-- differs (round-trip check below)"))

BOOSTER = lgb.Booster(model_file=str(BUNDLE / CFG["model_file"]))
if BOOSTER.feature_name() != MODEL_FEATS or BOOSTER.num_feature() != len(MODEL_FEATS):
    raise ValueError("model features differ from the bundle's model_features")
THRESHOLDS = CFG["thresholds"]
CLIM_NPZ = np.load(BUNDLE / "climatology.npz")
print("model:", CFG["selected_model"], f"({len(MODEL_FEATS)} features) | thresholds:", THRESHOLDS)
print("tune proxy_IoU (Notebook 1):", {k: round(v, 4) for k, v in CFG["tune_proxy_IoU"].items()})
print("evaluation proxy_IoU (Notebook 1, models trained on fit only):")
display(pd.DataFrame(CFG["eval_proxy_IoU"]).T.round(4))

# ======================================================================
# Cell 4 - release discovery, panels, contexts (fingerprints must match training)
REL = t2.find_release_root(RELEASE_ROOT, SEARCH_ROOT)
print("RELEASE ROOT:", REL)
avail = t2.discover_task2_panels(REL)
avail["complete"] = avail.window_index & avail.window_history & avail.template
complete = avail[avail.complete]
PANELS = sorted(p for p in complete.panel.unique()
                if {"validation", "private"} <= set(complete[complete.panel == p].split))
if PANELS != CFG["participating_panels"] or PANELS != CFG["panels"]:
    raise ValueError(f"panels {PANELS} differ from the bundle's {CFG['panels']}")
if any(p in t2.EXCLUDED_PANELS for p in PANELS):
    raise RuntimeError("excluded panel found among Task 2 panels")
FAMILY_MAP = t2.load_family_map(REL)
USED, ELIG_MODE, USE_T = CFG["used_channels"], CFG["elig_mode"], CFG["use_t"]
CTX = {}
for p in PANELS:
    fam = t2.read_window_index(REL, p, "validation", FAMILY_MAP).family_id.iloc[0]
    CTX[p] = t2.PanelContext(REL, p, fam, t2.collect_target_links(REL, p, FAMILY_MAP), CFG["panel_codes"][p])
    if CTX[p].fingerprint() != CFG["panel_fingerprints"][p]:
        raise ValueError(f"{p}: panel context (links/thresholds/topology/network) differs from training")
print("contexts rebuilt; fingerprints match for", PANELS)
CLIM = {p: t2.Climatology.from_totals(CLIM_NPZ[f"{p}__tp"], CLIM_NPZ[f"{p}__tc"], CLIM_NPZ[f"{p}__prior"])
        for p in PANELS}
for p in PANELS:
    if CLIM[p].tp.shape[2] != CTX[p].L:
        raise ValueError(f"{p}: climatology has {CLIM[p].tp.shape[2]} links, context {CTX[p].L}")
MASKED_DATE_MAP = {}
if USE_T:
    for split in SPLITS_TO_RUN:
        cm = CFG["layer_colmaps"].get(f"{split}_masked")
        if cm is None:
            raise FileNotFoundError(f"bundle expects the {split} masked layer (use_t=True)")
        for p in PANELS:
            files = t2.parquet_files(REL / "corridors" / p / split / "mainline_states_masked")
            if not files:
                raise FileNotFoundError(f"{p}/{split}: no mainline_states_masked files")
            cols = t2.parquet_columns(files[0])
            missing = [col for col in cm.values() if col not in cols]
            if missing:
                raise KeyError(f"{p}/{split} masked layer lacks columns {missing}")
            MASKED_DATE_MAP[(split, p)] = t2.scan_dates(files, cm)


def official_inputs(p, split):
    """Index, template, history, origins and T rows of one panel/split - inputs at or before T only."""
    idx = t2.read_window_index(REL, p, split, FAMILY_MAP)
    tpl = t2.restrict_template(t2.read_template(REL, p, split), idx, f"{p}/{split}")
    hist, raw = t2.load_official_history(REL, p, split, CFG["hist_colmaps"][split], USED)
    hmax = hist[hist._valid_ts].groupby("window_id")._min.max().to_dict()
    origins, osrc = t2.resolve_origins(idx, tpl, hmax, f"{p}/{split}")
    t_rows = None
    if USE_T:
        dm = MASKED_DATE_MAP.get((split, p))
        if dm is None:
            dm = t2.scan_dates(t2.parquet_files(REL / "corridors" / p / split / "mainline_states_masked"),
                               CFG["layer_colmaps"][f"{split}_masked"])
        T_list = list(origins.values())
        t_rows = t2.read_t_rows(t2.files_covering(dm, T_list), CFG["layer_colmaps"][f"{split}_masked"], USED, T_list)
    return idx, tpl, hist, raw, origins, osrc, t_rows

# ======================================================================
# Cell 5 - round-trip check against Notebook 1's saved reference rows
ref = pd.read_parquet(BUNDLE / "roundtrip_reference.parquet")
rp, rs = ref.panel.iloc[0], ref.split.iloc[0]
idx, tpl, hist, raw, origins, _, t_rows = official_inputs(rp, rs)
o = t2.prepare_official_panel(CTX[rp], idx, tpl, origins, hist, raw, t_rows, ELIG_MODE, USE_T, clim=CLIM[rp])
n = len(ref)
keys_ok = (tpl.iloc[:n][["window_id", "timestamp", "link_id"]].astype(str).to_numpy()
           == ref[["window_id", "timestamp", "link_id"]].astype(str).to_numpy()).all()
X_now = o["X"].iloc[:n].reset_index(drop=True)
same_X = np.array_equal(X_now.to_numpy(np.float32), ref[t2.FEATURES].to_numpy(np.float32), equal_nan=True)
print(f"round-trip on {n} rows of {rp}/{rs}: keys equal={keys_ok}, features equal={same_X}")
if not (keys_ok and same_X):
    raise AssertionError("features recomputed here differ from Notebook 1 - environment/data mismatch")
dp = float(np.abs(BOOSTER.predict(X_now[MODEL_FEATS].to_numpy(np.float32)) - ref.prob_final.to_numpy()).max())
print("max |probability difference| vs Notebook 1:", dp)
if dp > 1e-6:
    raise AssertionError("model probabilities differ from Notebook 1")

# ======================================================================
# Cell 6 - per-split inference
def run_split(split):
    if split not in ("validation", "private"):
        raise ValueError(split)
    outs, refs_A, diag_rows, row_meta, templates = [], [], [], [], []
    for p in PANELS:
        ctx = CTX[p]
        idx, tpl, hist, raw, origins, osrc, t_rows = official_inputs(p, split)
        o = t2.prepare_official_panel(ctx, idx, tpl, origins, hist, raw, t_rows, ELIG_MODE, USE_T, clim=CLIM[p])
        pA = o["pred_official"].astype(np.int8)
        prob = BOOSTER.predict(o["X"][MODEL_FEATS].to_numpy(np.float32))
        if not np.isfinite(prob).all():
            raise ValueError(f"{p}/{split}: non-finite probabilities")
        pred = t2.thresholded(prob, o["X"]["hist_cond"].to_numpy(), THRESHOLDS)
        base = tpl[["window_id", "timestamp", "link_id"]].reset_index(drop=True)
        outs.append(base.assign(queue_pred=pred))
        refs_A.append(base.assign(queue_pred=pA))
        templates.append(base)
        cond = idx.set_index("window_id").condition
        row_meta.append(pd.DataFrame({
            "panel": p, "window_id": base.window_id.to_numpy(), "condition": base.window_id.map(cond).to_numpy(),
            "horizon": (o["h_idx"] + 1) * t2.STEP_MIN, "pred": pred, "persistence": pA,
            "prob": prob, "hist_cond": o["X"]["hist_cond"].to_numpy(),
            "obs_age": t2.age_bucket(o["X"]["last_age_min"].to_numpy())}))
        diag_rows.append({"panel": p, "split": split, "origin_source": osrc, **o["diag"],
                          "pred_pos": int(pred.sum()), "persistence_pos": int(pA.sum()),
                          "disagree_vs_persistence": int((pred != pA).sum()),
                          "rows_no_own_obs": int((o["X"]["has_any_obs"] == 0).sum())})
    out = pd.concat(outs, ignore_index=True)
    out["queue_pred"] = out["queue_pred"].astype(np.int64)
    tpl_all = pd.concat(templates, ignore_index=True)
    t2.validate_queue_output(out, tpl_all, split)
    return {"out": out, "template": tpl_all, "ref_A": pd.concat(refs_A, ignore_index=True),
            "diag": pd.DataFrame(diag_rows), "rows": pd.concat(row_meta, ignore_index=True)}


def split_checks(split, r):
    out, tpl, rows = r["out"], r["template"], r["rows"]
    print(f"\n========== {split} ==========")
    print(f"rows {len(out):,} (template {len(tpl):,}, expected {EXPECTED_ROWS[split]:,}) | windows {out.window_id.nunique()}")
    if len(tpl) != EXPECTED_ROWS[split]:
        print(f"DISCREPANCY: template has {len(tpl):,} rows, expected {EXPECTED_ROWS[split]:,} "
              "(nothing truncated or fabricated)")
    tag = f"_{split}_"
    if not out.window_id.str.contains(tag, regex=False).all():
        print(f"NOTE: some window ids lack {tag!r}; membership is enforced through the split's own window_index")
    assert (rows.groupby("window_id").horizon.nunique() == 6).all(), "a window lacks one of the six horizons"
    pa = int(r["ref_A"].queue_pred.sum())
    exp = EXPECTED_PERSISTENCE_POSITIVES.get(split)
    print(f"exact official persistence positives: {pa:,} (earlier official run: {exp:,}) -> "
          + ("MATCH" if pa == exp else "MISMATCH - investigate before using"))
    print("positive share by panel / condition (predictions, NOT accuracy):")
    display(rows.groupby(["panel", "condition"]).agg(rows=("pred", "size"), pred_pos_share=("pred", "mean"),
                                                     persistence_pos_share=("persistence", "mean")).round(4))
    print("positive share by horizon:")
    display(rows.groupby("horizon").agg(pred_pos_share=("pred", "mean"),
                                        persistence_pos_share=("persistence", "mean")).round(4).T)
    print("age of the latest own observation (rows):")
    display(rows.obs_age.value_counts(normalize=True).round(4).to_frame("share").T)
    display(r["diag"])
    print("fallbacks: none (LightGBM consumes NaN natively; neighbour/climatology features cover missing links). "
          f"rows with no own observation in [T-60, T]: {int(r['diag'].rows_no_own_obs.sum()):,}")
    got = t2.predictions_digest(out.queue_pred.to_numpy())
    exp = CFG["expected_predictions"][split]
    same = got == exp
    print(f"reproduction of Notebook 1's predictions: {'MATCH' if same else 'MISMATCH'} "
          f"(positives {got['positives']:,} vs {exp['positives']:,})")
    if not same:
        raise AssertionError(f"{split}: predictions differ from the fingerprint recorded by Notebook 1")
    print(f"disagreement with official persistence: {int((rows.pred != rows.persistence).sum()):,} rows "
          f"({(rows.pred != rows.persistence).mean():.4f})")

# ======================================================================
# Cell 7 - run validation and private separately and save
RUNS = {}
for split in SPLITS_TO_RUN:
    t_s = time.time()
    RUNS[split] = r = run_split(split)
    split_checks(split, r)
    r["out"].to_csv(OUT_DIR / f"queue_{split}.csv", index=False)
    r["ref_A"].to_csv(REF_DIR / f"reference_persistence_official_{split}.csv", index=False)
    r["diag"].to_csv(REF_DIR / f"diagnostics_{split}.csv", index=False)
    back = pd.read_csv(OUT_DIR / f"queue_{split}.csv", dtype=str, keep_default_na=False)
    assert (back[["window_id", "timestamp", "link_id"]].to_numpy()
            == r["template"][["window_id", "timestamp", "link_id"]].astype(str).to_numpy()).all(), "text changed on write"
    print(f"{split} saved in {time.time() - t_s:.0f}s")

# ======================================================================
# Cell 8 - run the PINNED official persistence script per split and compare row-wise
if RUN_OFFICIAL_SCRIPT:
    repo = Path("/tmp/tfb_official_pinned") if Path("/tmp").is_dir() else REF_DIR / "tfb_official_pinned"
    try:
        if not (repo / ".git").exists():
            subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=60)
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", t2.OFFICIAL_REPO], check=True, timeout=60)
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin", t2.OFFICIAL_COMMIT],
                       check=True, timeout=180)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD"], check=True, timeout=60)
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != t2.OFFICIAL_COMMIT:
            raise RuntimeError(f"checked out {head}, expected {t2.OFFICIAL_COMMIT}")
        for split in SPLITS_TO_RUN:
            dest = REF_DIR / f"official_script_output_{split}.csv"
            r_ = subprocess.run([sys.executable, str(repo / "src/task2/build_task2_persistence_submission.py"),
                                 "--release-root", str(REL), "--split", split, "--output", str(dest)],
                                capture_output=True, text=True)
            print(r_.stdout[-500:], r_.stderr[-1500:])
            r_.check_returncode()
            off = pd.read_csv(dest, dtype=str)
            off["ts"] = pd.to_datetime(off.timestamp, utc=True)
            mine = RUNS[split]["ref_A"].copy()
            mine["ts"] = pd.to_datetime(mine.timestamp, utc=True)
            m = mine.merge(off.drop(columns=["timestamp"]), on=["window_id", "ts", "link_id"], how="outer",
                           suffixes=("_mine", "_off"), indicator=True, validate="one_to_one")
            counts = m["_merge"].value_counts().to_dict()
            both = m[m["_merge"] == "both"]
            same = float((both.queue_pred_mine.astype(int) == both.queue_pred_off.astype(int)).mean())
            print(f"{split}: key match {counts} | identical predictions on common keys: {same:.6f}")
            if counts.get("left_only", 0) or counts.get("right_only", 0) or same != 1.0:
                print(f"WARNING {split}: the exact persistence reproduction differs from the official script")
    except Exception as exc:     # internet off etc.: reported, never silently ignored
        print("official script comparison not completed:", repr(exc))

# ======================================================================
# Cell 9 - combined queue.csv and final checks
parts = [RUNS[s]["out"] for s in SPLITS_TO_RUN]
tpls = [RUNS[s]["template"] for s in SPLITS_TO_RUN]
if len(parts) == 2 and set(parts[0].window_id) & set(parts[1].window_id):
    raise ValueError("validation and private share window ids")
combined = pd.concat(parts, ignore_index=True)
t2.validate_queue_output(combined, pd.concat(tpls, ignore_index=True), "combined")
combined.to_csv(OUT_DIR / "queue.csv", index=False)
back = pd.read_csv(OUT_DIR / "queue.csv", dtype=str, keep_default_na=False)
assert list(back.columns) == t2.OUTPUT_COLUMNS and len(back) == len(combined)
assert back.queue_pred.isin(["0", "1"]).all()
exp_total = sum(EXPECTED_ROWS[s] for s in SPLITS_TO_RUN)
print(f"queue.csv rows {len(back):,} (expected {exp_total:,})" + ("" if len(back) == exp_total else "  <-- DISCREPANCY"))
summary = pd.DataFrame([{"split": s, "rows": len(RUNS[s]["out"]), "windows": RUNS[s]["out"].window_id.nunique(),
                         "pred_pos": int(RUNS[s]["out"].queue_pred.sum()),
                         "pred_pos_share": float(RUNS[s]["out"].queue_pred.mean()),
                         "persistence_pos": int(RUNS[s]["ref_A"].queue_pred.sum()),
                         "disagree_vs_persistence": int((RUNS[s]["out"].queue_pred
                                                         != RUNS[s]["ref_A"].queue_pred).sum())}
                        for s in SPLITS_TO_RUN])
display(summary)
summary.to_csv(REF_DIR / "summary.csv", index=False)
print("model used:", CFG["selected_model"], "| thresholds:", THRESHOLDS)
for f in sorted(list(OUT_DIR.glob("queue*.csv")) + list(REF_DIR.glob("*.csv"))):
    print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")
print(f"done in {time.time() - T_START:.0f}s. Nothing was submitted; merge queue.csv with your existing merger.")
