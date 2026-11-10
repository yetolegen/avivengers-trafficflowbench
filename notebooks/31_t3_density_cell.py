# Task 3 density post-processing (own implementation).
# Task 3 checks vehicle conservation on density k = q / v. Next to a hidden cell, the released readings move k by
# well under 1% per 5 min, so pulling the predicted k towards its released neighbours cuts the conservation error.
#   A: both t-5 and t+5 released -> k = (1 - 2*W_A) * k_hat + W_A * (k_prev + k_next); speed kept, flow = k * v
#   B: k_interp = linear interpolation between the nearest released readings before/after (run <= MAX_RUN);
#      flow moved in log space: q = q_hat * exp(W_B * (log k_interp - log k_hat)); speed kept
# Both: in congested cells (v < 0.58 vf) flow above the triangular-FD envelope is nudged 4% towards it.
import glob
VARIANT = "A"            # set per notebook
W_A, W_B, MAX_RUN = 0.03, 0.5, 2

STATE_IN = Path(sorted(glob.glob("/kaggle/input/**/state.csv", recursive=True))[0])
print("state input:", STATE_IN)
st = pd.read_csv(STATE_IN, dtype={"panel": str, "timestamp": str, "station_id": str, "link_id": str, "mask_regime": str})
n0 = len(st)
st["tmin"] = pd.to_datetime(st.timestamp, utc=True).astype("int64") // 60_000_000_000
v_hat = st.speed_kmh.to_numpy(float); q_hat = st.flow_vph.to_numpy(float)
q_new = q_hat.copy()
k_hat = q_hat / np.maximum(v_hat, 5.0)
links_all = st.link_id.to_numpy()


def released(panel):
    parts = []
    for split in ("validation", "private"):
        for f in sorted((REL / "corridors" / panel / split / "mainline_states_masked").rglob("*.parquet")):
            parts.append(pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]))
    d = pd.concat(parts, ignore_index=True)
    d["link_id"] = d.link_id.astype(str)
    ok = d.is_score_eligible.astype(bool) & np.isfinite(d.speed_kmh) & np.isfinite(d.flow_vph) & (d.speed_kmh > 1)
    d = d[ok].copy()
    d["tmin"] = pd.to_datetime(d.timestamp, utc=True).astype("int64") // 60_000_000_000
    d["k"] = d.flow_vph / np.maximum(d.speed_kmh, 5.0)
    return d.groupby(["link_id", "tmin"], as_index=False).k.mean()


stats = []
for panel, idx in st.groupby("panel").indices.items():
    obs = released(panel)
    s = pd.DataFrame({"link_id": links_all[idx], "tmin": st.tmin.to_numpy()[idx], "_i": idx})
    kh = k_hat[idx]
    if VARIANT == "A":
        prev = s.merge(obs.assign(tmin=obs.tmin + 5), on=["link_id", "tmin"], how="left").k.to_numpy()
        nxt = s.merge(obs.assign(tmin=obs.tmin - 5), on=["link_id", "tmin"], how="left").k.to_numpy()
        hit = np.isfinite(prev) & np.isfinite(nxt)
        k_t = np.where(hit, (1 - 2 * W_A) * kh + W_A * (np.nan_to_num(prev) + np.nan_to_num(nxt)), kh)
        q_p = k_t * v_hat[idx]
    else:
        ss = s.sort_values("tmin")
        o = obs.sort_values("tmin")
        bk = pd.merge_asof(ss, o.rename(columns={"tmin": "tp", "k": "kp"}), left_on="tmin", right_on="tp", by="link_id",
                           direction="backward", allow_exact_matches=False).set_index("_i")
        fw = pd.merge_asof(ss, o.rename(columns={"tmin": "tn", "k": "kn"}), left_on="tmin", right_on="tn", by="link_id",
                           direction="forward", allow_exact_matches=False).set_index("_i")
        tp, kp = bk.tp.reindex(idx).to_numpy(float), bk.kp.reindex(idx).to_numpy(float)
        tn, kn = fw.tn.reindex(idx).to_numpy(float), fw.kn.reindex(idx).to_numpy(float)
        tt = s.tmin.to_numpy(float)
        run_len = (tn - tp) / 5 - 1   # not "run": that name is the merge helper
        hit = np.isfinite(kp) & np.isfinite(kn) & (run_len <= MAX_RUN) & (kp > 0) & (kn > 0) & (kh > 0)
        frac = np.where(hit, (tt - tp) / np.maximum(tn - tp, 1), 0)
        k_i = kp * (1 - frac) + kn * frac
        d = np.where(hit, np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(kh, 1e-9)), 0.0)
        q_p = q_hat[idx] * np.exp(W_B * d)
    # triangular-FD envelope guard in congested cells
    fd = pd.read_csv(REL / "corridors" / panel / "network" / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    g = lambda c: fd[c].reindex(links_all[idx]).to_numpy(float) if c in fd else np.full(len(idx), np.nan)
    cap, kc, kj, vf = (g(c) for c in ("capacity_vph", "critical_density", "k_jam", "free_speed_kmh"))
    w_ = cap / np.maximum(kj - kc, 10.0)
    qmax = np.minimum(1.06 * cap, (w_ * kj) / (1.0 + w_ / np.maximum(v_hat[idx], 5.0)))
    over = np.isfinite(qmax) & (v_hat[idx] < 0.58 * vf) & (q_p > qmax)
    q_p = np.where(over, 0.96 * q_p + 0.04 * qmax, q_p)
    q_new[idx] = np.clip(q_p, 0, None)
    stats.append({"panel": panel, "cells": len(idx), "anchored": round(float(hit.mean()), 3), "fd_capped": int(over.sum()),
                  "mean_abs_flow_change": round(float(np.abs(q_new[idx] - q_hat[idx]).mean()), 2)})
print(pd.DataFrame(stats).to_string(index=False))
assert np.isfinite(q_new).all() and len(st) == n0
st["flow_vph"] = q_new
state = OUT / "state.csv"
st.drop(columns=["tmin"]).to_csv(state, index=False)
print("variant", VARIANT, "| rows", len(st), "| cells changed", int((np.abs(q_new - q_hat) > 1e-6).sum()))
