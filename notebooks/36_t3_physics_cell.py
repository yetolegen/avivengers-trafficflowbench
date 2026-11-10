# Task 3 run flattening (own implementation).
# Inside each run of consecutive hidden cells (same station, 5-min steps), pull the predicted density k = q/v 20% toward
# the run's mean predicted density; speed is kept and flow = k * v. Conservation (S_LWR, 2/3 of Task 3) scores 5-min
# changes of k*L, and inside hidden runs our density changes were noisier than the truth's. Local held-out check
# (10 panels x 10 train days, scorer maths, boundary flux emulated from truth): S_LWR 0.5877 -> 0.5927,
# flow RMSE/lane 32.74 -> 33.11, total +0.0004, positive on all 10 panels. Runs of length 1 are unchanged.
# Inputs: STATE_IN (a state.csv, e.g. BLEND from the blend cell). Output: OUT / "state.csv".
GAMMA = 0.2

def flatten_runs(st, gamma=GAMMA):
    t = pd.to_datetime(st.timestamp, utc=True)
    o = np.lexsort((t.to_numpy(), st.mask_regime.to_numpy(), st.station_id.to_numpy(), st.panel.to_numpy()))
    s = st.iloc[o]; ts = t.iloc[o]
    same = (s.panel.to_numpy()[1:] == s.panel.to_numpy()[:-1]) & (s.station_id.to_numpy()[1:] == s.station_id.to_numpy()[:-1]) & \
        (s.mask_regime.to_numpy()[1:] == s.mask_regime.to_numpy()[:-1]) & (np.diff(ts.to_numpy()) == np.timedelta64(5, 'm'))
    rid = np.cumsum(np.r_[True, ~same])
    v = np.clip(s.speed_kmh.to_numpy(float), 1, None); k = s.flow_vph.to_numpy(float) / v
    km = pd.Series(k).groupby(rid).transform('mean').to_numpy()
    q = np.empty(len(st)); q[o] = ((1 - gamma) * k + gamma * km) * v
    return np.clip(q, 0, None), rid

if "STATE_IN" in globals():
    print("state input:", STATE_IN)
    st = pd.read_csv(STATE_IN, dtype={"panel": str, "timestamp": str, "station_id": str, "link_id": str, "mask_regime": str})
    n0 = len(st); q0 = st.flow_vph.to_numpy(float)
    q_new, rid = flatten_runs(st)
    assert np.isfinite(q_new).all() and len(q_new) == n0
    st["flow_vph"] = q_new
    st.to_csv(OUT / "state.csv", index=False)
    print("run flatten gamma", GAMMA, "| rows", n0, "| runs", int(rid.max()), "| mean |flow change|", round(float(np.abs(q_new - q0).mean()), 2))
