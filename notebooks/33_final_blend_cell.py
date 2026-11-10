# Final: average the selected Task 1 runs (models that make different errors), then density A, then merge.
import glob
T1_USE = ["task-1-gap"]   # filled in by the build script from the held-out checks
keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
files = {Path(p).parent.name: p for p in glob.glob("/kaggle/input/**/state.csv", recursive=True)}
print("state files found:", sorted(files))
assert set(T1_USE) <= set(files), (T1_USE, sorted(files))
base = None
for name in T1_USE:
    d = pd.read_csv(files[name], dtype={k: str for k in keys})
    if base is None:
        base = d[keys].copy(); sp = d.speed_kmh.to_numpy(float); fl = d.flow_vph.to_numpy(float)
    else:
        assert len(d) == len(base) and (d[keys].to_numpy() == base[keys].to_numpy()).all(), f"{name}: row order differs"
        sp += d.speed_kmh.to_numpy(float); fl += d.flow_vph.to_numpy(float)
base["speed_kmh"] = np.clip(sp / len(T1_USE), 1, None); base["flow_vph"] = np.clip(fl / len(T1_USE), 0, None)
assert np.isfinite(base[["speed_kmh", "flow_vph"]].to_numpy()).all()
BLEND = OUT / "state_blend.csv"
base.to_csv(BLEND, index=False)
print("Task 1 blend of", T1_USE, "->", len(base), "rows")
del base, sp, fl
