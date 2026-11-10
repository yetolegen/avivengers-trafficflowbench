QH_EARLY = 24  # slots T-36..T-13 come from the masked mainline timeline; T-12..T come from the window tensor


def hist_block(G, Ti, spd13):
    """(W, L, 37) speeds for slots T-36..T: masked timeline G for the early part, window tensor (12 hist + T) for the last hour."""
    idx = np.asarray(Ti, np.int64)[:, None] + np.arange(-36, -12)[None, :]
    e = G[np.clip(idx, 0, None)].astype(np.float32)
    e[idx < 0] = np.nan
    return np.concatenate([e.transpose(0, 2, 1), np.asarray(spd13, np.float32)], axis=2)


def hist_feats(spd, thr, ch, dn1, up1):
    """Causal 3-hour queue history per (window, link). Unobserved slots carry the last observed state forward."""
    W, L, S = spd.shape; t = np.arange(S)
    obs = ~np.isnan(spd)
    jam = obs & (np.nan_to_num(spd, nan=np.inf) <= np.nan_to_num(thr, nan=-1.0)[None, :, None])
    li = np.maximum.accumulate(np.where(obs, t, -1), axis=2)
    known = li >= 0
    st = known & np.take_along_axis(jam, np.maximum(li, 0), axis=2)
    out = {}
    last_not = np.where(known & ~st, t, -1).max(2)
    out['qh_age'] = np.where(known[..., -1], np.where(st[..., -1], (S - 1 - last_not) * 5.0, 0.0), np.nan)
    lastj = np.where(st, t, -1).max(2)
    out['qh_since'] = np.where(lastj >= 0, (S - 1 - lastj) * 5.0, np.nan)
    for name, sl in (('qh_share', slice(None)), ('qh_share_early', slice(0, QH_EARLY))):
        no = obs[..., sl].sum(2)
        out[name] = np.where(no > 0, jam[..., sl].sum(2) / np.maximum(no, 1), np.nan)

    def nb(a, idx):
        o = np.full_like(a, np.nan); ok = idx >= 0; o[:, ok] = a[:, idx[ok]]; return o
    for k in ('qh_age', 'qh_share'):
        out[k + '_dn'] = nb(out[k], dn1); out[k + '_up'] = nb(out[k], up1)

    n = len(ch); ar = np.arange(n)[None, :]

    def jd(s):
        jc = st[:, ch, s]
        nxt = np.minimum.accumulate(np.where(jc, ar, BIG)[:, ::-1], axis=1)[:, ::-1]
        prv = np.maximum.accumulate(np.where(jc, ar, -BIG), axis=1)
        d = np.full((W, L), np.nan); d[:, ch] = np.where(nxt < BIG, nxt - ar, np.nan)
        u = np.full((W, L), np.nan); u[:, ch] = np.where(prv > -BIG, ar - prv, np.nan)
        return d, u, jc.sum(1).astype(np.float64)
    J = {lag: jd(S - 1 - 12 * k) for k, lag in enumerate(('now', 'm60', 'm120', 'm180'))}
    for lag, (d, u, nj) in J.items():
        out[f'qh_jd_dn_{lag}'] = d; out[f'qh_jd_up_{lag}'] = u
        out[f'qh_nj_{lag}'] = np.broadcast_to(nj[:, None], (W, L))
    for lag, m in (('m60', 12), ('m180', 36)):
        out[f'qh_front_v_{lag}'] = (J[lag][0] - J['now'][0]) / m   # links per 5 min the jam front moved upstream
        out[f'qh_nj_d_{lag}'] = out['qh_nj_now'] - out[f'qh_nj_{lag}']
    return {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}

