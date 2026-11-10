import numpy as np
BIG = 10 ** 6

def chain_of(ctx):
    L = ctx.L; dn = ctx.seq_dn[:, 0]; up = ctx.seq_up[:, 0]
    pos = np.full(L, -1, np.int64); order = []
    for h0 in np.flatnonzero((up < 0) & (dn >= 0)):
        cur = int(h0)
        while cur >= 0 and pos[cur] < 0:
            order.append(cur); pos[cur] = len(order) - 1; cur = int(dn[cur])
    return np.array(order, np.int64)

def head_totals(spd_u, thr, ch, day_mask_rows):
    S = spd_u[:, ch]; obs = ~np.isnan(S)
    jam = obs & (np.nan_to_num(S, nan=np.inf) <= thr[ch][None, :])
    head = jam[:, :-1] & obs[:, 1:] & ~jam[:, 1:]
    tot = np.zeros(len(ch), np.float64); tot[:-1] = head[day_mask_rows].sum(0)
    return tot

def new_features(wl, tensor, ctx, ch, bn_set, head_rate, last_valid):
    W, L = wl['last_ratio'].shape; n = len(ch); ar = np.arange(n)[None, :]
    spd = tensor['spd']; thr = ctx.thr; valid = ~np.isnan(spd); rows = np.arange(W)[:, None]
    out = {}
    def back(a):
        o = np.full((W, L), np.nan, np.float32); o[:, ch] = a; return o
    def bcast(v):
        return np.broadcast_to(np.asarray(v, np.float32)[:, None], (W, L)).copy()
    def jam_at(s):
        val = last_valid(spd[..., :s + 1], valid[..., :s + 1])[2]
        return np.nan_to_num(val, nan=np.inf) <= thr[None, :]
    jams = {'now': np.nan_to_num(wl['last_ratio'], nan=np.inf) <= 1.0, 'm15': jam_at(9), 'm30': jam_at(6)}
    dd = {}
    for key, jam in jams.items():
        jc = jam[:, ch]
        nxt = np.minimum.accumulate(np.where(jc, ar, BIG)[:, ::-1], axis=1)[:, ::-1]
        prv = np.maximum.accumulate(np.where(jc, ar, -BIG), axis=1)
        dn = np.where(nxt < BIG, nxt - ar, np.nan).astype(np.float32)
        up = np.where(prv > -BIG, ar - prv, np.nan).astype(np.float32)
        dd[key] = (dn, up, nxt, prv)
        out[f'jd_dn_{key}'] = back(dn); out[f'jd_up_{key}'] = back(up)
        starts = (jc[:, 1:] & ~jc[:, :-1]).sum(1) + jc[:, 0]
        out[f'n_jam_{key}'] = bcast(jc.sum(1)); out[f'n_clu_{key}'] = bcast(starts)
    for k in ('15', '30'):
        out[f'jd_dn_d{k}'] = back(dd['now'][0] - dd[f'm{k}'][0]); out[f'jd_up_d{k}'] = back(dd['now'][1] - dd[f'm{k}'][1])
    out['n_jam_d30'] = out['n_jam_now'] - out['n_jam_m30']
    nxt, prv = dd['now'][2], dd['now'][3]
    for name in ('last_ratio', 'trend_15', 'run_state_min', 'flow_over_cap_last', 'occ_trend_15'):
        v = wl[name][rows, ch[np.clip(nxt, 0, n - 1)]]
        out[f'nj_dn_{name}'] = back(np.where(nxt >= BIG, np.nan, v))
    for name in ('last_ratio', 'trend_15', 'run_state_min'):
        v = wl[name][rows, ch[np.clip(prv, 0, n - 1)]]
        out[f'nj_up_{name}'] = back(np.where(prv <= -BIG, np.nan, v))
    jslot = np.where(valid, spd, np.inf) <= thr[None, :, None]
    anyj = jslot.any(-1); sl = np.arange(spd.shape[-1])[None, None, :]; last = spd.shape[-1] - 1
    lj = last - np.where(jslot, sl, -1).max(-1); fj = last - np.where(jslot, sl, 99).min(-1)
    out['min_since_jam'] = np.where(anyj, lj * 5, np.nan).astype(np.float32)
    out['jam_span_min'] = np.where(anyj, (fj - lj) * 5, np.nan).astype(np.float32)
    out['jam_slots_n'] = jslot.sum(-1).astype(np.float32)
    out['recovering'] = ((np.nan_to_num(wl['trend_15'], nan=0) > 0) & (np.nan_to_num(wl['last_ratio'], nan=9) <= 1.25)).astype(np.float32)
    isb = np.zeros(n, bool); isb[list(bn_set)] = True
    nb = np.minimum.accumulate(np.where(isb, np.arange(n), BIG)[::-1])[::-1]; okb = nb < BIG
    nbl = ch[np.clip(nb, 0, n - 1)]
    out['bn_dist'] = back(np.broadcast_to(np.where(okb, nb - np.arange(n), np.nan), (W, n)))
    out['bn_rate_self'] = back(np.broadcast_to(head_rate.astype(np.float32), (W, n)))
    out['bn_rate_next'] = back(np.broadcast_to(np.where(okb, head_rate[np.clip(nb, 0, n - 1)], np.nan), (W, n)))
    for name in ('last_ratio', 'trend_15', 'flow_over_cap_last', 'occ_trend_15', 'flow_trend_15', 'run_state_min', 'qoff_last'):
        out[f'bn_{name}'] = back(np.where(okb[None, :], wl[name][:, nbl], np.nan))
    return {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}
