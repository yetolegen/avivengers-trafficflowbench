# v37: CNN onset model along the link chain (own design). Per onset window, a 1-D conv net over the corridor's links
# (ordered along the chain from chain_of) reads the tree features of each link at T+30 plus the raw speed/threshold ratio
# and observed flag of every chain link over the last 60 min and T, and predicts P(queued at T+30) per link.
# Causal: same inputs the trees get (nothing after T). Ongoing windows are untouched.
NS = 13  # 12 history slots + T

def posmaps():
    chs = {PC[p]: STATE[p]['ch'] for p in PANELS}
    lall = max(STATE[p]['ctx'].L for p in PANELS)
    pm = np.full((len(PANELS), lall), -1, np.int64)
    for k, c in chs.items():
        pm[k, c] = np.arange(len(c))
    return pm, chs, max(len(c) for c in chs.values())

def norm_fit(A):
    med = np.nan_to_num(np.nanmedian(A, 0)); sc = np.nanstd(A, 0)
    return med, np.where(np.isfinite(sc) & (sc > 1e-6), sc, 1.0)

def norm_apply(A, ms):
    return np.clip(np.nan_to_num((A - ms[0]) / ms[1], nan=0.0), -5, 5).astype(np.float32)

def make_grid(gid, pan, l, Fm, raw, POS, CHS, LM):
    """rows (window gid, panel, link, features) -> (windows, channels, chain position) grid."""
    ug, wi = np.unique(gid, return_inverse=True)
    nf = Fm.shape[1]
    G = np.zeros((len(ug), nf + 2 * NS + 1, LM), np.float32)
    pos = POS[pan, l]; k = pos >= 0
    G[wi[k], :nf, pos[k]] = Fm[k]
    pan_w = np.zeros(len(ug), np.int64); pan_w[wi] = pan
    for j, g in enumerate(ug):
        ch = CHS[int(pan_w[j])]; r = raw[int(g)][ch]
        G[j, nf:nf + NS, :len(ch)] = np.clip(np.nan_to_num(r, nan=0.0), 0, 2).T
        G[j, nf + NS:nf + 2 * NS, :len(ch)] = (~np.isnan(r)).T
        G[j, -1, :len(ch)] = 1
    return G, wi, pos, k

# ---- torch part
import torch, torch.nn as nn, torch.nn.functional as F_
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), flush=True)
CNN_EPOCHS, CNN_SEEDS, CNN_WIDTH = 30, 3, 64

class ChainNet(nn.Module):
    def __init__(self, c, w=CNN_WIDTH):
        super().__init__()
        self.inp = nn.Conv1d(c, w, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(w, w, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8)])
        self.out = nn.Conv1d(w, 1, 1)
    def forward(self, x):
        h = F_.gelu(self.inp(x))
        for b in self.blocks:
            h = h + F_.gelu(b(F_.dropout(h, 0.1, self.training)))
        return self.out(h)[:, 0]

def cnn_fit(G, Y, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = ChainNet(G.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CNN_EPOCHS)
    Gt = torch.from_numpy(G); Yt = torch.from_numpy(np.nan_to_num(Y, nan=0.0)); Mt = torch.from_numpy(~np.isnan(Y))
    for ep in range(CNN_EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(G)), max(1, len(G) // 64)):
            b = torch.from_numpy(b); m = Mt[b].to(DEV)
            loss = F_.binary_cross_entropy_with_logits(net(Gt[b].to(DEV))[m], Yt[b].to(DEV)[m])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net.eval()

def cnn_predict(nets, G):
    P = []
    with torch.no_grad():
        for net in nets:
            P.append(np.concatenate([torch.sigmoid(net(torch.from_numpy(G[i:i + 256]).to(DEV))).cpu().numpy()
                                     for i in range(0, len(G), 256)]))
    return np.mean(P, 0)

def with_tp(Fn, tp):
    """append the tree onset probability (as a logit) as one more per-link channel"""
    if tp is None:
        return Fn
    p = np.clip(np.asarray(tp, np.float32), 1e-4, 1 - 1e-4)
    return np.c_[Fn, np.log(p / (1 - p)) / 4].astype(np.float32)

def onset_oof(mask, k=4):
    """out-of-fold tree onset probability (folds by day) for the onset rows in mask; never in-sample"""
    idx = np.flatnonzero(mask & onset_rows); fold = XF['_day'].to_numpy()[idx] % k
    p = np.full(len(XF), np.nan, np.float32)
    for f in range(k):
        tr = np.zeros(len(XF), bool); tr[idx[fold != f]] = True
        te = idx[fold == f]; p[te] = onset_prob(fit_onset(CA, tr), XF.iloc[te], CA)
    return p

def cnn_train(rows_mask, tp=None):
    t0 = time.time(); A = XF.loc[rows_mask]; Fm = A[CA].to_numpy(np.float32); ms = norm_fit(Fm)
    G, wi, pos, k = make_grid(A['_gid'].to_numpy(np.int64), A['_pan'].to_numpy(np.int64), A['_l'].to_numpy(np.int64),
                              with_tp(norm_apply(Fm, ms), None if tp is None else tp[rows_mask]), RAW, POS, CHS, LMAX)
    Y = np.full((G.shape[0], LMAX), np.nan, np.float32); y = yF[rows_mask]; Y[wi[k], pos[k]] = y[k]
    nets = [cnn_fit(G, Y, 2026 + s) for s in range(CNN_SEEDS)]
    print('cnn trained on', G.shape[0], 'windows,', int((~np.isnan(Y)).sum()), 'cells, positive share',
          round(float(np.nanmean(Y)), 4), '| sec', round(time.time() - t0), flush=True)
    return nets, ms

def cnn_rows(model, A, gid, pan, l, raw, tp=None):
    nets, ms = model
    G, wi, pos, k = make_grid(gid, pan, l, with_tp(norm_apply(A[CA].to_numpy(np.float32), ms), tp), raw, POS, CHS, LMAX)
    P = cnn_predict(nets, G); out = np.full(len(gid), np.nan, np.float32); out[k] = P[wi[k], pos[k]]
    return out

