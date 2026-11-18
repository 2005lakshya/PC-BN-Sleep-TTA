"""Test-time adaptation of the source CNN to one unlabeled night.

Every method starts from a fresh copy of the source model (episodic: nothing carries over
from one patient to the next) and first swaps the BatchNorm statistics for the night's own
(`bn`, a.k.a. AdaBN). Gradient methods then update only the BatchNorm affine parameters,
on batches made of a few random runs of consecutive epochs, with these objectives:

    tent       mean entropy                                     (Wang et al. 2021)
    shot_im    mean entropy - entropy of the mean prediction    (SHOT-IM, Liang et al. 2020)
               -> pushes the night's stage mix toward uniform
    ent_prior  mean entropy + KL(pi_src || mean prediction)
               -> pushes the night's stage mix toward the source mix
    jump       ours: mean entropy + KL(J_src || J_pred), row by row. J_pred is the soft
               jump chain of the predictions: how often stage i is followed by a different
               stage j, normalised over j != i. It ignores how long stages last, so it
               does not fight a night whose stage mix differs from the source.
    hmm_teach  ours: temporal-prior self-training. Every few steps the whole night is decoded
               with the source transition matrix (tta.hmm, `hmm_src`), and the smoothed
               posteriors become soft targets for cross-entropy. This is EM: the HMM is the
               E-step, the BN update the M-step. It leans on the finding (study_dwell.py)
               that the source stage dynamics fit target nights almost as well as their own.

Gradient-free normalization variants (no BN affine updates):
    alpha_bn   alpha * source statistics + (1 - alpha) * night statistics
    pcbn       ours, prior-corrected BN: per-stage night statistics (soft labels from the
               HMM decoder) recombined with the source stage mix; see `balanced_bn`
    pcbn_iid   same, soft labels straight from the CNN (no temporal smoothing)
    ubbn       same, uniform stage weights (TRIBE-style balanced BN)
"""
import copy

import numpy as np
import torch
from torch import nn

from tta.hmm import decode, split_dwell

EPS = 1e-6


@torch.no_grad()
def probs_of(model, x, batch=1024):
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + batch]), -1) for i in range(0, len(x), batch)])


@torch.no_grad()
def night_bn(model, x, batch=1024):
    """Replace BN running statistics with this night's statistics (exact, not momentum).
    Only the BN layers go to train mode: dropout stays off, as it is when predicting."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None                     # cumulative average over all batches
            m.train()
    for i in range(0, len(x), batch):
        model.encoder(x[i:i + batch])
    model.eval()


@torch.no_grad()
def balanced_bn(model, x, resp, weights, tau=20.0, batch=1024):
    """Night BN statistics with the night's own stage mix taken out.

    A night's plain BN statistics are a mixture over stages, mean = sum_c pi_night(c) mu_c,
    so a night with little N3 shifts them even if the recording is perfect. The source
    running statistics were a mixture under pi_src. Here per-stage statistics mu_c and
    E[x^2]_c are estimated on the night with soft stage labels `resp` (n, k), then
    recombined with `weights` (pi_src = prior-corrected BN; uniform = TRIBE-style balanced
    BN). `tau` pseudo-epochs shrink a rare stage's estimate toward the night average.
    Layers are set one at a time, since each layer's input depends on the ones before."""
    r = torch.tensor(resp, dtype=torch.float32, device=x.device)
    w = torch.tensor(weights, dtype=torch.float32, device=x.device)
    model.eval()
    for bn in [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]:
        means, sq = [], []

        def grab(mod, inp, out):
            means.append(inp[0].mean(-1))
            sq.append((inp[0] ** 2).mean(-1))

        hook = bn.register_forward_hook(grab)
        for i in range(0, len(x), batch):
            model.encoder(x[i:i + batch])
        hook.remove()
        m, s = torch.cat(means), torch.cat(sq)                     # (n, C)
        mass = r.sum(0)[:, None]                                   # (k, 1)
        mu_c = (r.T @ m + tau * m.mean(0)) / (mass + tau)          # (k, C)
        sq_c = (r.T @ s + tau * s.mean(0)) / (mass + tau)
        mu = w @ mu_c
        bn.running_mean.copy_(mu)
        bn.running_var.copy_((w @ sq_c - mu ** 2).clamp_min(1e-5))


def entropy(p):
    return -(p * (p + EPS).log()).sum(-1)


def soft_jump_chain(p_runs):
    """p_runs: list of (L, k) probability runs in time order -> (k, k) soft jump chain and
    the share of jumps leaving each stage (used to weight the rows)."""
    C = sum(p[:-1].T @ p[1:] for p in p_runs)
    off = C * (1 - torch.eye(C.shape[0], device=C.device))
    out = off.sum(1)
    return off / (out[:, None] + EPS), out / (out.sum() + EPS)


def loss_fn(method, p_runs, prior_src, J_src, lam):
    p = torch.cat(p_runs)
    ent = entropy(p).mean()
    if method == "tent":
        return ent
    pbar = p.mean(0)
    if method == "shot_im":
        return ent - entropy(pbar)
    if method == "ent_prior":
        return ent + (prior_src * (prior_src.log() - (pbar + EPS).log())).sum()
    if method == "jump":
        J, w = soft_jump_chain(p_runs)
        kl = (J_src * (J_src.log() - (J + EPS).log())).sum(1)
        return ent + lam * (w.detach() * kl).sum()
    raise ValueError(method)


def adapt(source_model, x, method, prior_src, A_src, steps=30, lr=1e-3, runs=4, run_len=64,
          lam=1.0, seed=0, segments=None, refresh=5, rounds=2, tau=20.0, alpha=0.5):
    """Returns the night's (n, k) probabilities after adapting with `method`."""
    model = copy.deepcopy(source_model)
    if method == "source":
        return probs_of(model, x)
    src_stats = [(m.running_mean.clone(), m.running_var.clone())
                 for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    night_bn(model, x)
    if method == "bn":
        return probs_of(model, x)
    if method == "alpha_bn":                  # alpha-BN (You et al. 2021): mix source and night
        bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        for m, (mu, var) in zip(bns, src_stats):
            m.running_mean.mul_(1 - alpha).add_(alpha * mu)
            m.running_var.mul_(1 - alpha).add_(alpha * var)
        return probs_of(model, x)
    if method in BALANCED:
        segments = segments if segments is not None else [np.arange(len(x))]
        weights = np.full(len(prior_src), 1 / len(prior_src)) if method == "ubbn" else prior_src
        for _ in range(rounds):
            p = probs_of(model, x).cpu().numpy().astype(np.float64)
            resp = decode(p, prior_src, A_src, segments, BALANCED[method])
            balanced_bn(model, x, resp, weights, tau)
        return probs_of(model, x)

    dev = x.device
    prior = torch.tensor(prior_src, dtype=torch.float32, device=dev)
    J_src = torch.tensor(split_dwell(A_src)[1], dtype=torch.float32, device=dev)
    J_src = J_src + EPS * (1 - torch.eye(len(J_src), device=dev))
    params = model.bn_params()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    n = len(x)
    L = min(run_len, n)
    segments = segments if segments is not None else [np.arange(n)]
    for step in range(steps):
        starts = rng.integers(0, n - L + 1, size=runs)
        if method == "hmm_teach" and step % refresh == 0:
            q = decode(probs_of(model, x).cpu().numpy().astype(np.float64), prior_src, A_src,
                       segments, "hmm_src")
            q = torch.tensor(q, dtype=torch.float32, device=dev)
        p_runs = [torch.softmax(model(x[s:s + L]), -1) for s in starts]   # BN in eval mode
        if method == "hmm_teach":
            target = torch.cat([q[s:s + L] for s in starts])
            loss = -(target * (torch.cat(p_runs) + EPS).log()).sum(1).mean()
        else:
            loss = loss_fn(method, p_runs, prior, J_src, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probs_of(model, x)


METHODS = ["source", "bn", "tent", "shot_im", "ent_prior", "jump", "hmm_teach"]
# balanced BN variants -> decoder that supplies their soft stage labels
BALANCED = {"pcbn": "hmm_src",        # ours: source-prior weights, temporal soft labels
            "pcbn_iid": "argmax",     # ablation: per-epoch soft labels, no temporal prior
            "ubbn": "hmm_src"}        # ablation: uniform weights (TRIBE-style balance)
METHODS += list(BALANCED) + ["alpha_bn"]
